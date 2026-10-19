// Test expectations read better inline; the servyi `inline_literal` policy
// sanctions silencing sites with an explicit allow.
#![allow(unknown_lints)] // stable toolchain: the servyi lints exist only under the driver
#![allow(inline_literal)]
// Tests are the sanctioned exemption for panic/unwrap (servyi
// `non_test_panic_allow`); this file is an integration-test crate.
#![allow(clippy::unwrap_used, clippy::panic)]

//! Integration tests: real PyPTO-printed fixtures through the full pipeline.

use symir::dag::{Constraint, EdgeKind, NodeKind};
use symir::extract;

const RANGE_CARRY_PRE: &str = include_str!("fixtures/range_carry_pre.py");
const RANGE_CARRY_POST: &str = include_str!("fixtures/range_carry_post.py");
const PARALLEL_TASK_POST: &str = include_str!("fixtures/parallel_task_post.py");
const DEPS_TASKS: &str = include_str!("fixtures/deps_tasks.py");
const PARALLEL_CARRY: &str = include_str!("fixtures/parallel_carry.py");
const WHILE_LOOP: &str = include_str!("fixtures/while_loop.py");

#[test]
fn sequential_loop_is_one_node_with_bounds() {
    let dag = extract(RANGE_CARRY_PRE, None).unwrap();
    let seq_loops = dag
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::SeqLoop))
        .count();
    assert_eq!(seq_loops, 1, "one node for a sequential loop");
    assert_eq!(
        dag.constraints.len(),
        1,
        "exactly one constraint (the bounds)"
    );
    match &dag.constraints[0] {
        Constraint::Bounds { var, lo, hi, .. } => {
            assert_eq!(var, "mu0");
            assert_eq!(lo, "0");
            assert_eq!(hi, "4");
        }
        other => panic!("expected bounds, got {other:?}"),
    }
    let loop_node = dag
        .nodes
        .iter()
        .find(|n| matches!(n.kind, NodeKind::SeqLoop))
        .unwrap();
    assert!(
        !loop_node.reads.is_empty(),
        "the loop body reads a region (x)"
    );
    // The write to `out` happens after the loop, at host level: the loop
    // node itself only reads in this fixture.
    let host_writes_out = dag
        .nodes
        .iter()
        .any(|n| matches!(n.kind, NodeKind::Host) && !n.writes.is_empty());
    assert!(host_writes_out, "the host writes out after the loop");
}

#[test]
fn post_pass_form_parses_and_matches_pre_pass_shape() {
    let dag = extract(RANGE_CARRY_POST, None).unwrap();
    let seq_loops = dag
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::SeqLoop))
        .count();
    assert_eq!(seq_loops, 1);
    assert_eq!(dag.constraints.len(), 1);
}

#[test]
fn parallel_loop_is_two_nodes_with_distinct_constraint() {
    let dag = extract(PARALLEL_TASK_POST, None).unwrap();
    let instances = dag
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::ParInstance { .. }))
        .count();
    assert_eq!(instances, 2, "two nodes for a parallel loop");
    let distinct = dag
        .constraints
        .iter()
        .filter(|c| matches!(c, Constraint::Distinct { .. }))
        .count();
    assert_eq!(distinct, 1, "one mu_a != mu_b constraint");
    match dag
        .constraints
        .iter()
        .find(|c| matches!(c, Constraint::Distinct { .. }))
        .unwrap()
    {
        Constraint::Distinct { a, b, .. } => {
            assert_eq!(a, "mu0a");
            assert_eq!(b, "mu0b");
        }
        other => panic!("expected distinct, got {other:?}"),
    }
    // The InCore call inside the loop is a task executing under mu0.
    let task = dag
        .nodes
        .iter()
        .find(|n| matches!(n.kind, NodeKind::Task { .. }))
        .unwrap();
    assert_eq!(task.iter_ctx.len(), 1);
    assert_eq!(task.iter_ctx[0].var, "mu0");
    assert_eq!(task.parent_loop, Some(0));
}

#[test]
fn explicit_deps_and_raw_both_present() {
    let dag = extract(DEPS_TASKS, None).unwrap();
    let tasks = dag
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Task { .. }))
        .collect::<Vec<_>>();
    assert_eq!(tasks.len(), 2, "stage_a and stage_b submits");
    let dep_edges = dag.edges.iter().filter(|e| e.kind == EdgeKind::Dep).count();
    assert_eq!(dep_edges, 1, "the deps=[a_tid] edge");
    let raw_edges = dag.edges.iter().filter(|e| e.kind == EdgeKind::Raw).count();
    assert_eq!(raw_edges, 1, "stage_b reads what stage_a wrote (mid)");
    // Both edges connect the same pair a -> b.
    for edge in &dag.edges {
        assert_eq!(edge.from, tasks[0].id);
        assert_eq!(edge.to, tasks[1].id);
    }
}

#[test]
fn parallel_carry_is_a_race_warning() {
    let dag = extract(PARALLEL_CARRY, None).unwrap();
    assert_eq!(
        dag.warnings.len(),
        2,
        "one warning per parallel instance node"
    );
    let distinct = dag
        .constraints
        .iter()
        .filter(|c| matches!(c, Constraint::Distinct { .. }))
        .count();
    assert_eq!(distinct, 1);
    // The post-loop assemble reads what the loop wrote: a Raw edge from
    // the loop instances to the host.
    let raw_to_host = dag
        .edges
        .iter()
        .any(|e| e.kind == EdgeKind::Raw && e.to == 0);
    assert!(raw_to_host, "host reads the loop-carried region after");
}

#[test]
fn while_loop_gets_symbolic_trip_count() {
    let dag = extract(WHILE_LOOP, None).unwrap();
    match dag.constraints.first().unwrap() {
        Constraint::Bounds { var, lo, hi, .. } => {
            assert_eq!(var, "mu0");
            assert_eq!(lo, "0");
            assert!(
                hi.contains("trip_while"),
                "uninterpreted trip count, got {hi}"
            );
        }
        other => panic!("expected bounds, got {other:?}"),
    }
}

#[test]
fn emit_dot_and_json_render() {
    let dag = extract(DEPS_TASKS, None).unwrap();
    let dot = symir::graph::build(&dag).to_dot(&dag.program);
    assert!(dot.contains("digraph"));
    assert!(dot.contains("->"));
    let json = serde_json::to_string(&dag).unwrap();
    assert!(json.contains("\"edges\""));
}

const DEEPSEEK_INDEXER: &str = include_str!("fixtures/deepseek_indexer_pre.py");

#[test]
fn real_deepseek_indexer_extracts_a_rich_dag() {
    let dag = extract(DEEPSEEK_INDEXER, None).unwrap();
    // 428-line real dump: host + spmd tasks + the loop nest.
    assert!(dag.nodes.len() > 20, "rich DAG, got {}", dag.nodes.len());
    // After loop-ancestor edge suppression (ordering inside one loop
    // body is by construction), the indexer's cross-section guarantees
    // are the war[output] edges from the host's post-hoc read plus the
    // section edges — a handful, all real.
    assert!(
        dag.edges.len() >= 3,
        "cross-section edges, got {}",
        dag.edges.len()
    );
    assert!(
        dag.constraints.len() > 20,
        "loop bounds constraints, got {}",
        dag.constraints.len()
    );
    // The worker-strided SPMD loop has an uninterpreted lower bound (the
    // block index) — the signature of the strided-work idiom.
    let strided = dag.constraints.iter().any(|c| match c {
        Constraint::Bounds { lo, .. } => lo.contains("get_block_idx"),
        _ => false,
    });
    assert!(strided, "worker-strided loop bounds present");
    // At least one spmd task node exists.
    let spmd_tasks = dag.nodes.iter().any(|n| match &n.kind {
        NodeKind::Task { callee } => callee.contains("spmd"),
        _ => false,
    });
    assert!(spmd_tasks, "spmd task present");
}
