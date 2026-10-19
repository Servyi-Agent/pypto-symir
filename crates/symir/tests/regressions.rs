// Reproducing tests for observed regressions. Each was demonstrated red
// against the pre-fix code (commit 6a7aa0f) and is fixed by the follow-up
// commit — the "committed red, fixed by the fix" discipline.
//
// The servyi `inline_literal` policy sanctions silencing sites with an
// explicit allow; test expectations read better inline.
#![allow(unknown_lints)] // stable toolchain: the servyi lints exist only under the driver
#![allow(inline_literal)]
// Tests are the sanctioned exemption for panic/unwrap (servyi
// `non_test_panic_allow`); this file is an integration-test crate.
#![allow(clippy::unwrap_used, clippy::panic)]

use symir::dag::EdgeKind;
use symir::extract;

const PARALLEL_CARRY: &str = include_str!("fixtures/parallel_carry.py");
const WHILE_LOOP: &str = include_str!("fixtures/while_loop.py");

/// Observed 2026-10-15: post-loop roll-up emitted the same Raw edge three
/// times (host read recorded per body statement), so `parallel_carry` DOT
/// showed `n1 -> n0 [raw]` three times over.
#[test]
fn no_duplicate_edges_after_loop_rollup() {
    for (name, src) in [("parallel_carry", PARALLEL_CARRY), ("while", WHILE_LOOP)] {
        let dag = extract(src, None).unwrap();
        // Runtime assertion layer: validate() rejects duplicates outright.
        dag.validate().unwrap();
        // And directly: no two edges share (from, to, kind, regions).
        for (i, a) in dag.edges.iter().enumerate() {
            for b in dag.edges.iter().skip(i + 1) {
                assert!(
                    !(a.from == b.from
                        && a.to == b.to
                        && a.kind == b.kind
                        && a.regions == b.regions),
                    "{name}: duplicate edge n{} -> n{} ({:?})",
                    a.from,
                    a.to,
                    a.kind
                );
            }
        }
    }
}

/// Observed 2026-10-16: the parallel-carry warning existed, but the race
/// itself was invisible in the graph — no edge between the two unordered
/// instances. A race must be an edge, not just prose.
#[test]
fn parallel_carry_is_a_race_edge_not_just_a_warning() {
    let dag = extract(PARALLEL_CARRY, None).unwrap();
    let race_edges: Vec<_> = dag
        .edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Race)
        .collect();
    assert_eq!(
        race_edges.len(),
        1,
        "exactly one race edge between mu0a and mu0b"
    );
    let race = race_edges.first().unwrap();
    let instances: Vec<_> = dag
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, symir::dag::NodeKind::ParInstance { .. }))
        .map(|n| n.id)
        .collect();
    assert!(
        instances.contains(&race.from) && instances.contains(&race.to),
        "the race edge connects the two parallel instances, got {} -> {}",
        race.from,
        race.to
    );
}

/// The validator itself must catch what it exists to catch: a hand-broken
/// DAG with a duplicate edge fails validate().
#[test]
fn validator_rejects_duplicate_edges() {
    let mut dag = extract(PARALLEL_CARRY, None).unwrap();
    let clone = dag.edges.first().cloned().unwrap();
    dag.edges.push(clone);
    assert!(dag.validate().is_err(), "duplicate edge must be rejected");
}

/// The validator rejects dangling endpoints.
#[test]
fn validator_rejects_dangling_endpoints() {
    let mut dag = extract(PARALLEL_CARRY, None).unwrap();
    if let Some(edge) = dag.edges.first_mut() {
        edge.to = dag.nodes.len();
    }
    assert!(dag.validate().is_err(), "dangling edge must be rejected");
}

const SLICING: &str = include_str!("fixtures/slicing.py");

/// Observed 2026-10-16 on the real slicing dump: two `assemble` calls
/// writing disjoint halves of `out` reported `W:[x, out]` — the first
/// assemble's result term embeds the *read* of `x`, and when that term
/// became the second assemble's destination, read-transparency turned it
/// into a write. Destinations need their own extraction: only the base
/// region a view chain derives from, never the reads nested in the term.
#[test]
fn destination_extraction_never_turns_reads_into_writes() {
    let dag = extract(SLICING, None).unwrap();
    let host = &dag.nodes[0];
    let out_region = dag
        .regions
        .iter()
        .position(|r| r == "out")
        .unwrap_or_default();
    let x_region = dag
        .regions
        .iter()
        .position(|r| r == "x")
        .unwrap_or_default();
    assert!(
        host.writes.contains(&out_region),
        "out is written by both assembles"
    );
    assert!(
        !host.writes.contains(&x_region),
        "x is only ever read (sliced reads), never written"
    );
    assert!(host.reads.contains(&x_region));
}

const SLICING_POST: &str = include_str!("fixtures/slicing_post.py");

/// Observed 2026-10-16 on the real post-pass slicing dump: `_dump.call`
/// carries the op name as a *string* argument; the extractor read it with
/// `expr_text`, which returns "" for strings, so `tile.store` fell to the
/// read-only default and the second kernel's write vanished (`W:[]`).
#[test]
fn dump_call_string_op_names_classify() {
    let dag = extract(SLICING_POST, None).unwrap();
    let out_region = dag
        .regions
        .iter()
        .position(|r| r == "out__ssa_v0")
        .unwrap_or_default();
    for kernel in ["slicing_incore_0", "slicing_incore_1"] {
        let node = dag
            .nodes
            .iter()
            .find(|n| matches!(&n.kind, symir::dag::NodeKind::Task { callee } if callee == kernel))
            .unwrap_or_else(|| panic!("{kernel} task present"));
        assert!(
            node.writes.contains(&out_region),
            "{kernel} writes out through tile.store, got W:{:?}",
            node.writes
        );
    }
}

const DEEPSEEK_GATE: &str = include_str!("fixtures/deepseek_gate.py");

/// Observed 2026-10-16 on pypto-lib's real DeepSeek MoE gate dump: the
/// printer emits scientific-notation floats (`value=-3.4028234663852886e+38`)
/// and unparenthesized multi-value returns (`return a, b, c, d`). Both are
/// real printed-IR forms the grammar lacked; this fixture failed to parse
/// before the fix (the parse failure was the red demonstration).
#[test]
fn deepseek_gate_fixture_parses_and_extracts() {
    let dag = extract(DEEPSEEK_GATE, None).unwrap();
    assert!(dag.nodes.len() > 20, "rich DAG, got {}", dag.nodes.len());
    assert!(dag.constraints.len() > 20, "many loop bounds");
    // The gate's chunk loop bound is a real symbolic term over a tensor dim.
    assert!(
        dag.constraints
            .iter()
            .any(|c| matches!(c, symir::dag::Constraint::Bounds { hi, .. } if hi.contains("dim("))),
        "ceil((dim+7)/8)-style bounds present"
    );
}

/// Observed 2026-10-19 auditing the real DeepSeek gate fixture against its
/// source: (1) `pl.write(dst, [i], v)` fell to the read-only default, so
/// the indices/weights zero-fill was recorded as a READ; (2) the printed
/// allocation op is `pl.tensor.create` (leaf `create`) while the alloc
/// table knew only `create_tensor` — every local buffer (xg_buf,
/// route_scores_buf, local_scores, ...) was a term, not a region, and the
/// gate's internal ordering chain was absent from the graph.
#[test]
fn gate_zero_fill_writes_and_local_buffers_are_effects() {
    let dag = extract(DEEPSEEK_GATE, None).unwrap();
    let idx = dag
        .regions
        .iter()
        .position(|r| r == "indices")
        .unwrap_or_default();
    let wgt = dag
        .regions
        .iter()
        .position(|r| r == "weights")
        .unwrap_or_default();
    // The zero-fill loops (mu7/mu8 and mu19/mu20) write, not read.
    for id in [8, 9, 20, 21] {
        let node = &dag.nodes[id];
        assert!(
            node.writes.contains(&idx) && node.writes.contains(&wgt),
            "node {id} must write indices+weights, got R:{:?} W:{:?}",
            node.reads,
            node.writes
        );
    }
    // Local buffers exist as regions, named after their bindings.
    for expected in ["xg_buf", "route_scores_buf", "local_scores"] {
        assert!(
            dag.regions.iter().any(|r| r == expected),
            "local region {expected} present, got {:?}",
            dag.regions
        );
    };
    // The gate's internal chain (xg producers -> consumers) must produce
    // ordering edges far beyond the single 6->18 section edge.
    assert!(
        dag.edges.len() > 12,
        "buffer-chained ordering edges, got {}",
        dag.edges.len()
    );
}
