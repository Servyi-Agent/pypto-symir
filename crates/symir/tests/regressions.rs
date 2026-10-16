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
