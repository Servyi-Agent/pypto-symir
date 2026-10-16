// Manual-scope mode tracking, tested on REAL compiler dumps (nest_probe).
#![allow(unknown_lints)] // stable toolchain: the servyi lints exist only under the driver
#![allow(inline_literal)]
#![allow(clippy::unwrap_used, clippy::panic)]

use symir::dag::{EdgeKind, NodeKind, ScopeMode};
use symir::extract;

const MANUAL_IN_AUTO: &str = include_str!("fixtures/manual_in_auto.py");
const MANUAL_VIA_INLINE: &str = include_str!("fixtures/manual_via_inline.py");

/// k1 runs in the outer AUTO scope, k2 inside a nested MANUAL scope with no
/// deps= — the runtime skips the OverlapMap for k2, so the k1 -> k2 raw
/// order is established by nothing. symir must (a) record each task's
/// scope mode and (b) warn that the derived edge is unordered.
#[test]
fn manual_scope_unordered_access_is_flagged() {
    for (name, src) in [
        ("manual_in_auto", MANUAL_IN_AUTO),
        ("manual_via_inline", MANUAL_VIA_INLINE),
    ] {
        let dag = extract(src, None).unwrap();
        let k1 = dag
            .nodes
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Task { callee } if callee == "k1"))
            .unwrap();
        let k2 = dag
            .nodes
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Task { callee } if callee == "k2"))
            .unwrap();
        assert_eq!(
            k1.scope_mode,
            ScopeMode::Auto,
            "{name}: k1 launched under auto tracking"
        );
        assert_eq!(
            k2.scope_mode,
            ScopeMode::Manual,
            "{name}: k2 launched under manual scope"
        );
        assert!(
            dag.edges
                .iter()
                .any(|e| e.from == k1.id && e.to == k2.id && e.kind == EdgeKind::Raw),
            "{name}: the dataflow edge is derived"
        );
        assert!(
            !dag.edges
                .iter()
                .any(|e| e.kind == EdgeKind::Dep && e.from == k1.id && e.to == k2.id),
            "{name}: no declared dep in this variant"
        );
        let flagged = dag
            .warnings
            .iter()
            .any(|w| w.node == k2.id && w.message.contains("manual"));
        assert!(
            flagged,
            "{name}: manual-scope unordered access must be flagged"
        );
    }
}

/// The rejected nestings (AUTO-in-MANUAL, MANUAL-in-MANUAL) are compiler
/// errors, not symir inputs; the accepted dump must simply carry the modes.
#[test]
fn accepted_nesting_carries_both_modes() {
    let dag = extract(MANUAL_IN_AUTO, None).unwrap();
    let modes: Vec<_> = dag
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Task { .. }))
        .map(|n| n.scope_mode)
        .collect();
    assert!(modes.contains(&ScopeMode::Auto) && modes.contains(&ScopeMode::Manual));
}
