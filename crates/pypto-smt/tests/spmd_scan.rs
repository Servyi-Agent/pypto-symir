//! spmd scanning and the stride-partition lemma, tested against the real
//! compiler dump and against z3 itself.

// Test expectations read better inline; the servyi `inline_literal`
// policy sanctions silencing sites with an explicit allow.
#![allow(unknown_lints)] // stable toolchain: the servyi lints exist only under the driver
#![allow(inline_literal)]
// Tests are the sanctioned exemption for panic/unwrap (servyi
// `non_test_panic_allow`); this file is an integration-test crate.
#![allow(clippy::unwrap_used, clippy::panic)]

use pypto_ir::parse_program;
use pypto_smt::{find_spmd, stride_partition_is_disjoint};

/// The real pre-pass DeepSeek indexer dump: three spmd sites — `score`
/// (paired_block, width 32), `score__2` (page, width 16) and `score__3`
/// (page, width 32); `score__4` dispatches without an inline spmd body.
#[test]
fn finds_spmd_sites_in_real_indexer_dump() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../symir/tests/fixtures/deepseek_indexer_pre.py"
    ))
    .unwrap();
    let program = parse_program(&src).unwrap();
    let sites = find_spmd(&program);
    assert_eq!(sites.len(), 3, "three inline spmd bodies (score__4 dispatches)");
    assert_eq!(
        sites.iter().map(|s| s.width).collect::<Vec<_>>(),
        vec![32, 16, 32]
    );
    assert!(sites
        .iter()
        .any(|s| s.hint.as_deref().is_some_and(|h| h.contains("paired_block"))));
}

/// Stride == width (the spmd idiom): z3 proves distinct workers never
/// share a work item.
#[test]
fn stride_equal_width_is_disjoint() {
    assert!(stride_partition_is_disjoint(32, 32));
    assert!(stride_partition_is_disjoint(16, 16));
}

/// Negative control: stride < width lets workers collide (w1 + S*k1 can
/// equal w2 with w1 != w2), so the lemma must NOT hold — proving z3 is
/// really solving, not rubber-stamping.
#[test]
fn stride_below_width_collides() {
    assert!(!stride_partition_is_disjoint(32, 8));
}
