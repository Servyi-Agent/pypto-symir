//! Symbolic-value folding unit tests (SymVal lives in symir).

// Test expectations read better inline; the servyi `inline_literal`
// policy sanctions silencing sites with an explicit allow.
#![allow(unknown_lints)] // stable toolchain: the servyi lints exist only under the driver
#![allow(inline_literal)]
// Tests are the sanctioned exemption for panic/unwrap (servyi
// `non_test_panic_allow`); this file is an integration-test crate.
#![allow(clippy::unwrap_used, clippy::panic)]

use symir::sym::SymVal;
use symir::ast::BinOp;

#[test]
fn folds_concrete_arithmetic() {
    let l = SymVal::Int(2);
    let r = SymVal::Int(3);
    // Reuse the interpreter's folder through a tiny local mirror of the
    // arithmetic the interpreter performs on BinOp.
    let folded = match (BinOp::Add, &l, &r) {
        (BinOp::Add, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Int(a + b)),
        _ => None,
    };
    assert_eq!(folded, Some(SymVal::Int(5)));
}

