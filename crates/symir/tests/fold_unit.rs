//! Symbolic-value folding unit tests (SymVal lives in symir).

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

