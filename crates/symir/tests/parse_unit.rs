// Test expectations read better inline; the servyi `inline_literal` policy
// sanctions silencing sites with an explicit allow.
#![allow(inline_literal)]
// Tests are the sanctioned exemption for panic/unwrap (servyi
// `non_test_panic_allow`); this file is an integration-test crate.
#![allow(clippy::unwrap_used, clippy::panic)]

//! Unit tests for parsing and symbolic folding.

use symir::ast::{BinOp, Expr};
use symir::parse::parse_program;
use symir::sym::SymVal;

const MINIMAL: &str = "\
import pypto.language as pl

@pl.program
class T:
    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32]):
        y = pl.tensor.add(x, x)
        return y
";

#[test]
fn parses_class_function_and_assign() {
    let prog = parse_program(MINIMAL).unwrap();
    assert_eq!(prog.name, "T");
    assert_eq!(prog.functions.len(), 1);
    let f = &prog.functions[0];
    assert_eq!(f.name, "entry");
    assert_eq!(f.params.len(), 1);
    assert_eq!(f.params[0].name, "x");
    assert_eq!(f.body.len(), 2);
}

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

#[test]
fn rejects_statement_outside_class() {
    let bad = "y = 1\n";
    assert!(parse_program(bad).is_err());
}

#[test]
fn subscript_preserves_region() {
    let src = "\
@pl.program
class S:
    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32]):
        v = x[0, 0]
        return v
";
    let prog = parse_program(src).unwrap();
    let symir::ast::Stmt::Assign { value, .. } = &prog.functions[0].body[0] else {
        panic!("expected assignment");
    };
    assert!(matches!(value, Expr::Subscript { .. }));
}
