//! symir — a symbolic interpreter over the printed PyPTO IR.
//!
//! Parses the DSL's printed form (`.as_python()` output, pass dumps) with a
//! PEG grammar and symbolically interprets one function to extract a DAG of
//! symbolic tasks:
//!
//! * loops introduce an uninterpreted bounded variable (`muN`), one node
//!   for sequential loops, two nodes (`muNa != muNb`) for parallel loops;
//! * every dependency edge the explicit code establishes — `deps=` lists
//!   and region-level read/write dataflow — is present in the DAG.
//!
//! Lint policy: servyi/lints (no hand-rolled parsing primitives — the
//! grammar is pest; no lint tables in manifests; the classic flag list runs
//! in CI).

// Tests are the sanctioned exemption for panic/unwrap (servyi
// `non_test_panic_allow`).
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::panic))]

pub mod ast;
pub mod consts;
pub mod dag;
pub mod emit;
pub mod error;
pub mod interp;
pub mod msgs;
pub mod parse;
pub mod sym;

pub use dag::TaskDag;
pub use error::SymirError;

/// Parses and interprets one printed program in a single call.
pub fn extract(src: &str, entry: Option<&str>) -> Result<TaskDag, SymirError> {
    let program = parse::parse_program(src)?;
    interp::interpret(&program, entry)
}
