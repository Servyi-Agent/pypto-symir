//! pypto-ir — the frontend for the printed PyPTO IR.
//!
//! Parses the DSL's printed form (`.as_python()` output, pass dumps) with a
//! PEG grammar ([pypto_ir.pest], an off-the-shelf parser generator — no
//! hand-rolled parsing) into [AST][ast] types shared by every tool that
//! reasons over printed PyPTO programs.
//!
//! Lint policy: servyi/lints (no hand-rolled parsing primitives — the
//! grammar is pest; no lint tables in manifests; the classic flag list runs
//! in CI).

pub mod ast;
pub mod consts;
pub mod error;
pub mod msgs;
pub mod parse;

pub use error::IrError;

/// Parses one printed program into an [`ast::Program`].
pub fn parse_program(src: &str) -> Result<ast::Program, IrError> {
    parse::parse_program(src)
}
