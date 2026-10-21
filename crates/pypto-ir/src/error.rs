//! Errors raised while turning printed PyPTO IR into [`crate::ast::Program`].

use thiserror::Error;

/// Everything that can go wrong in [`crate::parse::parse_program`].
#[derive(Debug, Error)]
pub enum IrError {
    /// The grammar rejected the input.
    #[error("parse error: {0}")]
    Parse(String),
    /// The grammar accepted the file, but block structure or a construct is
    /// outside the supported subset.
    #[error("unsupported IR construct at line {line}: {message}")]
    Unsupported {
        /// 1-based source line.
        line: usize,
        /// What was rejected.
        message: String,
    },
}
