//! Error type for parsing and interpretation failures.

use pypto_ir::IrError;
use thiserror::Error;

/// Everything that can go wrong in [`crate::parse_program`] or
/// [`crate::interp`].
#[derive(Debug, Error)]
pub enum SymirError {
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
    /// The interpreter hit a form it cannot give meaning to.
    #[error("interpretation error at line {line}: {message}")]
    Interp {
        /// 1-based source line.
        line: usize,
        /// What failed.
        message: String,
    },
}

impl From<IrError> for SymirError {
    fn from(e: IrError) -> Self {
        match e {
            IrError::Parse(m) => SymirError::Parse(m),
            IrError::Unsupported { line, message } => SymirError::Unsupported { line, message },
        }
    }
}
