//! Symbolic values produced by the interpreter.
//!
//! Op calls become uninterpreted function applications; loop induction
//! variables become uninterpreted bounded variables; tensors and buffers
//! become region identities used for effect tracking.

use std::fmt;

use crate::dag::RegionId;

/// Prefix for generated uninterpreted terms.
pub const TERM_PREFIX: &str = "t";
/// Prefix for loop iteration variables.
pub const ITER_VAR_PREFIX: &str = "mu";

/// A symbolic value.
#[derive(Debug, Clone, PartialEq)]
pub enum SymVal {
    /// Concrete integer.
    Int(i64),
    /// Concrete float.
    Float(f64),
    /// Concrete boolean.
    Bool(bool),
    /// String constant (op names, attrs).
    Str(String),
    /// A tuple of values.
    Tuple(Vec<SymVal>),
    /// An uninterpreted function application `name(args...)`.
    Uninterp { name: String, args: Vec<SymVal> },
    /// A memory region (tensor / buffer identity).
    Region(RegionId),
    /// A task id produced by a submit or spmd scope.
    Task { node: usize },
    /// A loop's uninterpreted iteration variable.
    Iter { loop_id: usize, var: String },
    /// No value.
    Unit,
}

impl fmt::Display for SymVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymVal::Int(v) => write!(f, "{v}"),
            SymVal::Float(v) => write!(f, "{v}"),
            SymVal::Bool(v) => write!(f, "{v}"),
            SymVal::Str(v) => write!(f, "{v}"),
            SymVal::Unit => write!(f, "unit"),
            SymVal::Region(id) => write!(f, "R{id}"),
            SymVal::Task { node } => write!(f, "task#{node}"),
            SymVal::Iter { var, .. } => write!(f, "{var}"),
            SymVal::Tuple(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.to_string()).collect();
                write!(f, "({})", parts.join(crate::consts::SEP_COMMA_SPACE))
            }
            SymVal::Uninterp { name, args } => {
                let parts: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                write!(f, "{name}({})", parts.join(crate::consts::SEP_COMMA_SPACE))
            }
        }
    }
}

impl SymVal {
    /// Collects the regions referenced anywhere inside this value.
    pub fn regions(&self, out: &mut Vec<RegionId>) {
        match self {
            SymVal::Region(id) => out.push(*id),
            SymVal::Tuple(items) | SymVal::Uninterp { args: items, .. } => {
                for item in items {
                    item.regions(out);
                }
            }
            _ => {}
        }
    }
}
