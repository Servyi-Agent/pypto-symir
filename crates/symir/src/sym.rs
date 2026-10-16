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

/// The base of a view chain: the first argument when `name` is a
/// view-forming op.
fn view_base<'a>(name: &str, args: &'a [SymVal]) -> Option<&'a SymVal> {
    if crate::consts::VIEW_FORMING_OPS.contains(&name) {
        args.first()
    } else {
        None
    }
}

impl SymVal {
    /// Collects the regions a value denotes AS STORAGE — for destinations.
    ///
    /// The mirror image of [`SymVal::regions`]: reads collect every region
    /// mentioned anywhere (terms are read-transparent), but a destination
    /// is only the base region a view chain derives from. Treating reads
    /// and destinations the same way turns a read embedded in a view term
    /// into a write (observed on the real slicing dump: two assembles
    /// writing disjoint halves of `out` reported `W:[x, out]`).
    ///
    /// View-forming ops keep their first argument's destination; computed
    /// values (arith results) denote no storage; tuples union their
    /// elements' destinations.
    pub fn dest_regions(&self, out: &mut Vec<RegionId>) {
        match self {
            SymVal::Region(id) => out.push(*id),
            SymVal::Tuple(items) => {
                for item in items {
                    item.dest_regions(out);
                }
            }
            SymVal::Uninterp { name, args } => {
                if let Some(base) = view_base(name, args) {
                    base.dest_regions(out);
                }
            }
            _ => {}
        }
    }

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
