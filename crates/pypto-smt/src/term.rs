//! Symbolic terms for the SPMD body walk — solver-neutral (printed by the
//! WhyML emitter; the same trees can drive direct z3 later).

use std::fmt;

/// Integer-valued terms over the scalar algebra of one spmd body.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Concrete integer.
    Int(i64),
    /// A named constant (module dynamic dims, tensor dims).
    Var(String),
    /// Arithmetic; `Div`/`Mod` are Python floor semantics on the
    /// non-negative domain the IR's INDEX algebra operates in.
    Bin(BinOp, Box<Term>, Box<Term>),
    /// `min(a, b)`.
    Min(Box<Term>, Box<Term>),
    /// `max(a, b)`.
    Max(Box<Term>, Box<Term>),
    /// Value of `tensor.read` — uninterpreted function per region.
    Read { func: String, args: Vec<Term> },
    /// `get_block_idx()` — this block's index in `[0, width)`.
    BlockIdx,
}

/// Binary integer operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Boolean terms: path conditions and comparisons.
#[derive(Debug, Clone, PartialEq)]
pub enum Bool {
    /// Folded constant.
    B(bool),
    /// Integer comparison.
    Cmp(CmpOp, Box<Term>, Box<Term>),
    /// Negation.
    Not(Box<Bool>),
    /// Conjunction.
    And(Vec<Bool>),
    /// Disjunction.
    Or(Vec<Bool>),
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Term {
    /// Folds two concrete ints, else builds the node.
    pub fn bin(op: BinOp, a: Term, b: Term) -> Term {
        match (op, &a, &b) {
            (BinOp::Add, Term::Int(x), Term::Int(y)) => Term::Int(x + y),
            (BinOp::Sub, Term::Int(x), Term::Int(y)) => Term::Int(x - y),
            (BinOp::Mul, Term::Int(x), Term::Int(y)) => Term::Int(x * y),
            (BinOp::Div, Term::Int(x), Term::Int(y)) if *y != 0 => Term::Int(x / y),
            (BinOp::Mod, Term::Int(x), Term::Int(y)) if *y != 0 => Term::Int(x % y),
            _ => Term::Bin(op, Box::new(a), Box::new(b)),
        }
    }

    /// Folds `min` on concrete ints.
    pub fn min(a: Term, b: Term) -> Term {
        match (&a, &b) {
            (Term::Int(x), Term::Int(y)) => Term::Int(*x.min(y)),
            _ => Term::Min(Box::new(a), Box::new(b)),
        }
    }

    /// Folds `max` on concrete ints.
    pub fn max(a: Term, b: Term) -> Term {
        match (&a, &b) {
            (Term::Int(x), Term::Int(y)) => Term::Int(*x.max(y)),
            _ => Term::Max(Box::new(a), Box::new(b)),
        }
    }

    /// Every uninterpreted read function mentioned anywhere in the term.
    pub fn read_funcs(&self, out: &mut Vec<String>) {
        match self {
            Term::Read { func, args } => {
                if !out.contains(func) {
                    out.push(func.clone());
                }
                for a in args {
                    a.read_funcs(out);
                }
            }
            Term::Bin(_, a, b) | Term::Min(a, b) | Term::Max(a, b) => {
                a.read_funcs(out);
                b.read_funcs(out);
            }
            _ => {}
        }
    }
}

impl Bool {
    /// Conjunction that folds constants away.
    pub fn conj(parts: impl IntoIterator<Item = Bool>) -> Bool {
        let mut kept = Vec::new();
        for p in parts {
            match p {
                Bool::B(true) => {}
                Bool::B(false) => return Bool::B(false),
                other => kept.push(other),
            }
        }
        match kept.len() {
            0 => Bool::B(true),
            1 => kept.pop().unwrap_or(Bool::B(true)),
            _ => Bool::And(kept),
        }
    }

    /// Every uninterpreted read function mentioned in the condition.
    pub fn read_funcs(&self, out: &mut Vec<String>) {
        match self {
            Bool::Cmp(_, a, b) => {
                a.read_funcs(out);
                b.read_funcs(out);
            }
            Bool::Not(inner) => inner.read_funcs(out),
            Bool::And(parts) | Bool::Or(parts) => {
                for p in parts {
                    p.read_funcs(out);
                }
            }
            Bool::B(_) => {}
        }
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Eq => "=",
            CmpOp::Ne => "<>",
        };
        f.write_str(s)
    }
}
