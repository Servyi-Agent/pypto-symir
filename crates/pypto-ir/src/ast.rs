//! AST for the printed PyPTO IR subset accepted by the grammar in
//! [`crate::pypto_ir.pest`] (module docs: see `parse.rs`).
//!
//! The shapes mirror the C++ IR we model: statements are the structured set
//! (`IfStmt`, `ForStmt`, `WhileStmt`, scopes, submits), expressions carry the
//! scalar algebra plus op calls, and annotations are kept as expressions so
//! directions (`pl.Out[...]`, `pl.InOut[...]`) can be extracted structurally.

use std::fmt;

/// A whole parsed file: one `@pl.program` class with its functions, plus
/// module-level constant assignments (dynamic-dim declarations).
#[derive(Debug, Clone)]
pub struct Program {
    pub name: String,
    pub functions: Vec<Function>,
    /// Module-level `NAME = expr` bindings before the class.
    pub globals: Vec<(String, Expr)>,
}

/// One `@pl.function`-decorated method of the program class.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    /// `self` is stripped, matching the C++ side.
    pub params: Vec<Param>,
    pub decorators: Vec<Expr>,
    pub body: Vec<Stmt>,
}

/// A parameter with its annotation kept as an expression.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub annotation: Expr,
}

/// Statement set of the printed IR.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// `target[: type] = expr` (single or tuple target).
    Assign {
        targets: Vec<String>,
        annotation: Option<Expr>,
        value: Expr,
    },
    /// `if cond:` / `elif` / `else` — arms in source order.
    If {
        arms: Vec<(Option<Expr>, Vec<Stmt>)>,
    },
    /// `for targets in iterator:` — `pl.range`, `pl.parallel`,
    /// `pl.unroll`, `pl.pipeline`, or the `pl.while_` form.
    For {
        index_vars: Vec<String>,
        iter_arg_vars: Vec<String>,
        iterator: Expr,
        body: Vec<Stmt>,
    },
    /// Natural `while cond:` — parsed to `WhileStmt` without iter_args;
    /// `ConvertToSSA` threads them on the compiler side.
    While { cond: Expr, body: Vec<Stmt> },
    /// `with expr [as ident]:` — scopes (`pl.scope`, `pl.at`,
    /// `pl.spmd`, `pl.cluster`, ...).
    With {
        scope: Expr,
        binder: Option<String>,
        body: Vec<Stmt>,
    },
    /// `return expr?`
    Return(Option<Expr>),
    /// Bare expression statement (calls, `pl.func_attr({...})`).
    Expr(Expr),
}

/// Expression set of the printed IR.
#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Name(String),
    /// `base.field`
    Attr(Box<Expr>, String),
    /// `func(args, kwargs...)`
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },
    /// `base[indices]`
    Subscript {
        base: Box<Expr>,
        indices: Vec<IndexItem>,
    },
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnaryOp, Box<Expr>),
}

/// One subscript position: a slice-all (`:`), or an expression.
#[derive(Debug, Clone)]
pub enum IndexItem {
    SliceAll,
    Expr(Box<Expr>),
}

/// Binary operators of the scalar algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    FloatDiv,
    FloorDiv,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            BinOp::Add => crate::consts::TOK_PLUS,
            BinOp::Sub => crate::consts::TOK_MINUS,
            BinOp::Mul => crate::consts::TOK_STAR,
            BinOp::FloatDiv => crate::consts::TOK_SLASH,
            BinOp::FloorDiv => crate::consts::TOK_DOUBLE_SLASH,
            BinOp::Mod => crate::consts::TOK_PERCENT,
            BinOp::Eq => crate::consts::TOK_EQ,
            BinOp::Ne => crate::consts::TOK_NE,
            BinOp::Lt => crate::consts::TOK_LT,
            BinOp::Le => crate::consts::TOK_LE,
            BinOp::Gt => crate::consts::TOK_GT,
            BinOp::Ge => crate::consts::TOK_GE,
            BinOp::And => crate::consts::TOK_AND,
            BinOp::Or => crate::consts::TOK_OR,
        };
        f.write_str(repr)
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            UnaryOp::Neg => crate::consts::TOK_MINUS,
            UnaryOp::Not => crate::consts::TOK_NOT,
        };
        f.write_str(repr)
    }
}
