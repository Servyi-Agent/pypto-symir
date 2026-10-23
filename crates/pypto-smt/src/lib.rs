//! pypto-smt — bounded symbolic SMT analysis over the printed PyPTO IR.
//!
//! A separate tool from symir: it reuses the [`pypto_ir`] frontend
//! directly (grammar, parser, AST) and encodes questions about one
//! construct into bounded symbolic formulas solved with [z3].
//!
//! First target: the SPMD case. An `pl.spmd(core_num)` body runs
//! `core_num` concurrent block instances distinguished by
//! `get_block_idx()`; the body's worker-strided `pl.range` partitions
//! work across them. The core soundness lemma — two distinct workers
//! never draw the same work item — is exactly a bounded integer formula.
//!
//! [z3]: https://crates.io/crates/z3

// The servyi lint driver knows `inline_literal`; stable clippy does not —
// silence the unknown-lint noise at the crate root (tests do the same).
#![allow(unknown_lints)]

pub mod consts;
pub mod mlw;
pub mod term;
pub mod walk;

use pypto_ir::ast::{Expr, Program, Stmt};
use z3::ast::{self, Ast};
use z3::{Config, Context, SatResult, Solver};

/// One `with pl.spmd(core_num, ...)` site found in a program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpmdSite {
    /// Function containing the scope.
    pub function: String,
    /// `name_hint` kwarg, if present.
    pub hint: Option<String>,
    /// The `core_num` literal: how many concurrent block instances run
    /// this body.
    pub width: i64,
    /// Statements in the body.
    pub body_len: usize,
}

/// A call's payload: positional args and kwargs.
pub type CallArgs<'a> = (&'a Vec<Expr>, &'a Vec<(String, Expr)>);

/// Dotted-name check: is `expr` the call `pl.<leaf>`?
fn is_pl_call<'a>(expr: &'a Expr, leaf: &str) -> Option<CallArgs<'a>> {
    let Expr::Call { func, args, kwargs } = expr else {
        return None;
    };
    let Expr::Attr(base, field) = func.as_ref() else {
        return None;
    };
    if matches!(base.as_ref(), Expr::Name(module) if module == consts::MODULE_PL) && field == leaf {
        Some((args, kwargs))
    } else {
        None
    }
}

/// Walks every statement of every function, collecting spmd scope sites
/// (including nested ones).
pub fn find_spmd(program: &Program) -> Vec<SpmdSite> {
    let mut sites = Vec::new();
    for function in &program.functions {
        walk_stmts(&function.name, &function.body, &mut sites);
    }
    sites
}

fn walk_stmts(function: &str, stmts: &[Stmt], sites: &mut Vec<SpmdSite>) {
    for stmt in stmts {
        match stmt {
            Stmt::With { scope, body, .. } => {
                if let Some((args, kwargs)) = is_pl_call(scope, consts::SCOPE_SPMD) {
                    let width = args.first().and_then(|a| match a {
                        Expr::Int(n) => Some(*n),
                        _ => None,
                    });
                    let hint = kwargs
                        .iter()
                        .find(|(k, _)| k == consts::KWARG_NAME_HINT)
                        .and_then(|(_, v)| match v {
                            Expr::Str(s) => Some(s.clone()),
                            _ => None,
                        });
                    if let Some(width) = width {
                        sites.push(SpmdSite {
                            function: function.to_owned(),
                            hint,
                            width,
                            body_len: body.len(),
                        });
                    }
                }
                walk_stmts(function, body, sites);
            }
            Stmt::If { arms } => {
                for (_, body) in arms {
                    walk_stmts(function, body, sites);
                }
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                walk_stmts(function, body, sites);
            }
            _ => {}
        }
    }
}

/// The strided-partition lemma, solved by z3: with workers `w ∈ [0,
/// width)` drawing work items `work = w + stride * k` (`k >= 0`), can two
/// *distinct* workers ever draw the *same* work item?
///
/// `false` (the collision is satisfiable) means the partition is unsound
/// at these parameters — e.g. `stride < width` lets workers overlap.
pub fn stride_partition_is_disjoint(width: i64, stride: i64) -> bool {
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let zero = ast::Int::from_i64(&ctx, consts::INT_ZERO);
    let w_width = ast::Int::from_i64(&ctx, width);
    let s_stride = ast::Int::from_i64(&ctx, stride);
    let w1 = ast::Int::new_const(&ctx, consts::SYM_WORKER_A);
    let w2 = ast::Int::new_const(&ctx, consts::SYM_WORKER_B);
    let k1 = ast::Int::new_const(&ctx, consts::SYM_STEP_INDEX_A);
    let k2 = ast::Int::new_const(&ctx, consts::SYM_STEP_INDEX_B);

    let work1 = ast::Int::add(&ctx, &[&w1, &ast::Int::mul(&ctx, &[&s_stride, &k1])]);
    let work2 = ast::Int::add(&ctx, &[&w2, &ast::Int::mul(&ctx, &[&s_stride, &k2])]);

    let solver = Solver::new(&ctx);
    for w in [&w1, &w2] {
        solver.assert(&w.ge(&zero));
        solver.assert(&w.lt(&w_width));
    }
    for k in [&k1, &k2] {
        solver.assert(&k.ge(&zero));
    }
    solver.assert(&w1._eq(&w2).not());
    solver.assert(&work1._eq(&work2));

    matches!(solver.check(), SatResult::Unsat)
}
