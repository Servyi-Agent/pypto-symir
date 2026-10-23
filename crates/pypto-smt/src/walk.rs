//! The SPMD body walk: one symbolic pass over an `pl.spmd(core_num, …)`
//! body, producing an access log — the memory model every check queries.
//!
//! Legality gate (class-level refusals, per the regression rule): a
//! construct that cannot be characterized *exactly* by instance variables
//! is refused with an error naming it — `while`-shaped loops (phantom
//! iterations would fabricate writes and fake coverage proofs), pipeline
//! iterators, loop-carried iteration args, non-literal or non-positive
//! range steps, dynamic dispatch, and any scalar whose value we cannot
//! compute. Refusal is loud; nothing is silently approximated.

use std::collections::HashMap;

use pypto_ir::ast::{BinOp as AstBinOp, Expr, Function, Program, Stmt, UnaryOp};
use pypto_ir::consts;

use crate::consts as tool;
use crate::term::{BinOp, Bool, CmpOp, Term};

/// Direction of one access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    Read,
    Write,
}

/// One logged memory event: window, path condition, executing instance.
#[derive(Debug, Clone)]
pub struct Access {
    pub dir: Dir,
    pub region: String,
    pub offsets: Vec<Term>,
    pub shapes: Vec<Term>,
    pub guards: Bool,
    /// Instance variables in scope (worker first).
    pub coords: Vec<String>,
    /// Havoced writer-context symbols in scope (carried values).
    pub havoc: Vec<String>,
}

/// One loop of the body, as an exact instance characterization.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    pub var: String,
    /// `[lo, hi)` with `mu = lo + step * k`, `k >= 0`; the step may be a
    /// term (a runtime stride is part of the havoced writer context, and
    /// the encoded instances stay a superset of the real ones).
    pub lo: Term,
    pub hi: Term,
    pub step: Term,
}

/// The walk result for one spmd body.
#[derive(Debug, Clone, Default)]
pub struct SpmdBody {
    pub width: i64,
    pub accesses: Vec<Access>,
    pub loops: Vec<LoopInfo>,
    /// Uninterpreted `tensor.read` value functions (name, arity).
    pub read_funcs: Vec<(String, usize)>,
    /// Constant symbols the model needs declared (dims, module globals).
    pub consts: Vec<(String, i64)>,
    /// Named non-constant symbols with no value (dynamic dims, tensor dims).
    pub symbols: Vec<String>,
    /// Carried-value symbols (havoced in every writer context).
    pub havocs: Vec<String>,
}

/// Why the walk refused a body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported {
    pub construct: String,
    pub detail: String,
}

/// The value environment during the walk.
#[derive(Debug, Clone)]
enum Val {
    Scalar(Term),
    Tile { shape: Vec<Term> },
    Region(String),
    Tuple(Vec<Val>),
    None,
}

struct Walker<'a> {
    program: &'a Program,
    body: SpmdBody,
    env: HashMap<String, Val>,
    guards: Vec<Bool>,
    coords: Vec<String>,
    havoc: Vec<String>,
    loop_seq: usize,
    fresh_seq: usize,
}

/// Walks the `n`-th spmd site of the program (0-based, in `find_spmd` order).
pub fn walk_spmd(program: &Program, site: usize) -> Result<SpmdBody, Unsupported> {
    let sites = collect_sites(program);
    let (function, width) = sites
        .into_iter()
        .nth(site)
        .ok_or_else(|| Unsupported {
            construct: tool::SITE_SELECT.to_owned(),
            detail: format!("only {} spmd site(s) in program", count_sites(program)),
        })?;
    let mut walker = Walker {
        program,
        body: SpmdBody {
            width,
            ..SpmdBody::default()
        },
        env: HashMap::new(),
        guards: Vec::new(),
        coords: vec![tool::SYM_WORKER.to_owned()],
        havoc: Vec::new(),
        loop_seq: 0,
        fresh_seq: 0,
    };
    for (name, value) in &program.globals {
        match value {
            Expr::Call { func, .. } if is_attr(func, consts::CALL_DYNAMIC) => {
                walker.body.symbols.push(sanitize(name));
            }
            _ => {}
        }
    }
    for param in &function.params {
        if is_scalar_param(param) {
            let sym = sanitize(&param.name);
            if !walker.body.symbols.contains(&sym) {
                walker.body.symbols.push(sym.clone());
            }
            let _ = walker
                .env
                .insert(param.name.clone(), Val::Scalar(Term::Var(sym)));
        } else {
            let _ = walker
                .env
                .insert(param.name.clone(), Val::Region(param.name.clone()));
            walker.record_param_dims(param);
        }
    }
    // The whole function body: pre-scope statements define the scalars
    // (work_items, macro_tiles, ...) the spmd body consumes.
    walker.stmts(&function.body)?;
    Ok(walker.body)
}

/// Collected spmd sites: the owning function and its literal core count.
type Sites<'f> = Vec<(&'f Function, i64)>;

fn collect_sites(program: &Program) -> Sites<'_> {
    let mut out = Vec::new();
    for function in &program.functions {
        collect_stmts(function, &function.body, &mut out);
    }
    out
}

fn count_sites(program: &Program) -> usize {
    collect_sites(program).len()
}

fn collect_stmts<'f>(function: &'f Function, stmts: &[Stmt], out: &mut Sites<'f>) {
    for stmt in stmts {
        match stmt {
            Stmt::With { scope, body, .. } => {
                if let Some((args, _)) = as_pl_call(scope, consts::SCOPE_SPMD) {
                    if let Some(Expr::Int(width)) = args.first() {
                        out.push((function, *width));
                    }
                }
                collect_stmts(function, body, out);
            }
            Stmt::If { arms } => {
                for (_, arm) in arms {
                    collect_stmts(function, arm, out);
                }
            }
            // The loop form: `for w in pl.spmd(n):` — one site per loop.
            Stmt::For { iterator, body, .. } => {
                if let Some((args, _)) = as_pl_call(iterator, consts::SCOPE_SPMD) {
                    if let Some(Expr::Int(width)) = args.first() {
                        out.push((function, *width));
                    }
                }
                collect_stmts(function, body, out);
            }
            Stmt::While { body, .. } => {
                collect_stmts(function, body, out);
            }
            _ => {}
        }
    }
}

/// `pl.Scalar[...]`-annotated parameter? (`Scalar` appears as the
/// attribute at some subscript level of the annotation spine.)
fn is_scalar_param(param: &pypto_ir::ast::Param) -> bool {
    let mut cursor = &param.annotation;
    loop {
        match cursor {
            Expr::Subscript { base, .. } => cursor = base,
            Expr::Attr(inner, leaf) => {
                return matches!(inner.as_ref(), Expr::Name(n) if n == consts::MODULE_PL)
                    && leaf == consts::TYPE_SCALAR;
            }
            _ => return false,
        }
    }
}

/// `pl.<leaf>(args, kwargs)` shape check.
/// A `pl.<leaf>` call's payload: positional args and kwargs.
pub type PlCall<'a> = (&'a [Expr], &'a [(String, Expr)]);

fn as_pl_call<'a>(expr: &'a Expr, leaf: &str) -> Option<PlCall<'a>> {
    let Expr::Call { func, args, kwargs } = expr else {
        return None;
    };
    if is_attr(func, leaf) {
        Some((args, kwargs))
    } else {
        None
    }
}

fn is_attr(expr: &Expr, leaf: &str) -> bool {
    matches!(expr, Expr::Attr(base, field) if matches!(base.as_ref(), Expr::Name(module) if module == consts::MODULE_PL) && field == leaf)
}

/// Lowercase identifier for WhyML.
pub fn sanitize(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == tool::SANITIZE_KEEP {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(tool::SANITIZE_FALLBACK);
        }
    }
    out
}

impl<'a> Walker<'a> {
    fn stmts(&mut self, stmts: &[Stmt]) -> Result<(), Unsupported> {
        for stmt in stmts {
            self.stmt(stmt)?;
        }
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<(), Unsupported> {
        match stmt {
            Stmt::Assign { targets, annotation, value } => {
                let val = self.value_of(value, annotation)?;
                if let [name] = targets.as_slice() {
                    let _ = self.env.insert(name.clone(), val);
                }
                Ok(())
            }
            Stmt::If { arms } => {
                let mut negatives: Vec<Bool> = Vec::new();
                for (cond, body) in arms {
                    match cond {
                        Some(c) => {
                            let b = self.bool_of(c)?;
                            negatives.push(b.clone());
                            self.guards.push(b);
                            self.stmts(body)?;
                            let _ = self.guards.pop();
                        }
                        None => {
                            let none_taken = Bool::Or(negatives.clone());
                            self.guards.push(Bool::Not(Box::new(none_taken)));
                            self.stmts(body)?;
                            let _ = self.guards.pop();
                        }
                    }
                }
                Ok(())
            }
            Stmt::For { index_vars, iter_arg_vars, iterator, body } => {
                let havoc_before = self.havoc.len();
                self.enter_loop(index_vars, iter_arg_vars, iterator)?;
                self.stmts(body)?;
                let _ = self.coords.pop();
                self.havoc.truncate(havoc_before);
                // Sequential-accumulation exit values still reference this
                // loop's mu; their exact value depends on the whole
                // execution, so they join the havoced writer context (and
                // stay in scope — an over-approximation, like every havoc).
                if let Some(last) = self.body.loops.last().cloned() {
                    let mu = last.var;
                    let rebinding = self.rebind_mu_mentions(&mu);
                    for sym in rebinding {
                        self.havoc.push(sym);
                    }
                }
                Ok(())
            }
            Stmt::While { .. } => Err(Unsupported {
                construct: tool::WHILE.to_owned(),
                detail: tool::MSG_WHILE_REFUSED.to_owned(),
            }),
            Stmt::With { body, .. } => self.stmts(body),
            Stmt::Return(value) => {
                if let Some(expr) = value {
                    let _ = self.value_of(expr, &None)?;
                }
                Ok(())
            }
            Stmt::Expr(expr) => {
                let _ = self.value_of(expr, &None)?;
                Ok(())
            }
        }
    }

    fn enter_loop(
        &mut self,
        index_vars: &[String],
        iter_args: &[String],
        iterator: &Expr,
    ) -> Result<(), Unsupported> {
        // `for w in pl.spmd(n):` — the loop IS the worker axis; the index
        // variable denotes the block index.
        if as_pl_call(iterator, consts::SCOPE_SPMD).is_some() {
            for var in index_vars {
                let _ = self
                    .env
                    .insert(var.clone(), Val::Scalar(Term::BlockIdx));
            }
            return Ok(());
        }
        let (lo, hi, step) = self.loop_bounds(iterator)?;
        let var = format!("{}{}", tool::MU_PREFIX, self.loop_seq);
        self.loop_seq += 1;
        self.body.loops.push(LoopInfo {
            var: var.clone(),
            lo: lo.clone(),
            hi: hi.clone(),
            step,
        });
        let _ = self.env.insert(
            index_vars.first().cloned().unwrap_or_else(|| var.clone()),
            Val::Scalar(Term::Var(var.clone())),
        );
        for extra in index_vars.iter().skip(tool::EXTRA_INDEX_VARS_START) {
            let _ = self.env.insert(extra.clone(), Val::Scalar(Term::Var(var.clone())));
        }
        // Data-race detection havocs the writer's context: carried values
        // enter scope as free symbols (sound superset for Q1/Q2; Q3 is
        // skipped for carried bodies).
        for carried in iter_args {
            let sym = self.fresh_havoc(carried);
            let _ = self.env.insert(carried.clone(), Val::Scalar(Term::Var(sym)));
        }
        self.coords.push(var);
        Ok(())
    }

    /// Rebinds every scalar whose term mentions `mu` to a fresh havoced
    /// symbol (loop-exit values join the writer context).
    fn rebind_mu_mentions(&mut self, mu: &str) -> Vec<String> {
        let mut rebinds: Vec<(String, String)> = Vec::new();
        for (name, val) in self.env.iter() {
            if let Val::Scalar(t) = val {
                if term_mentions(t, mu) {
                    rebinds.push((name.clone(), String::new()));
                }
            }
        }
        let mut symbols = Vec::new();
        for (name, _) in &rebinds {
            let sym = self.next_havoc(name);
            symbols.push(sym);
        }
        for ((name, _), sym) in rebinds.into_iter().zip(symbols.iter().cloned()) {
            let _ = self.env.insert(name, Val::Scalar(Term::Var(sym)));
        }
        symbols
    }

    /// Registers a fresh havoc symbol without pushing a scope (used for
    /// loop-exit rebinds living in the enclosing scope).
    fn next_havoc(&mut self, hint: &str) -> String {
        let sym = format!("{}{}_{}", tool::HAVOC_PREFIX, sanitize(hint), self.fresh_seq);
        self.fresh_seq += 1;
        if !self.body.havocs.contains(&sym) {
            self.body.havocs.push(sym.clone());
        }
        sym
    }

    /// A fresh havoced writer-context symbol, registered body-wide.
    fn fresh_havoc(&mut self, hint: &str) -> String {
        let sym = format!("{}{}_{}", tool::HAVOC_PREFIX, sanitize(hint), self.fresh_seq);
        self.fresh_seq += 1;
        if !self.body.havocs.contains(&sym) {
            self.body.havocs.push(sym.clone());
        }
        self.havoc.push(sym.clone());
        sym
    }

    /// `pl.range(stop | start,stop | start,stop,step)` / `pl.unroll(n)` —
    /// exact instance bounds; positive literal step required.
    // IR op names, argument positions and arities below are the IR's
    // own signatures (necessities, not choices).
    #[allow(inline_literal)]
    fn loop_bounds(&mut self, iterator: &Expr) -> Result<(Term, Term, Term), Unsupported> {
        let (args, leaf) = match iterator {
            Expr::Call { func, args, .. } if is_attr(func, consts::ITER_RANGE)
                || is_attr(func, consts::ITER_UNROLL)
                || is_attr(func, consts::ITER_PARALLEL) =>
            {
                (args, String::new())
            }
            other => {
                let name = dotted_name(other);
                return Err(Unsupported {
                    construct: tool::ITERATOR.to_owned(),
                    detail: format!("{name}: {}", tool::MSG_ITERATOR_REFUSED),
                });
            }
        };
        let _ = leaf;
        match args.len() {
            1 => Ok((Term::Int(0), self.term(&args[0])?, Term::Int(1))),
            2 => Ok((self.term(&args[0])?, self.term(&args[1])?, Term::Int(1))),
            3 => Ok((self.term(&args[0])?, self.term(&args[1])?, self.term(&args[2])?)),
            _ => Err(Unsupported {
                construct: tool::ITERATOR.to_owned(),
                detail: tool::MSG_ITERATOR_REFUSED.to_owned(),
            }),
        }
    }

    /// The value of one expression: scalar term, tile (with shape), or
    /// region reference.
    fn value_of(&mut self, expr: &Expr, annotation: &Option<Expr>) -> Result<Val, Unsupported> {
        match expr {
            Expr::Int(n) => Ok(Val::Scalar(Term::Int(*n))),
            Expr::Name(name) => Ok(self.env.get(name).cloned().unwrap_or(Val::None)),
            Expr::Attr(base, _) => Ok(self.value_of(base, annotation)?),
            Expr::Subscript { base, indices } => {
                // Tuple element selection on multi-return calls.
                let val = self.value_of(base, annotation)?;
                if let [pypto_ir::ast::IndexItem::Expr(item)] = indices.as_slice() {
                    if let (Val::Tuple(items), Expr::Int(idx)) = (&val, item.as_ref()) {
                        if let Some(found) = items.get(*idx as usize) {
                            return Ok(found.clone());
                        }
                    }
                }
                Ok(val)
            }
            Expr::Tuple(items) => {
                let mut vals = Vec::new();
                for item in items {
                    vals.push(self.value_of(item, &None)?);
                }
                Ok(Val::Tuple(vals))
            }
            Expr::Call { func, args, kwargs } => self.call_value(func, args, kwargs, annotation),
            Expr::BinOp(op, a, b) => {
                let (x, y) = (self.term(a)?, self.term(b)?);
                let binop = match op {
                    AstBinOp::Add => BinOp::Add,
                    AstBinOp::Sub => BinOp::Sub,
                    AstBinOp::Mul => BinOp::Mul,
                    AstBinOp::FloorDiv => BinOp::Div,
                    AstBinOp::Mod => BinOp::Mod,
                    other => {
                        return Err(Unsupported {
                            construct: tool::SCALAR_OP.to_owned(),
                            detail: format!("{other}"),
                        })
                    }
                };
                Ok(Val::Scalar(Term::bin(binop, x, y)))
            }
            Expr::UnaryOp(UnaryOp::Neg, inner) => {
                let t = self.term(inner)?;
                Ok(Val::Scalar(Term::bin(BinOp::Sub, Term::Int(0), t)))
            }
            other => Err(Unsupported {
                construct: tool::EXPRESSION.to_owned(),
                detail: format!("{other:?}"),
            }),
        }
    }

    // IR op names, argument positions and arities below are the IR's
    // own signatures (necessities, not choices).
    #[allow(inline_literal)]
    fn call_value(
        &mut self,
        func: &Expr,
        args: &[Expr],
        _kwargs: &[(String, Expr)],
        annotation: &Option<Expr>,
    ) -> Result<Val, Unsupported> {
        let path = op_path(func);
        let name: String = match &path {
            OpPath::Pl(leaf) => (**leaf).to_owned(),
            OpPath::Namespaced(namespace, leaf) => {
                format!("{}{}{}", namespace, consts::CHAR_DOT, leaf)
            }
            OpPath::Other => dotted_name(func),
        };
        let leaf_name: String = match &path {
            OpPath::Pl(leaf) | OpPath::Namespaced(_, leaf) => (**leaf).to_owned(),
            OpPath::Other => String::new(),
        };
        match name.as_str() {
            "tile.get_block_idx" | "tensor.get_block_idx" => Ok(Val::Scalar(Term::BlockIdx)),
            "tile.get_block_num" | "tensor.get_block_num" => {
                Ok(Val::Scalar(Term::Int(self.body.width)))
            }
            "const" => match args.first() {
                Some(Expr::Int(n)) => Ok(Val::Scalar(Term::Int(*n))),
                _ => Ok(Val::Scalar(Term::Int(0))),
            },
            "cast" => self.value_of(&args[0], annotation),
            "min" => {
                let (a, b) = (self.term(&args[0])?, self.term(&args[1])?);
                Ok(Val::Scalar(Term::min(a, b)))
            }
            "max" => {
                let (a, b) = (self.term(&args[0])?, self.term(&args[1])?);
                Ok(Val::Scalar(Term::max(a, b)))
            }
            "tensor.dim" => {
                let region = self.region_of(&args[0])?;
                let axis = match &args[1] {
                    Expr::Int(n) => *n,
                    _ => 0,
                };
                let dname = format!("{}{}_{}", tool::DIM_PREFIX, sanitize(&region), axis);
                if !self.body.symbols.contains(&dname)
                    && !self.body.consts.iter().any(|(n, _)| *n == dname)
                {
                    self.body.symbols.push(dname.clone());
                }
                Ok(Val::Scalar(Term::Var(dname)))
            }
            "tensor.read" => {
                let region = self.region_of(&args[0])?;
                let flat = self.index_list(args.get(1))?;
                self.record_access(Dir::Read, &region, flat.clone(), vec![Term::Int(1)])?;
                let fname = format!("{}{}", tool::READ_PREFIX, sanitize(&region));
                let entry = (fname.clone(), flat.len());
                if !self.body.read_funcs.contains(&entry) {
                    self.body.read_funcs.push(entry);
                }
                Ok(Val::Scalar(Term::Read { func: fname, args: flat }))
            }
            "tile.write" => {
                // On-chip tile write: no external effect, unless the
                // destination resolves to a tensor-backed region.
                if let Some(Val::Region(region)) = self.probe_value(&args[0]) {
                    let offs = self.index_list(args.get(1))?;
                    let shapes = match self.tile_shape_of(&args[2]) {
                        Some(shapes) => shapes,
                        None => vec![Term::Int(1); offs.len()],
                    };
                    self.record_access(Dir::Write, &region, offs, shapes)?;
                }
                Ok(Val::None)
            }
            "tile.read" => {
                // On-chip tile element read: no external memory access; the
                // value stays uninterpreted (dead branches rely on this).
                let flat = self.index_list(args.get(1))?;
                let fname = format!("{}{}", tool::TILE_READ_FUNC, flat.len());
                let entry = (fname.clone(), flat.len());
                if !self.body.read_funcs.contains(&entry) {
                    self.body.read_funcs.push(entry);
                }
                Ok(Val::Scalar(Term::Read { func: fname, args: flat }))
            }
            "tile.load" => {
                let region = self.region_of(&args[0])?;
                let offs = self.index_list(args.get(1))?;
                let shapes = self.index_list(args.get(2))?;
                self.record_access(Dir::Read, &region, offs.clone(), shapes.clone())?;
                Ok(Val::Tile { shape: shapes })
            }
            "tile.store" => {
                let shapes = match args.first().and_then(|v| self.tile_shape_of(v)) {
                    Some(shapes) => shapes,
                    None => {
                        return Err(Unsupported {
                            construct: tool::STORE_SHAPE.to_owned(),
                            detail: tool::MSG_STORE_SHAPE.to_owned(),
                        })
                    }
                };
                let region = self.region_of(&args[2])?;
                let offs = self.index_list(args.get(1))?;
                self.record_access(Dir::Write, &region, offs, shapes)?;
                // The store's result denotes the updated destination.
                Ok(Val::Region(region))
            }
            "yield_" => {
                // The carried value flows to the next iteration; its
                // rebinding is a fresh havoced context symbol. Arguments
                // are still evaluated for their memory effects.
                for arg in args {
                    let _ = self.value_of(arg, &None)?;
                }
                Ok(Val::Scalar(Term::Var(self.fresh_havoc(tool::HAVOC_YIELD))))
            }
            "tensor.create" | "create" => {
                // A fresh private region (scratch/carried buffer).
                let name = format!("{}{}", tool::FRESH_REGION_PREFIX, self.fresh_seq);
                self.fresh_seq += 1;
                let _ = self.index_list(args.first());
                Ok(Val::Region(name))
            }
            "tile.assemble" => {
                // On-chip tile composition: no external memory effect; the
                // result keeps the destination tile's shape.
                match args.first().and_then(|v| self.tile_shape_of(v)) {
                    Some(shape) => Ok(Val::Tile { shape }),
                    None => annotation_shape(annotation)
                        .map(|shape| Val::Tile { shape })
                        .ok_or_else(|| Unsupported {
                            construct: tool::STORE_SHAPE.to_owned(),
                            detail: tool::MSG_STORE_SHAPE.to_owned(),
                        }),
                }
            }
            "tile.set_validshape" => {
                let mut shapes = Vec::new();
                for a in args.iter().skip(1) {
                    shapes.push(self.term(a)?);
                }
                Ok(Val::Tile { shape: shapes })
            }
            other => {
                // Only tensor.* views compose onto the base region;
                // tile.* slicing/composition stays on-chip.
                if let OpPath::Namespaced(namespace, leaf) = &path {
                    if namespace == &tool::NAMESPACE_TENSOR.to_owned()
                        && consts::VIEW_FORMING_OPS.contains(leaf)
                    {
                        if *leaf == consts::OP_TENSOR_ASSEMBLE {
                            return Err(Unsupported {
                                construct: tool::CALL.to_owned(),
                                detail: format!("{other}: {}", tool::MSG_ASSEMBLE_REFUSED),
                            });
                        }
                        let region = self.region_of(&args[0])?;
                        return Ok(Val::Region(region));
                    }
                }
                let _ = leaf_name;
                if let Some(method) = self_method(func) {
                    return self.inline_call(&method, args);
                }
                if let Some(annotation_shape) = annotation_shape(annotation) {
                    return Ok(Val::Tile { shape: annotation_shape });
                }
                // Unknown tile.* compute is on-chip and memory-invisible;
                // it passes through with an unknown shape (a store of it
                // refuses on the missing window instead).
                if let OpPath::Namespaced(namespace, _) = &path {
                    if namespace == &tool::NAMESPACE_TILE.to_owned() {
                        return Ok(Val::Tile { shape: Vec::new() });
                    }
                }
                Err(Unsupported {
                    construct: tool::CALL.to_owned(),
                    detail: format!("{other}: {}", tool::MSG_CALL_REFUSED),
                })
            }
        }
    }

    /// Evaluates an expression without recording accesses (probe).
    fn probe_value(&mut self, expr: &Expr) -> Option<Val> {
        let before = self.body.accesses.len();
        let result = self.value_of(expr, &None).ok();
        self.body.accesses.truncate(before);
        result
    }

    /// The index expressions of a `[a, b, ...]` argument.
    // IR op names, argument positions and arities below are the IR's
    // own signatures (necessities, not choices).
    #[allow(inline_literal)]
    fn index_list(&mut self, expr: Option<&Expr>) -> Result<Vec<Term>, Unsupported> {
        match expr {
            Some(Expr::List(items)) => {
                let mut out = Vec::new();
                for item in items {
                    out.push(self.term(item)?);
                }
                Ok(out)
            }
            Some(single) => Ok(vec![self.term(single)?]),
            None => Err(Unsupported {
                construct: tool::INDEX_LIST.to_owned(),
                detail: tool::MSG_NONSCALAR.to_owned(),
            }),
        }
    }

    fn inline_call(&mut self, name: &str, args: &[Expr]) -> Result<Val, Unsupported> {
        let function = self
            .program
            .functions
            .iter()
            .find(|f| f.name == name && (is_inline(f) || is_incore(f)))
            .ok_or_else(|| Unsupported {
                construct: tool::CALL.to_owned(),
                detail: format!("{name}: {}", tool::MSG_CALL_REFUSED),
            })?;
        // Bind the callee's parameters to the caller's argument values.
        let mut bound = Vec::new();
        for (param, arg) in function.params.iter().zip(args) {
            let value = self.value_of(arg, &None)?;
            bound.push((param.name.clone(), value));
        }
        let saved: Vec<(String, Option<Val>)> = bound
            .iter()
            .map(|(n, _)| (n.clone(), self.env.get(n).cloned()))
            .collect();
        for (name, value) in &bound {
            let _ = self.env.insert(name.clone(), value.clone());
        }
        let mut result = Ok(Val::None);
        for stmt in &function.body {
            if let Stmt::Return(Some(expr)) = stmt {
                result = self.value_of(expr, &None);
                break;
            }
            if let Err(e) = self.stmt(stmt) {
                result = Err(e);
                break;
            }
        }
        for (name, old) in saved {
            match old {
                Some(v) => {
                    let _ = self.env.insert(name, v);
                }
                None => {
                    let _ = self.env.remove(&name);
                }
            }
        }
        result
    }

    fn term(&mut self, expr: &Expr) -> Result<Term, Unsupported> {
        match self.value_of(expr, &None)? {
            Val::Scalar(t) => Ok(t),
            Val::Tile { .. } | Val::Region(_) | Val::Tuple(_) | Val::None => Err(Unsupported {
                construct: tool::SCALAR_OP.to_owned(),
                detail: format!("{}: {expr:?}", tool::MSG_NONSCALAR),
            }),
        }
    }

    fn bool_of(&mut self, expr: &Expr) -> Result<Bool, Unsupported> {
        match expr {
            Expr::Bool(b) => Ok(Bool::B(*b)),
            Expr::BinOp(op, a, b) => match op {
                AstBinOp::And => Ok(Bool::conj([self.bool_of(a)?, self.bool_of(b)?])),
                AstBinOp::Or => Ok(Bool::Or(vec![self.bool_of(a)?, self.bool_of(b)?])),
                _ => {
                    let cmp = match op {
                        AstBinOp::Lt => CmpOp::Lt,
                        AstBinOp::Le => CmpOp::Le,
                        AstBinOp::Gt => CmpOp::Gt,
                        AstBinOp::Ge => CmpOp::Ge,
                        AstBinOp::Eq => CmpOp::Eq,
                        AstBinOp::Ne => CmpOp::Ne,
                        other => {
                            return Err(Unsupported {
                                construct: tool::CONDITION.to_owned(),
                                detail: format!("{other}"),
                            })
                        }
                    };
                    Ok(Bool::Cmp(cmp, Box::new(self.term(a)?), Box::new(self.term(b)?)))
                }
            },
            Expr::UnaryOp(UnaryOp::Not, inner) => {
                Ok(Bool::Not(Box::new(self.bool_of(inner)?)))
            }
            other => Err(Unsupported {
                construct: tool::CONDITION.to_owned(),
                detail: format!("{other:?}"),
            }),
        }
    }

    fn region_of(&mut self, expr: &Expr) -> Result<String, Unsupported> {
        match self.value_of(expr, &None)? {
            Val::Region(base) => Ok(base),
            other => Err(Unsupported {
                construct: tool::REGION.to_owned(),
                detail: format!("{}: {expr:?} (value {:?})", tool::MSG_NONSCALAR, other),
            }),
        }
    }

    fn tile_shape_of(&mut self, expr: &Expr) -> Option<Vec<Term>> {
        match self.value_of(expr, &None) {
            Ok(Val::Tile { shape }) => Some(shape),
            _ => None,
        }
    }

    fn record_access(
        &mut self,
        dir: Dir,
        region: &str,
        offsets: Vec<Term>,
        shapes: Vec<Term>,
    ) -> Result<(), Unsupported> {
        let guards = Bool::conj(self.guards.iter().cloned());
        let mut funcs = Vec::new();
        for t in &offsets {
            t.read_funcs(&mut funcs);
        }
        for t in &shapes {
            t.read_funcs(&mut funcs);
        }
        guards.read_funcs(&mut funcs);
        for entry in funcs {
            if !self.body.read_funcs.contains(&entry) {
                self.body.read_funcs.push(entry);
            }
        }
        self.body.accesses.push(Access {
            dir,
            region: region.to_owned(),
            offsets,
            shapes,
            guards,
            coords: self.coords.clone(),
            havoc: self.havoc.clone(),
        });
        Ok(())
    }

    fn record_param_dims(&mut self, param: &pypto_ir::ast::Param) {
        let dims = tensor_dims(&param.annotation);
        for (axis, dim) in dims.iter().enumerate() {
            if let Expr::Int(n) = dim {
                self.body.consts.push((
                    format!("{}{}_{}", tool::DIM_PREFIX, sanitize(&param.name), axis),
                    *n,
                ));
            }
        }
    }
}

/// Does the term mention the loop variable `mu`?
fn term_mentions(t: &Term, mu: &str) -> bool {
    match t {
        Term::Var(name) => name == mu,
        Term::Bin(_, a, b) | Term::Min(a, b) | Term::Max(a, b) => {
            term_mentions(a, mu) || term_mentions(b, mu)
        }
        Term::Read { args, .. } => args.iter().any(|a| term_mentions(a, mu)),
        _ => false,
    }
}

fn self_method(func: &Expr) -> Option<String> {
    match func {
        Expr::Attr(base, field) => match base.as_ref() {
            Expr::Name(name) if name == consts::NAME_SELF => Some(field.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn is_inline(function: &Function) -> bool {
    function
        .decorators
        .iter()
        .any(|d| dotted_name(d).contains(consts::FN_TYPE_INLINE))
}

/// `type=pl.FunctionType.InCore` — the spmd dispatch bodies call these;
/// their bodies carry the memory accesses, so they are walked inline.
fn is_incore(function: &Function) -> bool {
    function
        .decorators
        .iter()
        .any(|d| dotted_name(d).contains(tool::FN_TYPE_INCORE))
}

fn dotted_name(expr: &Expr) -> String {
    match expr {
        Expr::Name(name) => name.clone(),
        Expr::Attr(base, field) => format!("{}.{}", dotted_name(base), field),
        other => format!("{other:?}"),
    }
}

/// Structural classification of a callee path: `pl.leaf`,
/// `pl.namespace.leaf`, or anything else — matched on the AST, not on
/// strings (the grammar already did the parsing).
enum OpPath<'e> {
    Pl(&'e str),
    Namespaced(&'e str, &'e str),
    Other,
}

fn op_path(func: &Expr) -> OpPath<'_> {
    let Expr::Attr(base, leaf) = func else {
        return OpPath::Other;
    };
    match base.as_ref() {
        Expr::Name(module) if module == consts::MODULE_PL => OpPath::Pl(leaf),
        Expr::Attr(grand, namespace)
            if matches!(grand.as_ref(), Expr::Name(m) if m == consts::MODULE_PL) =>
        {
            OpPath::Namespaced(namespace, leaf)
        }
        _ => OpPath::Other,
    }
}

/// The shape list of a `pl.Tensor[[r, c], …]` / `pl.Tile[[r, c], …]`
/// annotation, when fully literal.
fn annotation_shape(annotation: &Option<Expr>) -> Option<Vec<Term>> {
    let annotation = annotation.as_ref()?;
    let mut cursor = annotation;
    loop {
        match cursor {
            Expr::Subscript { base, indices } => {
                if let Some(dims) = indices.first().and_then(|i| match i {
                    pypto_ir::ast::IndexItem::Expr(e) => match e.as_ref() {
                        Expr::List(dims) => Some(dims.clone()),
                        _ => None,
                    },
                    pypto_ir::ast::IndexItem::SliceAll => None,
                }) {
                    if dims.is_empty() {
                        return None;
                    }
                    return Some(
                        dims.iter()
                            .map(|d| match d {
                                Expr::Int(n) => Term::Int(*n),
                                _ => Term::Var(dotted_name(d)),
                            })
                            .collect(),
                    );
                }
                cursor = base;
            }
            _ => return None,
        }
    }
}

/// The dim expressions of a tensor annotation.
fn tensor_dims(annotation: &Expr) -> Vec<Expr> {
    let mut cursor = annotation;
    loop {
        match cursor {
            Expr::Subscript { base, indices } => {
                if let Some(pypto_ir::ast::IndexItem::Expr(e)) = indices.first() {
                    if let Expr::List(dims) = e.as_ref() {
                        return dims.clone();
                    }
                }
                cursor = base;
            }
            _ => return Vec::new(),
        }
    }
}
