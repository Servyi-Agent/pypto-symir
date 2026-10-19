//! The symbolic interpreter: walks one function of the parsed program and
//! extracts a [`TaskDag`].
//!
//! Semantics:
//!
//! * Op calls become **uninterpreted function applications**; scalar algebra
//!   folds only over concrete operands.
//! * Tensors, buffers, and allocations become **region identities**; op
//!   effects default to *reads of every region argument* (mirroring the C++
//!   `ArgEffect::Read` default), with a small writer table for the ops that
//!   name a destination.
//! * Tasks (submits, spmd scopes, kernel calls) are DAG nodes; their reads
//!   and writes come from the callee's declared directions or the explicit
//!   `arg_directions` attr.
//! * **Sequential loops** (`pl.range`, `pl.unroll`, `pl.pipeline`,
//!   `pl.while_`) produce ONE node carrying an uninterpreted bounded
//!   iteration variable `muN`; everything inside rolls up into it.
//! * **Parallel loops** (`pl.parallel`) produce TWO nodes — the body
//!   evaluated at `muNa` and `muNb` — with a `Distinct` constraint
//!   (`muNa != muNb`). A region both read and written by the body is a
//!   carried dependency: for a parallel loop that is a race warning, since
//!   the two instances are unordered.
//! * **Edges**: explicit `deps=[...]` always connect task nodes directly;
//!   memory-derived edges (Raw/Waw/War) connect nodes within one nesting
//!   level, and cross loop boundaries through the loop summary nodes.

use std::collections::HashMap;

use crate::ast::{BinOp, Expr, Function, IndexItem, Program, Stmt, UnaryOp};
use crate::consts;
use crate::dag::{
    Constraint, DagEdge, DagNode, EdgeKind, IterRef, NodeKind, ParRole, RegionId, ScopeMode,
    TaskDag, Warning,
};
use crate::error::SymirError;
use crate::sym::{SymVal, ITER_VAR_PREFIX};

/// Argument direction of a task parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Input,
    Output,
    OutputExisting,
    Inout,
    Scalar,
    Unknown,
}

impl Dir {
    /// (reads, writes) under this direction. `Unknown` is conservative:
    /// read and write (an unclassified buffer argument must never silently
    /// become a read-only one).
    fn effects(self) -> (bool, bool) {
        match self {
            Dir::Input => (true, false),
            Dir::Scalar => (false, false),
            Dir::Output | Dir::OutputExisting => (false, true),
            Dir::Inout | Dir::Unknown => (true, true),
        }
    }
}

/// One nesting scope: representative nodes plus per-level writer/reader maps.
struct Frame {
    nodes: Vec<usize>,
    writers: HashMap<RegionId, Vec<usize>>,
    readers: HashMap<RegionId, Vec<usize>>,
}

/// The interpreter state.
pub struct Interp<'a> {
    prog: &'a Program,
    dag: TaskDag,
    env: HashMap<String, SymVal>,
    frames: Vec<Frame>,
    iter_stack: Vec<IterRef>,
    guard_stack: Vec<String>,
    sigs: HashMap<String, FnSig>,
    term_counter: usize,
    region_counter: usize,
    loop_counter: usize,
    inline_depth: usize,
    /// Innermost tracking-scope mode stack (`pl.scope` / `pl.manual_scope`).
    scope_modes: Vec<ScopeMode>,
    /// Loop-id -> representative node ids (for ancestor checks).
    loop_nodes: HashMap<usize, Vec<usize>>,
}

/// Extracts the DAG from `prog`, interpreting `entry` (or the detected
/// orchestration function).
pub fn interpret(prog: &Program, entry: Option<&str>) -> Result<TaskDag, SymirError> {
    let sigs = signature_map(prog);
    let chosen = select_entry(prog, entry)?;
    let mut interp = Interp {
        prog,
        dag: TaskDag::default(),
        env: HashMap::new(),
        frames: Vec::new(),
        iter_stack: Vec::new(),
        guard_stack: Vec::new(),
        sigs,
        term_counter: 0,
        region_counter: 0,
        loop_counter: 0,
        inline_depth: 0,
        scope_modes: Vec::new(),
        loop_nodes: HashMap::new(),
    };
    interp.run(&chosen)?;
    interp.flag_manual_unordered();
    Ok(interp.dag)
}

/// Picks the function to interpret.
fn select_entry(prog: &Program, requested: Option<&str>) -> Result<Function, SymirError> {
    if let Some(name) = requested {
        return prog
            .functions
            .iter()
            .find(|f| f.name == name)
            .cloned()
            .ok_or_else(|| SymirError::Interp {
                line: 0,
                message: format!("no function named {name} in the program"),
            });
    }
    let orchestration = prog.functions.iter().find(|f| {
        f.decorators
            .iter()
            .any(|d| contains_attr_leaf(d, consts::DECORATOR_ORCHESTRATION))
    });
    orchestration
        .or_else(|| prog.functions.first())
        .cloned()
        .ok_or_else(|| SymirError::Interp {
            line: 0,
            message: crate::msgs::MSG_NO_FUNCTIONS.to_owned(),
        })
}

/// Full dotted text of a name-shaped expression.
fn expr_text(e: &Expr) -> String {
    match e {
        Expr::Name(n) => n.clone(),
        Expr::Attr(base, field) => format!("{}.{}", expr_text(base), field),
        Expr::Call { func, .. } => expr_text(func),
        _ => String::new(),
    }
}

/// True when `e` contains an attribute leaf named `leaf`.
fn contains_attr_leaf(e: &Expr, leaf: &str) -> bool {
    match e {
        Expr::Attr(base, field) => field == leaf || contains_attr_leaf(base, leaf),
        Expr::Call { func, args, kwargs } => {
            contains_attr_leaf(func, leaf)
                || args.iter().any(|a| contains_attr_leaf(a, leaf))
                || kwargs.iter().any(|(_, v)| contains_attr_leaf(v, leaf))
        }
        Expr::Subscript { base, indices } => {
            contains_attr_leaf(base, leaf)
                || indices.iter().any(|i| match i {
                    IndexItem::Expr(e) => contains_attr_leaf(e, leaf),
                    IndexItem::SliceAll => false,
                })
        }
        Expr::Tuple(items) | Expr::List(items) => items.iter().any(|a| contains_attr_leaf(a, leaf)),
        Expr::Dict(entries) => entries
            .iter()
            .any(|(k, v)| contains_attr_leaf(k, leaf) || contains_attr_leaf(v, leaf)),
        _ => false,
    }
}

/// Direction of one annotation expression.
fn direction_of(annotation: &Expr) -> Dir {
    if contains_attr_leaf(annotation, consts::DIRECTION_OUT) {
        return Dir::Output;
    }
    if contains_attr_leaf(annotation, consts::DIRECTION_INOUT) {
        return Dir::Inout;
    }
    for tensor_leaf in [
        consts::TYPE_TENSOR,
        consts::TYPE_DISTRIBUTED_TENSOR,
        consts::TYPE_TILE,
    ] {
        if contains_attr_leaf(annotation, tensor_leaf) {
            return Dir::Input;
        }
    }
    if contains_attr_leaf(annotation, consts::TYPE_SCALAR) {
        return Dir::Scalar;
    }
    Dir::Unknown
}

/// One parsed callee signature.
#[derive(Clone)]
struct FnSig {
    dirs: Vec<Dir>,
    /// `FunctionType.Inline` / `Opaque`: spliced at the call site, not a
    /// task — the body is interpreted in the caller's context.
    inline: bool,
}

/// Callee name → signature.
fn signature_map(prog: &Program) -> HashMap<String, FnSig> {
    let mut sigs = HashMap::new();
    for f in &prog.functions {
        let dirs: Vec<Dir> = f
            .params
            .iter()
            .map(|p| direction_of(&p.annotation))
            .collect();
        let inline = f
            .decorators
            .iter()
            .any(|d| contains_attr_leaf(d, consts::FN_TYPE_INLINE))
            || f.decorators
                .iter()
                .any(|d| contains_attr_leaf(d, consts::FN_TYPE_OPAQUE));
        let _ = sigs.insert(f.name.clone(), FnSig { dirs, inline });
    }
    sigs
}

impl<'a> Interp<'a> {
    fn run(&mut self, func: &Function) -> Result<(), SymirError> {
        self.dag.program = self.prog.name.clone();
        self.dag.entry = func.name.clone();
        let host = self.push_node(
            NodeKind::Host,
            format!("host:{}", func.name),
            Vec::new(),
            Vec::new(),
        );
        self.frames.push(Frame {
            nodes: vec![host],
            writers: HashMap::new(),
            readers: HashMap::new(),
        });
        for param in &func.params {
            let value = match direction_of(&param.annotation) {
                Dir::Scalar => self.fresh_named(format!("param.{}", param.name)),
                _ => SymVal::Region(self.fresh_region(&param.name)),
            };
            let _ = self.env.insert(param.name.clone(), value);
        }
        // Module-level constants: `NAME = pl.dynamic("NAME")` becomes a
        // symbolic dimension constant named by its string argument.
        for (name, value) in &self.prog.globals {
            let bound = match value {
                Expr::Call { func, args, .. } if leaf_name(func) == consts::CALL_DYNAMIC => {
                    match args.first() {
                        Some(Expr::Str(text)) => self.fresh_named(text.clone()),
                        _ => self.eval(value)?,
                    }
                }
                _ => self.eval(value)?,
            };
            let _ = self.env.insert(name.clone(), bound);
        }
        self.exec_block(&func.body)?;
        Ok(())
    }

    // -- construction helpers ---------------------------------------------------

    fn fresh_region(&mut self, name: &str) -> RegionId {
        let id = self.region_counter;
        self.region_counter += 1;
        self.dag.regions.push(name.to_owned());
        id
    }

    /// A named uninterpreted constant (no arguments).
    fn fresh_named(&mut self, name: String) -> SymVal {
        SymVal::Uninterp {
            name,
            args: Vec::new(),
        }
    }

    /// A numbered uninterpreted application `tN=name(args...)`.
    fn fresh_term(&mut self, name: String, args: Vec<SymVal>) -> SymVal {
        let id = self.term_counter;
        self.term_counter += 1;
        let prefix = crate::sym::TERM_PREFIX;
        SymVal::Uninterp {
            name: format!("{prefix}{id}={name}"),
            args,
        }
    }

    fn push_node(
        &mut self,
        kind: NodeKind,
        label: String,
        iter_ctx: Vec<IterRef>,
        guards: Vec<String>,
    ) -> usize {
        let id = self.dag.nodes.len();
        self.dag.nodes.push(DagNode {
            id,
            kind,
            label,
            reads: Default::default(),
            writes: Default::default(),
            parent_loop: self.iter_stack.last().map(|i| i.loop_id),
            iter_ctx,
            guards,
            line: 0,
            scope_mode: self.current_scope_mode(),
        });
        id
    }

    /// True when `maybe_ancestor` is a strict loop-ancestor of `node`
    /// (walks the parent_loop chain through the loop-id -> nodes map).
    fn is_loop_ancestor(&self, maybe_ancestor: usize, node: usize) -> bool {
        let mut current = self.dag.nodes.get(node).and_then(|n| n.parent_loop);
        while let Some(loop_id) = current {
            let Some(reps) = self.loop_nodes.get(&loop_id) else {
                return false;
            };
            if reps.contains(&maybe_ancestor) {
                return true;
            }
            current = self
                .dag
                .nodes
                .get(*reps.first().unwrap_or(&node))
                .and_then(|n| n.parent_loop);
        }
        false
    }

    fn add_edge(&mut self, from: usize, to: usize, kind: EdgeKind, regions: Vec<RegionId>) {
        if from == to {
            return;
        }
        // Ordering inside one loop body is by construction: the body is
        // interpreted in statement order, and a loop's effects roll up
        // into its own node. Edges between a loop and its own nested
        // loops (or the reverse) are spurious — they attributed body-op
        // writes to the loop node and closed cycles (red-demonstrated by
        // the indexer and gate fixtures after the write/create fixes).
        if self.is_loop_ancestor(from, to) || self.is_loop_ancestor(to, from) {
            return;
        }
        if self
            .dag
            .edges
            .iter()
            .any(|e| e.from == from && e.to == to && e.kind == kind && e.regions == regions)
        {
            return;
        }
        self.dag.edges.push(DagEdge {
            from,
            to,
            kind,
            regions,
        });
    }

    // -- statements ---------------------------------------------------------------

    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<(), SymirError> {
        for stmt in stmts {
            self.exec_stmt(stmt)?;
        }
        Ok(())
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<(), SymirError> {
        match stmt {
            Stmt::Assign { targets, value, .. } => {
                // Name a freshly allocated region after its binding: the
                // op name (`create`) carries no information; the variable
                // name (xg_buf, route_scores_buf, ...) is what the source
                // and the graph should agree on.
                let before = self.region_counter;
                let val = self.eval(value)?;
                if let SymVal::Region(region) = &val {
                    if *region >= before {
                        if let (Some(target), Some(name)) =
                            (targets.first(), self.dag.regions.get_mut(*region))
                        {
                            *name = target.clone();
                        }
                    }
                }
                self.bind_targets(targets, val)
            }
            Stmt::If { arms } => self.exec_if(arms),
            Stmt::For {
                index_vars,
                iter_arg_vars,
                iterator,
                body,
            } => self.exec_for(index_vars, iter_arg_vars, iterator, body),
            Stmt::While { cond, body } => self.exec_while(cond, body),
            Stmt::With {
                scope,
                binder,
                body,
            } => self.exec_with(scope, binder.as_deref(), body),
            Stmt::Return(value) => {
                if let Some(value) = value {
                    let _ = self.eval(value)?;
                }
                Ok(())
            }
            Stmt::Expr(expr) => {
                let _ = self.eval(expr)?;
                Ok(())
            }
        }
    }

    fn bind_targets(&mut self, targets: &[String], value: SymVal) -> Result<(), SymirError> {
        match (targets, &value) {
            ([single], _) => {
                let _ = self.env.insert(single.clone(), value);
                Ok(())
            }
            (_, SymVal::Tuple(items)) if targets.len() == items.len() => {
                for (name, item) in targets.iter().zip(items.iter()) {
                    let _ = self.env.insert(name.clone(), item.clone());
                }
                Ok(())
            }
            _ => Err(SymirError::Interp {
                line: 0,
                message: format!("cannot bind {targets:?} from {value}"),
            }),
        }
    }

    fn exec_if(&mut self, arms: &[(Option<Expr>, Vec<Stmt>)]) -> Result<(), SymirError> {
        let snapshot_writers = self.frame_writers();
        let snapshot_readers = self.frame_readers();
        let mut merged_writers = snapshot_writers.clone();
        let mut merged_readers = snapshot_readers.clone();
        for (cond, body) in arms {
            let guard = match cond {
                Some(cond) => self.eval(cond)?.to_string(),
                None => format!("!({})", self.guard_stack.len()),
            };
            self.guard_stack.push(guard);
            *self.frame_writers_mut() = snapshot_writers.clone();
            *self.frame_readers_mut() = snapshot_readers.clone();
            self.exec_block(body)?;
            let _ = self.guard_stack.pop();
            merge_maps(&mut merged_writers, &self.frame_writers());
            merge_maps(&mut merged_readers, &self.frame_readers());
        }
        *self.frame_writers_mut() = merged_writers;
        *self.frame_readers_mut() = merged_readers;
        Ok(())
    }

    fn exec_for(
        &mut self,
        index_vars: &[String],
        iter_arg_vars: &[String],
        iterator: &Expr,
        body: &[Stmt],
    ) -> Result<(), SymirError> {
        let (args, kwargs): (&[Expr], &[(String, Expr)]) = match iterator {
            Expr::Call { args, kwargs, .. } => (args, kwargs),
            _ => (&[], &[]),
        };
        let loop_id = self.loop_counter;
        self.loop_counter += 1;
        let mu = format!("{ITER_VAR_PREFIX}{loop_id}");
        let parallel = iterator_kind(iterator) == IterKind::Parallel;
        let (lo, hi) = self.loop_bounds(iterator_kind(iterator), loop_id, args);
        let iter_ctx = self.iter_stack.clone();
        let guards = self.guard_stack.clone();

        let nodes: Vec<usize> = if parallel {
            let a = self.push_node(
                NodeKind::ParInstance { role: ParRole::A },
                format!(
                    "{ITER_VAR_PREFIX}{loop_id}{}: [{lo}, {hi})",
                    consts::SUFFIX_PAR_A
                ),
                iter_ctx.clone(),
                guards.clone(),
            );
            let b = self.push_node(
                NodeKind::ParInstance { role: ParRole::B },
                format!(
                    "{ITER_VAR_PREFIX}{loop_id}{}: [{lo}, {hi})",
                    consts::SUFFIX_PAR_B
                ),
                iter_ctx.clone(),
                guards.clone(),
            );
            for var in [
                format!("{ITER_VAR_PREFIX}{loop_id}{}", consts::SUFFIX_PAR_A),
                format!("{ITER_VAR_PREFIX}{loop_id}{}", consts::SUFFIX_PAR_B),
            ] {
                self.dag.constraints.push(Constraint::Bounds {
                    loop_id,
                    var,
                    lo: lo.clone(),
                    hi: hi.clone(),
                });
            }
            self.dag.constraints.push(Constraint::Distinct {
                loop_id,
                a: format!("{ITER_VAR_PREFIX}{loop_id}{}", consts::SUFFIX_PAR_A),
                b: format!("{ITER_VAR_PREFIX}{loop_id}{}", consts::SUFFIX_PAR_B),
            });
            vec![a, b]
        } else {
            let node = self.push_node(
                NodeKind::SeqLoop,
                format!("{mu}: [{lo}, {hi})"),
                iter_ctx.clone(),
                guards.clone(),
            );
            let _ = self.loop_nodes.insert(loop_id, vec![node]);
            self.dag.constraints.push(Constraint::Bounds {
                loop_id,
                var: mu.clone(),
                lo,
                hi,
            });
            vec![node]
        };
        let _ = self.loop_nodes.insert(loop_id, nodes.clone());

        let iter_val = SymVal::Iter {
            loop_id,
            var: mu.clone(),
        };
        for name in index_vars {
            let _ = self.env.insert(name.clone(), iter_val.clone());
        }
        let init_values = self.init_values(kwargs)?;
        for (name, init) in iter_arg_vars.iter().zip(init_values) {
            let _ = self.env.insert(name.clone(), init);
        }

        let outer_writers = self.frame_writers();
        let outer_readers = self.frame_readers();

        self.frames.push(Frame {
            nodes: nodes.clone(),
            writers: HashMap::new(),
            readers: HashMap::new(),
        });
        self.iter_stack.push(IterRef {
            loop_id,
            var: mu.clone(),
        });
        self.exec_block(body)?;
        let inner_writers = self.frame_writers();
        let inner_readers = self.frame_readers();
        let _ = self.iter_stack.pop();
        let _ = self.frames.pop();

        // Roll inner effects up into the loop nodes.
        for node in &nodes {
            let dag_node = self.dag.nodes.get_mut(*node).ok_or_else(internal_node)?;
            for region in inner_writers.keys() {
                let _ = dag_node.writes.insert(*region);
            }
            for region in inner_readers.keys() {
                let _ = dag_node.reads.insert(*region);
            }
        }

        // Cross-boundary summary edges.
        for (region, writers) in &outer_writers {
            if inner_readers.contains_key(region) {
                for writer in writers {
                    for node in &nodes {
                        self.add_edge(*writer, *node, EdgeKind::Raw, vec![*region]);
                    }
                }
            }
            if inner_writers.contains_key(region) {
                for writer in writers {
                    for node in &nodes {
                        self.add_edge(*writer, *node, EdgeKind::Waw, vec![*region]);
                    }
                }
            }
        }
        for (region, readers) in &outer_readers {
            if inner_writers.contains_key(region) {
                for reader in readers {
                    for node in &nodes {
                        self.add_edge(*reader, *node, EdgeKind::War, vec![*region]);
                    }
                }
            }
        }

        // Parallel carry = unordered read+write of one region: a race.
        // ADVISORY ONLY: region-granular set intersection, superseded by
        // offset- and sync-aware pair reasoning — see
        // https://github.com/Servyi-Agent/pypto-symir/issues/1
        if parallel {
            for region in inner_writers.keys() {
                if inner_readers.contains_key(region) {
                    let name = self.dag.regions.get(*region).cloned().unwrap_or_default();
                    let mut roles = nodes.iter();
                    if let (Some(first), Some(second)) = (roles.next(), roles.last()) {
                        self.add_edge(*first, *second, EdgeKind::Race, vec![*region]);
                    }
                    for node in &nodes {
                        self.dag.warnings.push(Warning {
                            node: *node,
                            message: format!(
                                "parallel loop mu{loop_id}a/mu{loop_id}b carries region {name}: the body reads and writes it, so the two unordered instances race"
                            ),
                        });
                    }
                }
            }
        }

        for region in inner_writers.keys() {
            let _ = self.frame_writers_mut().insert(*region, nodes.clone());
        }
        for region in inner_readers.keys() {
            let entry = self.frame_readers_mut().entry(*region).or_default();
            for node in &nodes {
                if !entry.contains(node) {
                    entry.push(*node);
                }
            }
        }
        Ok(())
    }

    /// Symbolic bounds of a loop, as printable terms.
    fn loop_bounds(&mut self, kind: IterKind, loop_id: usize, args: &[Expr]) -> (String, String) {
        match kind {
            IterKind::While => (consts::ZERO_TERM.to_owned(), format!("trip_while{loop_id}")),
            _ => {
                let stop = match args.len() {
                    1 => args.first(),
                    _ => args.get(consts::BOUND_STOP_INDEX).or_else(|| args.first()),
                };
                let start = if args.len() > 1 {
                    self.term_text(args.first())
                } else {
                    consts::ZERO_TERM.to_owned()
                };
                (start, self.term_text(stop))
            }
        }
    }

    fn term_text(&mut self, expr: Option<&Expr>) -> String {
        match expr {
            Some(e) => match self.eval(e) {
                Ok(v) => v.to_string(),
                Err(_) => consts::UNKNOWN_TERM.to_owned(),
            },
            None => consts::UNKNOWN_TERM.to_owned(),
        }
    }

    /// Evaluates the `init_values` kwarg into initial carried values.
    fn init_values(&mut self, kwargs: &[(String, Expr)]) -> Result<Vec<SymVal>, SymirError> {
        for (key, value) in kwargs {
            if key == consts::KWARG_INIT_VALUES {
                return match self.eval(value)? {
                    SymVal::Tuple(items) => Ok(items),
                    single => Ok(vec![single]),
                };
            }
        }
        Ok(Vec::new())
    }

    /// Natural `while cond:` — one sequential loop node; the trip count is
    /// uninterpreted (`trip_whileN`) and the condition is recorded as the
    /// node's guard.
    fn exec_while(&mut self, cond: &Expr, body: &[Stmt]) -> Result<(), SymirError> {
        let loop_id = self.loop_counter;
        self.loop_counter += 1;
        let mu = format!("{ITER_VAR_PREFIX}{loop_id}");
        let lo = consts::ZERO_TERM.to_owned();
        let hi = format!("trip_while{loop_id}");
        let guard = self.eval(cond)?.to_string();
        let iter_ctx = self.iter_stack.clone();
        let mut guards = self.guard_stack.clone();
        guards.push(guard);
        let node = self.push_node(
            NodeKind::SeqLoop,
            format!("{mu}: [{lo}, {hi})"),
            iter_ctx.clone(),
            guards,
        );
        let _ = self.loop_nodes.insert(loop_id, vec![node]);
        self.dag.constraints.push(Constraint::Bounds {
            loop_id,
            var: mu.clone(),
            lo,
            hi,
        });
        let iter_val = SymVal::Iter {
            loop_id,
            var: mu.clone(),
        };
        let _ = iter_val;

        let outer_writers = self.frame_writers();
        let outer_readers = self.frame_readers();
        self.frames.push(Frame {
            nodes: vec![node],
            writers: HashMap::new(),
            readers: HashMap::new(),
        });
        self.iter_stack.push(crate::dag::IterRef {
            loop_id,
            var: mu.clone(),
        });
        self.exec_block(body)?;
        let inner_writers = self.frame_writers();
        let inner_readers = self.frame_readers();
        let _ = self.iter_stack.pop();
        let _ = self.frames.pop();

        let dag_node = self.dag.nodes.get_mut(node).ok_or_else(internal_node)?;
        for region in inner_writers.keys() {
            let _ = dag_node.writes.insert(*region);
        }
        for region in inner_readers.keys() {
            let _ = dag_node.reads.insert(*region);
        }
        for (region, writers) in &outer_writers {
            if inner_readers.contains_key(region) {
                for writer in writers {
                    self.add_edge(*writer, node, EdgeKind::Raw, vec![*region]);
                }
            }
            if inner_writers.contains_key(region) {
                for writer in writers {
                    self.add_edge(*writer, node, EdgeKind::Waw, vec![*region]);
                }
            }
        }
        for (region, readers) in &outer_readers {
            if inner_writers.contains_key(region) {
                for reader in readers {
                    self.add_edge(*reader, node, EdgeKind::War, vec![*region]);
                }
            }
        }
        for region in inner_writers.keys() {
            let _ = self.frame_writers_mut().insert(*region, vec![node]);
        }
        for region in inner_readers.keys() {
            let entry = self.frame_readers_mut().entry(*region).or_default();
            if !entry.contains(&node) {
                entry.push(node);
            }
        }
        Ok(())
    }

    /// The effective tracking mode: the innermost tracking scope, or Auto
    /// outside one (the compiler owns AUTO placement by default).
    fn current_scope_mode(&self) -> ScopeMode {
        self.scope_modes.last().copied().unwrap_or(ScopeMode::Auto)
    }

    /// Whether a `with` scope is a tracking scope, and in which mode.
    fn tracking_mode(scope: &Expr) -> Option<ScopeMode> {
        let leaf = leaf_name(scope);
        if leaf == consts::SCOPE_MANUAL {
            return Some(ScopeMode::Manual);
        }
        if leaf == consts::SCOPE_GENERIC {
            let manual = match scope {
                Expr::Call { kwargs, .. } => kwargs.iter().any(|(k, v)| {
                    k == consts::KWARG_MODE && contains_attr_leaf(v, consts::MODE_MANUAL)
                }),
                _ => false,
            };
            return Some(if manual {
                ScopeMode::Manual
            } else {
                ScopeMode::Auto
            });
        }
        None
    }

    fn exec_with(
        &mut self,
        scope: &Expr,
        binder: Option<&str>,
        body: &[Stmt],
    ) -> Result<(), SymirError> {
        let (args, kwargs) = match scope {
            Expr::Call { args, kwargs, .. } => (args.clone(), kwargs.clone()),
            _ => (Vec::new(), Vec::new()),
        };
        let leaf = leaf_name(scope);
        let tracking = Self::tracking_mode(scope);
        if let Some(mode) = tracking {
            self.scope_modes.push(mode);
        }
        let result = self.exec_with_inner(binder, body, &args, &kwargs, &leaf);
        if tracking.is_some() {
            let _ = self.scope_modes.pop();
        }
        result
    }

    fn exec_with_inner(
        &mut self,
        binder: Option<&str>,
        body: &[Stmt],
        args: &[Expr],
        kwargs: &[(String, Expr)],
        leaf: &str,
    ) -> Result<(), SymirError> {
        if leaf == consts::SCOPE_SPMD {
            let _ = args;
            let hint = self.scope_hint(kwargs);
            let iter_ctx = self.iter_stack.clone();
            let guards = self.guard_stack.clone();
            let spmd = consts::SCOPE_SPMD;
            let label = format!("{spmd}:{hint}");
            let node = self.push_node(
                NodeKind::Task {
                    callee: label.clone(),
                },
                label,
                iter_ctx,
                guards,
            );
            self.apply_explicit_deps(kwargs, node)?;
            self.run_scope_body(node, body)?;
            if let Some(binder) = binder {
                let _ = self.env.insert(binder.to_owned(), SymVal::Task { node });
            }
            return Ok(());
        }
        // Transparent scopes (`pl.scope`, `pl.at`, `pl.cluster`, ...).
        self.exec_block(body)
    }

    /// Interprets `body` as the interior of the scope node `node`, rolling
    /// effects into it and connecting outer writers/readers.
    fn run_scope_body(&mut self, node: usize, body: &[Stmt]) -> Result<(), SymirError> {
        let outer_writers = self.frame_writers();
        let outer_readers = self.frame_readers();
        self.frames.push(Frame {
            nodes: vec![node],
            writers: HashMap::new(),
            readers: HashMap::new(),
        });
        self.exec_block(body)?;
        let inner_writers = self.frame_writers();
        let inner_readers = self.frame_readers();
        let _ = self.frames.pop();
        self.connect_summary(
            node,
            &inner_writers,
            &inner_readers,
            &outer_writers,
            &outer_readers,
        )
    }

    /// Rolls inner effects into `node` and creates boundary edges.
    fn connect_summary(
        &mut self,
        node: usize,
        inner_writers: &HashMap<RegionId, Vec<usize>>,
        inner_readers: &HashMap<RegionId, Vec<usize>>,
        outer_writers: &HashMap<RegionId, Vec<usize>>,
        outer_readers: &HashMap<RegionId, Vec<usize>>,
    ) -> Result<(), SymirError> {
        let dag_node = self.dag.nodes.get_mut(node).ok_or_else(internal_node)?;
        for region in inner_writers.keys() {
            let _ = dag_node.writes.insert(*region);
        }
        for region in inner_readers.keys() {
            let _ = dag_node.reads.insert(*region);
        }
        for (region, writers) in outer_writers {
            if inner_readers.contains_key(region) {
                for writer in writers {
                    self.add_edge(*writer, node, EdgeKind::Raw, vec![*region]);
                }
            }
            if inner_writers.contains_key(region) {
                for writer in writers {
                    self.add_edge(*writer, node, EdgeKind::Waw, vec![*region]);
                }
            }
        }
        for (region, readers) in outer_readers {
            if inner_writers.contains_key(region) {
                for reader in readers {
                    self.add_edge(*reader, node, EdgeKind::War, vec![*region]);
                }
            }
        }
        for region in inner_writers.keys() {
            let _ = self.frame_writers_mut().insert(*region, vec![node]);
        }
        for region in inner_readers.keys() {
            let entry = self.frame_readers_mut().entry(*region).or_default();
            if !entry.contains(&node) {
                entry.push(node);
            }
        }
        Ok(())
    }

    fn scope_hint(&mut self, kwargs: &[(String, Expr)]) -> String {
        for (key, value) in kwargs {
            if key == consts::KWARG_NAME_HINT {
                if let Expr::Str(text) = value {
                    return text.clone();
                }
            }
        }
        format!("n{}", self.dag.nodes.len())
    }

    fn apply_explicit_deps(
        &mut self,
        kwargs: &[(String, Expr)],
        node: usize,
    ) -> Result<(), SymirError> {
        for (key, value) in kwargs {
            if key == consts::KWARG_DEPS {
                match value {
                    Expr::List(items) => {
                        for item in items {
                            self.add_dep_edge(item, node)?;
                        }
                    }
                    single => self.add_dep_edge(single, node)?,
                }
            }
        }
        Ok(())
    }

    fn add_dep_edge(&mut self, item: &Expr, node: usize) -> Result<(), SymirError> {
        if let SymVal::Task { node: producer } = self.eval(item)? {
            self.add_edge(producer, node, EdgeKind::Dep, Vec::new());
        }
        Ok(())
    }

    // -- expressions ----------------------------------------------------------------

    fn eval(&mut self, expr: &Expr) -> Result<SymVal, SymirError> {
        match expr {
            Expr::Int(v) => Ok(SymVal::Int(*v)),
            Expr::Float(v) => Ok(SymVal::Float(*v)),
            Expr::Bool(v) => Ok(SymVal::Bool(*v)),
            Expr::Str(v) => Ok(SymVal::Str(v.clone())),
            Expr::Name(name) => self
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| SymirError::Interp {
                    line: 0,
                    message: format!("unbound name {name}"),
                }),
            Expr::Attr(base, field) => Ok(SymVal::Str(format!("{}.{field}", expr_text(base)))),
            Expr::Tuple(items) | Expr::List(items) => {
                let mut values = Vec::new();
                for item in items {
                    values.push(self.eval(item)?);
                }
                Ok(SymVal::Tuple(values))
            }
            Expr::Dict(entries) => {
                let mut values = Vec::new();
                for (key, value) in entries {
                    values.push(self.eval(key)?);
                    values.push(self.eval(value)?);
                }
                Ok(self.fresh_term(consts::TERM_DICT.to_owned(), values))
            }
            Expr::Subscript { base, indices } => self.eval_subscript(base, indices),
            Expr::UnaryOp(op, inner) => {
                let value = self.eval(inner)?;
                Ok(match (op, &value) {
                    (UnaryOp::Neg, SymVal::Int(v)) => SymVal::Int(-v),
                    (UnaryOp::Neg, SymVal::Float(v)) => SymVal::Float(-v),
                    (UnaryOp::Not, SymVal::Bool(v)) => SymVal::Bool(!v),
                    _ => self.fresh_term(format!("{op}"), vec![value]),
                })
            }
            Expr::BinOp(op, lhs, rhs) => {
                let l = self.eval(lhs)?;
                let r = self.eval(rhs)?;
                if let Some(folded) = fold_binop(*op, &l, &r) {
                    return Ok(folded);
                }
                Ok(self.fresh_term(format!("{op}"), vec![l, r]))
            }
            Expr::Call { func, args, kwargs } => self.eval_call(func, args, kwargs),
        }
    }

    fn eval_subscript(&mut self, base: &Expr, indices: &[IndexItem]) -> Result<SymVal, SymirError> {
        let value = self.eval(base)?;
        if indices.len() == 1 {
            if let Some(IndexItem::Expr(inner)) = indices.first() {
                let idx = self.eval(inner);
                if let (SymVal::Tuple(items), Ok(SymVal::Int(i))) = (&value, idx) {
                    if i >= 0 {
                        if let Some(picked) = items.get(i.unsigned_abs() as usize) {
                            return Ok(picked.clone());
                        }
                    }
                }
            }
        }
        // Views keep the region identity.
        Ok(value)
    }

    fn eval_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<SymVal, SymirError> {
        let leaf = leaf_name(func);

        // `pl._dump.submit(inner, type)`: the wrapper's first argument is
        // the real submit call; `pl._dump.call(...)` is a recorded op or
        // kernel call.
        if dump_namespace(func) == Some(consts::MODULE_DUMP) {
            if leaf == consts::MODULE_DUMP_SUBMIT_CALL {
                if let Some(Expr::Call { func, args, kwargs }) = args.first() {
                    return self.eval_call(func, args, kwargs);
                }
                return Ok(SymVal::Unit);
            }
            if leaf == consts::MODULE_DUMP_OP_CALL {
                return self.eval_dump_call(args, kwargs);
            }
        }
        if leaf == consts::CALL_SUBMIT || leaf == consts::CALL_SPMD_SUBMIT {
            return self.eval_submit(args, kwargs);
        }
        if leaf == consts::CALL_YIELD || leaf == consts::CALL_COND {
            return match args.first() {
                Some(arg) => self.eval(arg),
                None => Ok(SymVal::Unit),
            };
        }
        if leaf == consts::CALL_FUNC_ATTR {
            return Ok(SymVal::Unit);
        }
        // Cross-function call: `self.fn(...)` or a known callee. Inline /
        // Opaque functions are spliced into the caller's context; InCore
        // and friends are task launches.
        if let Some(callee) = self.callee_of(func) {
            if self.sigs.get(&callee).is_some_and(|sig| sig.inline) {
                return self.eval_inline_call(&callee, args);
            }
            if self.sigs.contains_key(&callee) {
                return self.launch_task(&callee, args, &[]);
            }
        }
        // Plain op call: uninterpreted with default read effects.
        let mut values = Vec::new();
        for arg in args {
            values.push(self.eval(arg)?);
        }
        self.op_call(&leaf, &values)
    }

    /// `pl._dump.call(op_or_fn, [payload...], type, attrs=...)`: a recorded
    /// op, or a recorded kernel call when the head names a program function.
    fn eval_dump_call(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<SymVal, SymirError> {
        let Some(head) = args.first() else {
            return Ok(SymVal::Unit);
        };
        let payload = match args.get(consts::DUMP_PAYLOAD_INDEX) {
            Some(Expr::List(items)) => items.clone(),
            _ => Vec::new(),
        };
        if let Some(callee) = self.callee_of(head) {
            if self.sigs.get(&callee).is_some_and(|sig| sig.inline) {
                return self.eval_inline_call(&callee, &payload);
            }
            if self.sigs.contains_key(&callee) {
                // The recorded form binds one name to the kernel's result;
                // the (result, task-id) tuple belongs to the pl.submit
                // source form only.
                if let SymVal::Tuple(items) = self.launch_task(&callee, &payload, kwargs)? {
                    return Ok(items.first().cloned().unwrap_or(SymVal::Unit));
                }
                return self.launch_task(&callee, &payload, kwargs);
            }
        }
        let op = expr_text(head);
        let mut values = Vec::new();
        for item in &payload {
            values.push(self.eval(item)?);
        }
        self.op_call(&op, &values)
    }

    /// The callee name when `func` is `self.fn` or a bare program function.
    fn callee_of(&self, func: &Expr) -> Option<String> {
        match func {
            Expr::Attr(base, field) if matches!(**base, Expr::Name(ref n) if n == consts::NAME_SELF) => {
                Some(field.clone())
            }
            Expr::Name(name) => Some(name.clone()),
            _ => None,
        }
    }

    /// Interprets an inline function's body in the caller's context: the
    /// parameters bind the caller's (already symbolic) argument values, so
    /// regions and task ids flow through unchanged.
    fn eval_inline_call(&mut self, callee: &str, payload: &[Expr]) -> Result<SymVal, SymirError> {
        if self.inline_depth >= consts::INLINE_DEPTH_LIMIT {
            return Err(SymirError::Interp {
                line: 0,
                message: format!("inline call depth exceeds {}", consts::INLINE_DEPTH_LIMIT),
            });
        }
        let Some(function) = self.prog.functions.iter().find(|f| f.name == callee) else {
            return Ok(SymVal::Unit);
        };
        let params = function.params.clone();
        let body = function.body.clone();
        self.inline_depth += 1;
        let saved_env = self.env.clone();
        for (param, arg) in params.iter().zip(payload.iter()) {
            let value = self.eval(arg)?;
            let _ = self.env.insert(param.name.clone(), value);
        }
        self.exec_block(&body)?;
        // The return expression (if any) is evaluated in the inline scope.
        let mut result = SymVal::Unit;
        for stmt in &body {
            if let crate::ast::Stmt::Return(Some(value)) = stmt {
                result = self.eval(value)?;
            }
        }
        self.env = saved_env;
        self.inline_depth -= 1;
        Ok(result)
    }

    fn eval_submit(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<SymVal, SymirError> {
        let Some(callee_expr) = args.first() else {
            return Ok(SymVal::Unit);
        };
        let payload: Vec<Expr> = args.iter().skip(consts::SKIP_CALLEE).cloned().collect();
        let callee = self
            .callee_of(callee_expr)
            .unwrap_or_else(|| expr_text(callee_expr));
        self.launch_task(&callee, &payload, kwargs)
    }

    /// Creates a task node with effects from directions and deps; returns
    /// `(result, task_id)`.
    fn launch_task(
        &mut self,
        callee: &str,
        payload: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<SymVal, SymirError> {
        let iter_ctx = self.iter_stack.clone();
        let guards = self.guard_stack.clone();
        let node = self.push_node(
            NodeKind::Task {
                callee: callee.to_owned(),
            },
            callee.to_owned(),
            iter_ctx,
            guards,
        );
        self.apply_explicit_deps(kwargs, node)?;
        let directions = self.task_directions(callee, payload, kwargs);
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for (arg, dir) in payload.iter().zip(directions) {
            let value = self.eval(arg)?;
            let mut arg_regions = Vec::new();
            value.regions(&mut arg_regions);
            let (do_read, do_write) = dir.effects();
            if do_read {
                for region in arg_regions {
                    reads.push(region);
                }
            }
            if do_write {
                let mut destinations = Vec::new();
                value.dest_regions(&mut destinations);
                for region in destinations {
                    writes.push(region);
                }
            }
        }
        self.record_task_effects(node, &reads, &writes)?;
        // The submit's return aliases its output argument — the compiler
        // does the same (GenerateSubmitReturnAliases), so the result
        // denotes that storage rather than a fresh value.
        let result = match writes.first() {
            Some(region) => SymVal::Region(*region),
            None => self.fresh_named(format!("launch.{callee}")),
        };
        Ok(SymVal::Tuple(vec![result, SymVal::Task { node }]))
    }

    /// Directions for a task: explicit `arg_directions` attr wins, else the
    /// callee's declared signature, else conservative Unknown. Payload args
    /// bind the callee's *leading* parameters (the C++ Submit invariant).
    fn task_directions(
        &self,
        callee: &str,
        payload: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Vec<Dir> {
        let dirs = dump_arg_directions(kwargs)
            .or_else(|| self.sigs.get(callee).map(|sig| sig.dirs.clone()))
            .unwrap_or_default();
        if dirs.len() == payload.len() {
            return dirs;
        }
        if dirs.len() > payload.len() {
            return dirs.iter().take(payload.len()).copied().collect();
        }
        vec![Dir::Unknown; payload.len()]
    }

    /// Memory effects of an op call at the current level; fresh allocations
    /// bind a new region.
    fn op_call(&mut self, op: &str, values: &[SymVal]) -> Result<SymVal, SymirError> {
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        if is_alloc_op(op) {
            let region = self.fresh_region(op);
            writes.push(region);
            self.record_effects(&reads, &writes)?;
            return Ok(SymVal::Region(region));
        }
        collect_op_effects(op, values, &mut reads, &mut writes);
        self.record_effects(&reads, &writes)?;
        // Memory-destinating ops (assemble/store/write) return their
        // destination REGION, not a term wrapping the producer tree: the
        // value is the buffer after the write (the compiler's own return
        // aliasing), and read-transparency must not carry the producer
        // history into later readers — those regions were consumed here,
        // by this node (red-demonstrated by the gate table: the topk
        // inherited the matmul's xg_buf/gate_w reads transitively).
        if let Some(primary) = writes.first() {
            return Ok(SymVal::Region(*primary));
        }
        Ok(self.fresh_term(op.to_owned(), values.to_vec()))
    }

    /// Applies reads/writes at the current nesting level, creating edges to
    /// the frame's representative nodes.
    fn record_effects(
        &mut self,
        reads: &[RegionId],
        writes: &[RegionId],
    ) -> Result<(), SymirError> {
        let frame_nodes = self
            .frames
            .last()
            .map(|f| f.nodes.clone())
            .ok_or_else(internal_frame)?;
        for region in reads {
            for target in &frame_nodes {
                let dag_node = self.dag.nodes.get_mut(*target).ok_or_else(internal_node)?;
                let _ = dag_node.reads.insert(*region);
            }
            if let Some(writers) = self.frame().writers.get(region).cloned() {
                for writer in writers {
                    for target in &frame_nodes {
                        self.add_edge(writer, *target, EdgeKind::Raw, vec![*region]);
                    }
                }
            }
            let entry = self.frame().readers.entry(*region).or_default();
            for target in &frame_nodes {
                if !entry.contains(target) {
                    entry.push(*target);
                }
            }
        }
        for region in writes {
            for target in &frame_nodes {
                let dag_node = self.dag.nodes.get_mut(*target).ok_or_else(internal_node)?;
                let _ = dag_node.writes.insert(*region);
            }
            if let Some(writers) = self.frame().writers.get(region).cloned() {
                for writer in writers {
                    for target in &frame_nodes {
                        self.add_edge(writer, *target, EdgeKind::Waw, vec![*region]);
                    }
                }
            }
            if let Some(readers) = self.frame().readers.get(region).cloned() {
                for reader in readers {
                    if !frame_nodes.contains(&reader) {
                        for target in &frame_nodes {
                            self.add_edge(reader, *target, EdgeKind::War, vec![*region]);
                        }
                    }
                }
            }
            let _ = self.frame().writers.insert(*region, frame_nodes.clone());
        }
        Ok(())
    }

    /// Task-specific effect recording: edges target the task node itself.
    fn record_task_effects(
        &mut self,
        node: usize,
        reads: &[RegionId],
        writes: &[RegionId],
    ) -> Result<(), SymirError> {
        for region in reads {
            let dag_node = self.dag.nodes.get_mut(node).ok_or_else(internal_node)?;
            let _ = dag_node.reads.insert(*region);
            if let Some(writers) = self.frame().writers.get(region).cloned() {
                for writer in writers {
                    self.add_edge(writer, node, EdgeKind::Raw, vec![*region]);
                }
            }
            let entry = self.frame().readers.entry(*region).or_default();
            if !entry.contains(&node) {
                entry.push(node);
            }
        }
        for region in writes {
            let dag_node = self.dag.nodes.get_mut(node).ok_or_else(internal_node)?;
            let _ = dag_node.writes.insert(*region);
            if let Some(writers) = self.frame().writers.get(region).cloned() {
                for writer in writers {
                    self.add_edge(writer, node, EdgeKind::Waw, vec![*region]);
                }
            }
            if let Some(readers) = self.frame().readers.get(region).cloned() {
                for reader in readers {
                    if reader != node {
                        self.add_edge(reader, node, EdgeKind::War, vec![*region]);
                    }
                }
            }
            let _ = self.frame().writers.insert(*region, vec![node]);
        }
        Ok(())
    }

    /// Flags derived edges that nothing will order: no `dep` edge covers
    /// the pair, and at least one endpoint was launched under manual
    /// tracking (so the runtime's OverlapMap inference skipped it).
    fn flag_manual_unordered(&mut self) {
        let dep_pairs: Vec<(usize, usize)> = self
            .dag
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Dep)
            .flat_map(|e| [(e.from, e.to), (e.to, e.from)])
            .collect();
        let mut findings: Vec<(usize, String)> = Vec::new();
        for edge in &self.dag.edges {
            if edge.kind == EdgeKind::Dep {
                continue;
            }
            if dep_pairs.contains(&(edge.from, edge.to)) {
                continue;
            }
            let manual = [edge.from, edge.to].iter().any(|id| {
                self.dag
                    .nodes
                    .get(*id)
                    .is_some_and(|n| n.scope_mode == ScopeMode::Manual)
            });
            if !manual {
                continue;
            }
            let region = edge
                .regions
                .first()
                .and_then(|r| self.dag.regions.get(*r).cloned())
                .unwrap_or_default();
            findings.push((edge.to, manual_unordered_message(&region)));
        }
        for (node, message) in findings {
            self.dag.warnings.push(Warning { node, message });
        }
    }

    fn frame(&mut self) -> &mut Frame {
        self.frames
            .last_mut()
            .expect("frame stack is never empty — run() pushes one")
    }

    fn frame_writers(&mut self) -> HashMap<RegionId, Vec<usize>> {
        self.frames
            .last()
            .map(|f| f.writers.clone())
            .unwrap_or_default()
    }

    fn frame_readers(&mut self) -> HashMap<RegionId, Vec<usize>> {
        self.frames
            .last()
            .map(|f| f.readers.clone())
            .unwrap_or_default()
    }

    fn frame_writers_mut(&mut self) -> &mut HashMap<RegionId, Vec<usize>> {
        &mut self
            .frames
            .last_mut()
            .expect("frame stack is never empty — run() pushes one")
            .writers
    }

    fn frame_readers_mut(&mut self) -> &mut HashMap<RegionId, Vec<usize>> {
        &mut self
            .frames
            .last_mut()
            .expect("frame stack is never empty — run() pushes one")
            .readers
    }
}

/// Warning text for an unordered manual-scope access (kept literal-free).
fn manual_unordered_message(region: &str) -> String {
    format!(
        "manual-scope unordered access on region {region}: the dataflow edge is derived but nothing establishes it (no deps= edge, and manual mode skips inference for at least one endpoint)"
    )
}

fn internal_frame() -> SymirError {
    SymirError::Interp {
        line: 0,
        message: crate::msgs::MSG_INTERNAL_NO_FRAME.to_owned(),
    }
}

fn internal_node() -> SymirError {
    SymirError::Interp {
        line: 0,
        message: crate::msgs::MSG_INTERNAL_NODE_RANGE.to_owned(),
    }
}

fn merge_maps(target: &mut HashMap<RegionId, Vec<usize>>, source: &HashMap<RegionId, Vec<usize>>) {
    for (region, nodes) in source {
        let entry = target.entry(*region).or_default();
        for node in nodes {
            if !entry.contains(node) {
                entry.push(*node);
            }
        }
    }
}

/// The iterator forms.
#[derive(PartialEq, Eq)]
enum IterKind {
    Range,
    Parallel,
    Unroll,
    Pipeline,
    While,
}

/// Classifies a for-iterator expression.
fn iterator_kind(iterator: &Expr) -> IterKind {
    let leaf = leaf_name(iterator);
    if leaf == consts::ITER_PARALLEL {
        return IterKind::Parallel;
    }
    if leaf == consts::ITER_UNROLL {
        return IterKind::Unroll;
    }
    if leaf == consts::ITER_PIPELINE {
        return IterKind::Pipeline;
    }
    if leaf == consts::ITER_WHILE {
        return IterKind::While;
    }
    IterKind::Range
}

/// Some("pl") when `expr` is `pl.x`; None otherwise (namespace detector).
fn top_namespace(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Name(name) => Some(name.clone()),
        Expr::Attr(base, _) => top_namespace(base),
        _ => None,
    }
}

/// The inner namespace of a `pl.<ns>.<leaf>` call target: Some("pl._dump")
/// style membership check helper — returns the namespace segment when the
/// chain is exactly three deep under `pl`.
fn dump_namespace(func: &Expr) -> Option<&str> {
    if let Expr::Attr(base, _) = func {
        if top_namespace(func) == Some(consts::MODULE_PL.to_owned()) {
            if let Expr::Attr(_, ns) = &**base {
                return Some(ns);
            }
        }
    }
    None
}

/// The attribute leaf of an expression (`a.b.c` → `c`).
fn leaf_name(expr: &Expr) -> String {
    match expr {
        Expr::Attr(_, field) => field.clone(),
        Expr::Name(name) => name.clone(),
        Expr::Call { func, .. } => leaf_name(func),
        _ => String::new(),
    }
}

/// Reads `attrs={"arg_directions": [pl.adir.*...]}` out of call kwargs.
fn dump_arg_directions(kwargs: &[(String, Expr)]) -> Option<Vec<Dir>> {
    for (key, value) in kwargs {
        if key == consts::KWARG_ATTRS {
            if let Expr::Dict(entries) = value {
                for (k, v) in entries {
                    if matches!(k, Expr::Str(s) if s == consts::KWARG_ARG_DIRECTIONS) {
                        if let Expr::List(items) = v {
                            return Some(items.iter().map(dir_of_attr).collect());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Maps a `pl.adir.*` attr value to a [`Dir`].
fn dir_of_attr(expr: &Expr) -> Dir {
    let text = expr_text(expr);
    if text == adir_path(consts::ADIR_INPUT) {
        return Dir::Input;
    }
    if text == adir_path(consts::ADIR_OUTPUT) {
        return Dir::Output;
    }
    if text == adir_path(consts::ADIR_OUTPUT_EXISTING) {
        return Dir::OutputExisting;
    }
    if text == adir_path(consts::ADIR_INOUT) {
        return Dir::Inout;
    }
    if text == adir_path(consts::ADIR_SCALAR) {
        return Dir::Scalar;
    }
    Dir::Unknown
}

/// The dotted text of one `pl.adir.X` reference.
fn adir_path(leaf: &str) -> String {
    let pl = consts::MODULE_PL;
    let adir = consts::MODULE_ADIR;
    format!("{pl}.{adir}.{leaf}")
}

/// True for the allocating ops.
fn is_alloc_op(op: &str) -> bool {
    op == consts::OP_CREATE_TENSOR
        || op == consts::OP_TENSOR_CREATE
        || op == consts::OP_CREATE_TILE
        || op == consts::OP_BUFFER_ALLOC
        || op == consts::OP_TILE_ALLOC
}

/// Folds concrete binary operations.
fn fold_binop(op: BinOp, l: &SymVal, r: &SymVal) -> Option<SymVal> {
    match (op, l, r) {
        (BinOp::Add, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Int(a.checked_add(*b)?)),
        (BinOp::Sub, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Int(a.checked_sub(*b)?)),
        (BinOp::Mul, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Int(a.checked_mul(*b)?)),
        (BinOp::FloorDiv, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Int(a.checked_div(*b)?)),
        (BinOp::Mod, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Int(a.checked_rem(*b)?)),
        (BinOp::Eq, a, b) => Some(SymVal::Bool(a == b)),
        (BinOp::Ne, a, b) => Some(SymVal::Bool(a != b)),
        (BinOp::Lt, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Bool(a < b)),
        (BinOp::Le, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Bool(a <= b)),
        (BinOp::Gt, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Bool(a > b)),
        (BinOp::Ge, SymVal::Int(a), SymVal::Int(b)) => Some(SymVal::Bool(a >= b)),
        (BinOp::And, SymVal::Bool(a), SymVal::Bool(b)) => Some(SymVal::Bool(*a && *b)),
        (BinOp::Or, SymVal::Bool(a), SymVal::Bool(b)) => Some(SymVal::Bool(*a || *b)),
        _ => None,
    }
}

/// Op-specific effect classification: named writers, everyone else reads.
fn collect_op_effects(
    op: &str,
    values: &[SymVal],
    reads: &mut Vec<RegionId>,
    writes: &mut Vec<RegionId>,
) {
    if op == consts::OP_TENSOR_ASSEMBLE {
        dest_regions_in(values.first(), writes);
        regions_in(values.get(consts::DUMP_PAYLOAD_INDEX), reads);
        return;
    }
    if op == consts::OP_WRITE {
        // write(dst, [idx], value): writes the destination, reads the value.
        dest_regions_in(values.first(), writes);
        for value in values.iter().skip(consts::SKIP_CALLEE) {
            let mut found = Vec::new();
            value.regions(&mut found);
            reads.extend(found);
        }
        return;
    }
    if op == consts::OP_STORE {
        regions_in(values.first(), reads);
        // The 3-arg form ends with the destination; the 4-arg tile form
        // ends with a shape list — the destination is the last argument
        // that actually denotes storage.
        let mut dest: Vec<crate::dag::RegionId> = Vec::new();
        for value in values.iter().rev() {
            dest_regions_in(Some(value), &mut dest);
            if !dest.is_empty() {
                break;
            }
        }
        writes.extend(dest);
        return;
    }
    for value in values {
        let mut found = Vec::new();
        value.regions(&mut found);
        reads.extend(found);
    }
}

/// Destination regions of an optional value.
fn dest_regions_in(value: Option<&SymVal>, out: &mut Vec<RegionId>) {
    if let Some(value) = value {
        value.dest_regions(out);
    }
}

/// Regions referenced by an optional value.
fn regions_in(value: Option<&SymVal>, out: &mut Vec<RegionId>) {
    if let Some(value) = value {
        value.regions(out);
    }
}
