//! The symbolic task DAG produced by the interpreter.
//!
//! Nodes are tasks (kernel submits), loop summaries, and the host entry
//! block. Edges carry their origin: explicit `deps=` (Dep), or derived from
//! memory effects (Raw/Waw/War). Constraints record the uninterpreted
//! bounded loop variables: one `mu` per sequential loop, two *distinct*
//! `mu`s per parallel loop.

use std::collections::BTreeSet;

use crate::consts;

use serde::Serialize;

/// Identity of a memory region (tensor / buffer / tile allocation).
pub type RegionId = usize;

/// Identity of a loop context.
pub type LoopId = usize;

/// A whole extracted DAG.
#[derive(Debug, Default, Serialize)]
pub struct TaskDag {
    /// Program (class) name.
    pub program: String,
    /// Entry function the DAG was extracted from.
    pub entry: String,
    /// All nodes, indexed by id.
    pub nodes: Vec<DagNode>,
    /// All edges.
    pub edges: Vec<DagEdge>,
    /// Uninterpreted-variable constraints.
    pub constraints: Vec<Constraint>,
    /// Race-shaped findings.
    pub warnings: Vec<Warning>,
    /// Region names, indexed by id.
    pub regions: Vec<String>,
}

/// What a node represents.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeKind {
    /// Host-side orchestration code (the function body outside tasks/loops).
    Host,
    /// An explicit task launch (submit / spmd_submit / kernel call).
    Task { callee: String },
    /// A sequential loop summary (one uninterpreted iteration variable).
    SeqLoop,
    /// One of the two instances of a parallel loop (two uninterpreted
    /// iteration variables constrained distinct).
    ParInstance { role: ParRole },
}

/// Which side of a parallel loop's two-instance encoding a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParRole {
    /// The `mu_a` instance.
    A,
    /// The `mu_b` instance.
    B,
}

/// One DAG node with its summary effects.
#[derive(Debug, Clone, Serialize)]
pub struct DagNode {
    pub id: usize,
    pub kind: NodeKind,
    /// Human-readable label (callee / loop description).
    pub label: String,
    /// Regions read.
    pub reads: BTreeSet<RegionId>,
    /// Regions written.
    pub writes: BTreeSet<RegionId>,
    /// Enclosing loop (nested loop / task context).
    pub parent_loop: Option<LoopId>,
    /// The uninterpreted iteration variables this node executes under.
    pub iter_ctx: Vec<IterRef>,
    /// Symbolic guard (the if-conditions in effect), if any.
    pub guards: Vec<String>,
    /// Source line of the defining statement.
    pub line: usize,
}

/// A reference to a loop's uninterpreted iteration variable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IterRef {
    pub loop_id: LoopId,
    /// The variable's printed name (`mu0`, `mu0a`, `mu0b`).
    pub var: String,
}

/// One DAG edge.
#[derive(Debug, Clone, Serialize)]
pub struct DagEdge {
    pub from: usize,
    pub to: usize,
    pub kind: EdgeKind,
    /// The regions responsible for a derived edge.
    pub regions: Vec<RegionId>,
}

/// How the edge came about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Explicit `deps=[...]` on a submit.
    Dep,
    /// Read after write.
    Raw,
    /// Write after write.
    Waw,
    /// Write after read.
    War,
    /// A parallel loop's two unordered instances both touch a carried
    /// region: a data race, not an ordering.
    Race,
}

/// An uninterpreted-variable constraint.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Constraint {
    /// `lo <= mu < hi` for a loop's iteration variable.
    Bounds {
        loop_id: LoopId,
        var: String,
        lo: String,
        hi: String,
    },
    /// `mu_a != mu_b` for a parallel loop's two instances.
    Distinct {
        loop_id: LoopId,
        a: String,
        b: String,
    },
}

/// A finding worth reporting.
#[derive(Debug, Clone, Serialize)]
pub struct Warning {
    pub node: usize,
    pub message: String,
}

impl TaskDag {
    /// Structural invariants every extracted DAG must satisfy. Run by
    /// [`crate::extract`] on every interpretation — the runtime assertion
    /// layer for graph-shaped regressions (duplicate edges, dangling
    /// endpoints, warnings naming absent nodes).
    pub fn validate(&self) -> Result<(), crate::SymirError> {
        use crate::SymirError;
        let node_count = self.nodes.len();
        for edge in &self.edges {
            if edge.from >= node_count || edge.to >= node_count {
                return Err(SymirError::Interp {
                    line: 0,
                    message: format!(
                        "edge n{} -> n{} references a node outside the graph ({node_count} nodes)",
                        edge.from, edge.to
                    ),
                });
            }
            if edge.from == edge.to {
                return Err(SymirError::Interp {
                    line: 0,
                    message: format!("self-edge on node {}", edge.from),
                });
            }
        }
        for later in self.edges.iter().skip(consts::EDGE_UNIQUENESS_SKIP) {
            let twins = self
                .edges
                .iter()
                .filter(|e| {
                    e.from == later.from
                        && e.to == later.to
                        && e.kind == later.kind
                        && e.regions == later.regions
                })
                .count();
            if twins > consts::EDGE_SINGLE_OCCURRENCE {
                return Err(SymirError::Interp {
                    line: 0,
                    message: format!(
                        "duplicate edge n{} -> n{} ({:?}) appears {twins} times",
                        later.from, later.to, later.kind
                    ),
                });
            }
        }
        for warning in &self.warnings {
            if warning.node >= node_count {
                return Err(SymirError::Interp {
                    line: 0,
                    message: format!(
                        "warning names node {} outside the graph ({node_count} nodes)",
                        warning.node
                    ),
                });
            }
        }
        Ok(())
    }

    /// The printed name of a node's iteration variables.
    pub fn iter_names(&self, node: usize) -> Vec<String> {
        self.nodes
            .get(node)
            .map(|n| n.iter_ctx.iter().map(|i| i.var.to_owned()).collect())
            .unwrap_or_default()
    }
}
