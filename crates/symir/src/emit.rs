//! Output emitters: the report. DOT/SVG/PNG drawing lives in
//! [`crate::graph`] (petgraph + graphviz — no hand-formatted graph output
//! in this crate).

use crate::dag::TaskDag;

/// Renders the constraints and warnings as a plain-text report.
pub fn to_report(dag: &TaskDag) -> String {
    let mut out = format!("program: {}  entry: {}\n", dag.program, dag.entry);
    out.push_str(&format!(
        "nodes: {}  edges: {}\n",
        dag.nodes.len(),
        dag.edges.len()
    ));
    for c in &dag.constraints {
        out.push_str(&format!("constraint: {c:?}\n"));
    }
    for w in &dag.warnings {
        let label = dag
            .nodes
            .get(w.node)
            .map(|n| n.label.clone())
            .unwrap_or_default();
        out.push_str(&format!("warning: [{label}] {}\n", w.message));
    }
    out
}
