//! Output emitters: DOT graph and a human-readable summary.

use crate::consts;
use crate::dag::{EdgeKind, NodeKind, TaskDag};

/// Renders the DAG as a Graphviz DOT digraph.
pub fn to_dot(dag: &TaskDag) -> String {
    let q = consts::DOT_QUOTE;
    let lbl = consts::DOT_LABEL;
    let shp = consts::DOT_SHAPE;
    let sty = consts::DOT_STYLE;
    let kw = consts::DOT_DIGRAPH;
    let mut out = format!("{kw} {q}{}{q} {{\n", dag.program);
    out.push_str(consts::DOT_RANK_LR);
    for node in &dag.nodes {
        let shape = match node.kind {
            NodeKind::Host => consts::DOT_BOX,
            _ => consts::DOT_ELLIPSE,
        };
        let reads = region_list(dag, &node.reads.iter().copied().collect::<Vec<_>>());
        let writes = region_list(dag, &node.writes.iter().copied().collect::<Vec<_>>());
        let iters = node
            .iter_ctx
            .iter()
            .map(|i| i.var.clone())
            .collect::<Vec<_>>();
        let guards = if node.guards.is_empty() {
            String::new()
        } else {
            format!("\\nwhen: {}", node.guards.join(" && "))
        };
        let effects = format!("R:[{reads}] W:[{writes}]");
        let iter_note = if iters.is_empty() {
            String::new()
        } else {
            format!("\\niter: {}", iters.join(consts::SEP_COMMA_SPACE))
        };
        out.push_str(&format!(
            "  n{} [{lbl}={q}{}\\n{}{}{}{q}, {shp}={q}{shape}{q}];\n",
            node.id, node.label, effects, iter_note, guards,
        ));
    }
    for edge in &dag.edges {
        let style = match edge.kind {
            EdgeKind::Dep => consts::DOT_SOLID,
            _ => consts::DOT_DASHED,
        };
        let regions = region_list(dag, &edge.regions);
        let tag = match edge.kind {
            EdgeKind::Dep => consts::EDGE_TAG_DEP.to_owned(),
            EdgeKind::Raw => format!("raw [{regions}]"),
            EdgeKind::Waw => format!("waw [{regions}]"),
            EdgeKind::War => format!("war [{regions}]"),
            EdgeKind::Race => format!("race [{regions}]"),
        };
        out.push_str(&format!(
            "  n{} -> n{} [{lbl}={q}{tag}{q}, {sty}={q}{style}{q}];\n",
            edge.from, edge.to,
        ));
    }
    out.push_str(consts::DOT_CLOSE);
    out
}

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

/// Comma-joined region names.
fn region_list(dag: &TaskDag, regions: &[usize]) -> String {
    regions
        .iter()
        .map(|r| {
            dag.regions
                .get(*r)
                .cloned()
                .unwrap_or_else(|| format!("R{r}"))
        })
        .collect::<Vec<_>>()
        .join(consts::SEP_COMMA_SPACE)
}
