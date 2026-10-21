//! Constants used only by symir (interpretation, graph and report
//! emission). The IR surface constants live in `pypto_ir::consts` and
//! are re-exported so `crate::consts::*` keeps resolving everywhere.

pub use pypto_ir::consts::*;

/// DOT's directed-graph keyword.
pub const DOT_DIGRAPH: &str = "digraph";

/// DOT node/edge attribute keys.
pub const DOT_LABEL: &str = "label";

pub const DOT_SHAPE: &str = "shape";

pub const DOT_STYLE: &str = "style";

pub const DOT_DASHED: &str = "dashed";

pub const DOT_SOLID: &str = "solid";

pub const DOT_BOX: &str = "box";

pub const DOT_ELLIPSE: &str = "ellipse";

/// The quote character used when assembling DOT attributes at runtime.
pub const DOT_QUOTE: char = '"';

pub const FLAG_EMIT: &str = "--emit";

pub const FLAG_ENTRY: &str = "--entry";

pub const FLAG_OUTPUT: &str = "-o";

pub const FORMAT_DOT: &str = "dot";

pub const FORMAT_JSON: &str = "json";

pub const FORMAT_REPORT: &str = "report";

pub const FORMAT_SVG: &str = "svg";

pub const FORMAT_PNG: &str = "png";

pub const MSG_USAGE: &str =
    "usage: symir <printed-ir.py> [--emit dot|json|report] [--entry NAME] [-o FILE]";

pub const MSG_READ_FAILURE: &str = "cannot read";

pub const MSG_WRITE_FAILURE: &str = "cannot write";

pub const MSG_PIPELINE_FAILURE: &str = "cannot process";

pub const MSG_JSON_FAILURE: &str = "cannot serialize:";

/// DOT left-to-right layout directive line.
pub const DOT_RANK_LR: &str = "  rankdir=LR;\n";

/// Edge tag for explicit deps.
pub const EDGE_TAG_DEP: &str = "dep";

/// Placeholder term when a bound cannot be evaluated.
pub const UNKNOWN_TERM: &str = "?";

/// Term name for dictionary literals.
pub const TERM_DICT: &str = "dict";

/// Argument position of the loop stop bound in `(start, stop, step)`.
pub const BOUND_STOP_INDEX: usize = 1;

/// Argument position of the recorded payload in `_dump.call`.
pub const DUMP_PAYLOAD_INDEX: usize = 1;

/// Host-node header in graph labels.
pub const GRAPH_KIND_HOST: &str = "host:";

/// The two-character backslash-n sequence that must never appear in
/// labels (petgraph escapes it into literal text).
pub const BACKSLASH_N: &str = "\\n";

/// A real newline inside graphviz labels.
pub const LABEL_NEWLINE: &str = "\n";

/// Guard separator in node labels.
pub const SEP_GUARD: &str = " && ";

/// Edge tags.
pub const EDGE_TAG_RAW: &str = "raw";

pub const EDGE_TAG_WAW: &str = "waw";

pub const EDGE_TAG_WAR: &str = "war";

pub const EDGE_TAG_RACE: &str = "race";

/// The graphviz renderer binary.
pub const GRAPHVIZ_DOT: &str = "dot";

/// graphviz output flag.
pub const GRAPHVIZ_OUTPUT_FLAG: &str = "-o";

/// Temp-file prefix for rendered graphs.
pub const DOT_TEMP_PREFIX: &str = "symir-";

/// Characters kept verbatim in file-name fragments.
pub const NAME_SAFE_DOT: char = '.';

pub const NAME_SAFE_FALLBACK: char = '_';

/// DOT graph closing line.
pub const DOT_CLOSE: &str = "}\n";

/// Duplicate-edge scan skips nothing after this offset (the first edge
/// cannot duplicate anything before it).
pub const EDGE_UNIQUENESS_SKIP: usize = 1;

/// How many occurrences of one edge shape are allowed.
pub const EDGE_SINGLE_OCCURRENCE: usize = 1;

/// List separator used when joining terms.
pub const SEP_COMMA_SPACE: &str = ", ";
