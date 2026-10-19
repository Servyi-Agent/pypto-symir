//! Keyword and op-name constants.
//!
//! The interpreter classifies parsed constructs by comparing against these
//! names — every literal lives here with a semantic name (the servyi
//! `inline_literal` policy), never inline at a use site.

// --- module / receiver names ------------------------------------------------

/// The DSL namespace everything hangs off.
pub const MODULE_PL: &str = "pl";
/// The module-level self reference used for cross-function calls.
pub const NAME_SELF: &str = "self";
/// The inner namespace the printer uses for lossless op roundtrips.
pub const MODULE_DUMP: &str = "_dump";
/// The argument-direction enum namespace.
pub const MODULE_ADIR: &str = "adir";
/// The function-type leaf marking an orchestration entry.
pub const DECORATOR_ORCHESTRATION: &str = "Orchestration";

// --- iterator forms ----------------------------------------------------------

pub const ITER_RANGE: &str = "range";
pub const ITER_PARALLEL: &str = "parallel";
pub const ITER_UNROLL: &str = "unroll";
pub const ITER_PIPELINE: &str = "pipeline";
pub const ITER_WHILE: &str = "while_";

// --- special call names ------------------------------------------------------

pub const CALL_YIELD: &str = "yield_";
pub const CALL_COND: &str = "cond";
pub const CALL_SUBMIT: &str = "submit";
pub const CALL_SPMD_SUBMIT: &str = "spmd_submit";
pub const CALL_FUNC_ATTR: &str = "func_attr";
pub const CALL_DYNAMIC: &str = "dynamic";
/// Decorator leaf marking inline functions.
pub const FN_TYPE_INLINE: &str = "Inline";
/// Decorator leaf marking opaque functions (interpreted inline).
pub const FN_TYPE_OPAQUE: &str = "Opaque";
/// How deep inline splicing may nest before giving up.
pub const INLINE_DEPTH_LIMIT: usize = 8;
/// The leaf of `pl._dump.call(...)`.
pub const MODULE_DUMP_OP_CALL: &str = "call";
/// The leaf of `pl._dump.submit(...)`.
pub const MODULE_DUMP_SUBMIT_CALL: &str = "submit";
/// The two parallel-instance suffixes (`mu0a` / `mu0b`).
pub const SUFFIX_PAR_A: &str = "a";
pub const SUFFIX_PAR_B: &str = "b";
/// The default lower bound term for 1-argument loops.
pub const ZERO_TERM: &str = "0";

// --- submit kwargs -----------------------------------------------------------

pub const KWARG_DEPS: &str = "deps";
pub const KWARG_CORE_NUM: &str = "core_num";
pub const KWARG_INIT_VALUES: &str = "init_values";
pub const KWARG_ATTRS: &str = "attrs";
pub const KWARG_ARG_DIRECTIONS: &str = "arg_directions";
pub const KWARG_NAME_HINT: &str = "name_hint";

/// How many leading arguments of a submit call form the callee reference.
pub const SKIP_CALLEE: usize = 1;

// --- scope heads (`with ...`) ------------------------------------------------

pub const SCOPE_GENERIC: &str = "scope";
/// The manual-tracking opt-out scope.
pub const SCOPE_MANUAL: &str = "manual_scope";
/// The mode kwarg of `pl.scope(...)`.
pub const KWARG_MODE: &str = "mode";
/// The MANUAL enum leaf of `pl.ScopeMode`.
pub const MODE_MANUAL: &str = "MANUAL";
pub const SCOPE_AT: &str = "at";
pub const SCOPE_SPMD: &str = "spmd";
pub const SCOPE_CLUSTER: &str = "cluster";
pub const SCOPE_GRAPH: &str = "graph";

// --- annotation leaves that carry direction ----------------------------------

pub const DIRECTION_OUT: &str = "Out";
pub const DIRECTION_INOUT: &str = "InOut";

/// Annotation leaves that denote region-valued (memory-backed) parameters.
pub const TYPE_TENSOR: &str = "Tensor";
pub const TYPE_DISTRIBUTED_TENSOR: &str = "DistributedTensor";
pub const TYPE_TILE: &str = "Tile";
/// Annotation leaf for scalar parameters.
pub const TYPE_SCALAR: &str = "Scalar";

// --- `arg_directions` attr values --------------------------------------------

pub const ADIR_INPUT: &str = "input";
pub const ADIR_OUTPUT: &str = "output";
pub const ADIR_OUTPUT_EXISTING: &str = "output_existing";
pub const ADIR_INOUT: &str = "inout";
pub const ADIR_SCALAR: &str = "scalar";

// --- ops with memory effects (writers; everything else defaults to
// functional reads, mirroring the C++ `ArgEffect::Read` default).
// Classified by the attribute LEAF of the call (`pl.tensor.assemble` →
// `assemble`), since the namespaces agree on these shapes. ---------------

/// `tensor.assemble(out, src, offsets)` — writes the `out` region.
pub const OP_TENSOR_ASSEMBLE: &str = "assemble";
/// `buffer.store` / `tile.store` — writes its destination region.
pub const OP_STORE: &str = "store";
/// Ops whose result is a view of their first argument: as destinations
/// they denote that argument's storage, not the reads nested in the term.
pub const VIEW_FORMING_OPS: &[&str] = &["slice", "reshape", "view", "assemble"];
/// Element write: `write(dst, [idx], value)`.
pub const OP_WRITE: &str = "write";
/// Allocating ops introduce a fresh region. The printed forms vary:
/// `pl.create_tensor` and `pl.tensor.create` (leaf `create`), `pl.full`.
pub const OP_CREATE_TENSOR: &str = "create_tensor";
pub const OP_TENSOR_CREATE: &str = "create";
/// Constant initializer (`tile.full(..., value=const)`): a materialized
/// VALUE, deliberately not an allocation — scratch constants are
/// task-private and must not order anything (a cycle appeared when it
/// was modeled as storage; red-demonstrated by the indexer fixture).
pub const OP_FULL: &str = "full";
pub const OP_CREATE_TILE: &str = "create_tile";
pub const OP_BUFFER_ALLOC: &str = "alloc";
pub const OP_TILE_ALLOC: &str = "alloc_tile";

// --- dot emission ------------------------------------------------------------

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

// --- misc --------------------------------------------------------------------

/// The parameter-name convention for a program-class method's self slot.
pub const BUILTIN_TRUE: &str = "True";
pub const BUILTIN_FALSE: &str = "False";

// --- operator tokens recognized during parsing -------------------------------

pub const TOK_PLUS: &str = "+";
pub const TOK_MINUS: &str = "-";
pub const TOK_STAR: &str = "*";
pub const TOK_SLASH: &str = "/";
pub const TOK_DOUBLE_SLASH: &str = "//";
pub const TOK_PERCENT: &str = "%";
pub const TOK_EQ: &str = "==";
pub const TOK_NE: &str = "!=";
pub const TOK_LT: &str = "<";
pub const TOK_LE: &str = "<=";
pub const TOK_GT: &str = ">";
pub const TOK_GE: &str = ">=";

/// Separator between a numbered term's display prefix and its op name.
pub const CHAR_EQ: char = '=';
/// How far one step through the flat line list moves.
pub const LINE_ADVANCE: usize = 1;

// --- CLI ----------------------------------------------------------------------

/// The program name `env::args()` yields before real arguments.
pub const SKIP_PROGRAM_NAME: usize = 1;
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
pub const TOK_AND: &str = "and";
pub const TOK_OR: &str = "or";
pub const TOK_NOT: &str = "not";
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
