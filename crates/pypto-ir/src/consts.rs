//! Constants of the printed PyPTO IR surface: module and op names,
//! iterators, scopes, kwargs, arg directions, and operator tokens.
//! Everything that names a syntactic or semantic element of the
//! printed IR lives here, shared by every tool over that surface.

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

pub const ITER_RANGE: &str = "range";

pub const ITER_PARALLEL: &str = "parallel";

pub const ITER_UNROLL: &str = "unroll";

pub const ITER_PIPELINE: &str = "pipeline";

pub const ITER_WHILE: &str = "while_";

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

pub const KWARG_DEPS: &str = "deps";

pub const KWARG_CORE_NUM: &str = "core_num";

pub const KWARG_INIT_VALUES: &str = "init_values";

pub const KWARG_ATTRS: &str = "attrs";

pub const KWARG_ARG_DIRECTIONS: &str = "arg_directions";

pub const KWARG_NAME_HINT: &str = "name_hint";

/// How many leading arguments of a submit call form the callee reference.
pub const SKIP_CALLEE: usize = 1;

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

pub const DIRECTION_OUT: &str = "Out";

pub const DIRECTION_INOUT: &str = "InOut";

/// Annotation leaves that denote region-valued (memory-backed) parameters.
pub const TYPE_TENSOR: &str = "Tensor";

pub const TYPE_DISTRIBUTED_TENSOR: &str = "DistributedTensor";

pub const TYPE_TILE: &str = "Tile";

/// Annotation leaf for scalar parameters.
pub const TYPE_SCALAR: &str = "Scalar";

pub const ADIR_INPUT: &str = "input";

pub const ADIR_OUTPUT: &str = "output";

pub const ADIR_OUTPUT_EXISTING: &str = "output_existing";

pub const ADIR_INOUT: &str = "inout";

pub const ADIR_SCALAR: &str = "scalar";

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

/// The parameter-name convention for a program-class method's self slot.
pub const BUILTIN_TRUE: &str = "True";

pub const BUILTIN_FALSE: &str = "False";

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

/// The attribute/namespace separator.
pub const CHAR_DOT: char = '.';

/// Separator between a numbered term's display prefix and its op name.
pub const CHAR_EQ: char = '=';

/// How far one step through the flat line list moves.
pub const LINE_ADVANCE: usize = 1;

/// The program name `env::args()` yields before real arguments.
pub const SKIP_PROGRAM_NAME: usize = 1;

pub const TOK_AND: &str = "and";

pub const TOK_OR: &str = "or";

pub const TOK_NOT: &str = "not";
