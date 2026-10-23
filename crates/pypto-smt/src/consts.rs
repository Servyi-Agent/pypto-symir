//! Named constants for the pypto-smt tool.

/// The DSL module.
pub const MODULE_PL: &str = "pl";
/// The spmd scope leaf name.
pub const SCOPE_SPMD: &str = "spmd";
/// The spmd name-hint kwarg.
pub const KWARG_NAME_HINT: &str = "name_hint";
/// Integer zero, the base of worker/step-index ranges.
pub const INT_ZERO: i64 = 0;
/// z3 symbol: first worker.
pub const SYM_WORKER_A: &str = "w1";
/// z3 symbol: second worker.
pub const SYM_WORKER_B: &str = "w2";
/// z3 symbol: first worker's step index.
pub const SYM_STEP_INDEX_A: &str = "k1";
/// z3 symbol: second worker's step index.
pub const SYM_STEP_INDEX_B: &str = "k2";
/// CLI usage line.
/// File-read failure prose.
pub const MSG_READ_FAILURE: &str = "cannot read";
/// argv[0] is the program name: exactly one more argument is expected.
pub const EXPECTED_ARG_COUNT: usize = 2;
/// argv[1] is the input path.
pub const PATH_ARG_INDEX: usize = 1;
/// Spmd subcommand flag.
pub const FLAG_SPMD: &str = "--spmd";
/// Site-selection flag (`--site N`, 0-based in `find_spmd` order).
pub const FLAG_SITE: &str = "--site";
/// Namespace prefix of tensor-level (view-forming) ops.
pub const NAMESPACE_TENSOR: &str = "tensor";
/// The on-chip tile op namespace.
pub const NAMESPACE_TILE: &str = "tile";
/// Prefix for havoced writer-context (carried) symbols.
pub const HAVOC_PREFIX: &str = "carry_";
/// Hint name for bare yield_ rebinds.
pub const HAVOC_YIELD: &str = "rebind";
/// Prefix for fresh private regions from `tensor.create`.
pub const FRESH_REGION_PREFIX: &str = "fresh_";
/// The InCore function-type marker.
pub const FN_TYPE_INCORE: &str = "InCore";
/// Character kept verbatim by the identifier sanitizer.
pub const SANITIZE_KEEP: char = '_';
/// Replacement character for non-alphanumeric characters.
pub const SANITIZE_FALLBACK: char = '_';
/// Index vars after the first (already bound) in a loop header.
pub const EXTRA_INDEX_VARS_START: usize = 1;
/// WhyML conjunction separator in emitted formulas.
pub const SEP_WHY3_AND: &str = " /\\ ";
/// The mlw emission mode.
pub const MODE_MLW: &str = "mlw";
/// The first spmd site index.
pub const FIRST_SITE: usize = 0;
/// Provenance line for emitted models.
pub const MSG_PROVENANCE: &str = "source: printed PyPTO IR (pre-pass), first spmd site";
/// Refusal header.
pub const MSG_REFUSED: &str = "refused";
/// Updated usage line covering the spmd modes.
pub const MSG_USAGE: &str = "usage: pypto-smt <printed-ir.py> [--spmd mlw]";
/// Report line for one spmd site.
pub const MSG_SITE_LINE: &str = "spmd site";
/// Lemma verdicts.
pub const MSG_LEMMA_HOLDS: &str = "stride-partition lemma holds: distinct workers never share a work item";
pub const MSG_LEMMA_BROKEN: &str =
    "stride-partition lemma BROKEN at these parameters: workers can collide";
/// The worker's z3/WhyML symbol.
pub const SYM_WORKER: &str = "w";
/// Prefix for loop instance variables.
pub const MU_PREFIX: &str = "mu";
/// Prefix for tensor-dimension constant symbols.
pub const DIM_PREFIX: &str = "dim_";
/// Prefix for uninterpreted `tensor.read` value functions.
pub const READ_PREFIX: &str = "read_";
/// Construct names used in refusals. Each string is the construct's own
/// name (a necessity, not a choice), so the sites carry an explicit allow.
#[allow(inline_literal)]
pub const WHILE: &str = "while";
#[allow(inline_literal)]
pub const LOOP_CARRY: &str = "loop-carried iteration args";
#[allow(inline_literal)]
pub const ITERATOR: &str = "loop iterator";
#[allow(inline_literal)]
pub const RANGE_STEP: &str = "range step";
#[allow(inline_literal)]
pub const SCALAR_OP: &str = "scalar op";
#[allow(inline_literal)]
pub const EXPRESSION: &str = "expression";
#[allow(inline_literal)]
pub const CALL: &str = "call";
#[allow(inline_literal)]
pub const CONDITION: &str = "condition";
#[allow(inline_literal)]
pub const REGION: &str = "region";
#[allow(inline_literal)]
pub const INDEX_LIST: &str = "index list";
#[allow(inline_literal)]
pub const STORE_SHAPE: &str = "store shape";
#[allow(inline_literal)]
pub const SITE_SELECT: &str = "spmd site selection";
pub const SCOPE_SPMD_TOOL: &str = "spmd";
/// Refusal prose.
pub const MSG_NO_SPMD_BODY: &str = "function has no spmd scope body";
pub const MSG_WHILE_REFUSED: &str =
    "while-shaped control cannot be characterized exactly by instance variables; phantom iterations would fabricate accesses";
pub const MSG_ITERATOR_REFUSED: &str =
    "only pl.range / pl.unroll / pl.parallel iterators are modeled";
pub const MSG_STEP_REFUSED: &str = "range step must be a positive integer literal";
pub const MSG_CALL_REFUSED: &str = "unknown callee; refusing rather than approximating";
pub const MSG_NONSCALAR: &str = "expected a scalar value";
pub const MSG_STORE_SHAPE: &str = "stored tile's window shape is not computable";
pub const MSG_ASSEMBLE_REFUSED: &str =
    "tensor.assemble writes through views; offset composition not modeled yet";
/// Goal name prefixes.
#[allow(inline_literal)]
pub const GOAL_WAW: &str = "waw";
#[allow(inline_literal)]
pub const GOAL_RAW: &str = "raw";
/// Uninterpreted on-chip tile-read value function name.
#[allow(inline_literal)]
pub const TILE_READ_FUNC: &str = "tile_read";
/// Instance bound symbols (documentation of the encoding).
pub const BOUND_WORKER_A: &str = "0 <= w_a < width";
pub const BOUND_WORKER_B: &str = "0 <= w_b < width";
