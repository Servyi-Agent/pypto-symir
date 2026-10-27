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
/// Kernel-selection flag for DSL-source inputs.
pub const FLAG_KERNEL: &str = "--kernel";
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
/// WhyML disjunction separator in emitted formulas.
pub const SEP_WHY3_OR: &str = " \\/ ";
/// The mlw emission mode.
pub const MODE_MLW: &str = "mlw";
/// The check mode (walk + emit + why3 in one command).
pub const MODE_CHECK: &str = "check";
/// The why3 binary run by --spmd check.
pub const WHY3_BIN: &str = "why3";
/// why3 subcommand for proving.
pub const WHY3_ARG_PROVE: &str = "prove";
/// why3 prover-selection flag.
pub const WHY3_ARG_PROVER: &str = "-P";
/// why3 timelimit flag.
pub const WHY3_ARG_TIME: &str = "-t";
/// The configured prover used by --spmd check.
pub const WHY3_PROVER: &str = "Z3,4.16.0";
/// Per-goal time limit (seconds) for --spmd check.
pub const WHY3_TIMELIMIT: &str = "120";
/// Temp directory for check-mode models.
pub const CHECK_TEMP_DIR: &str = "pypto-smt-check";
/// Why3's goal-line marker parsed by check mode.
pub const VERDICT_GOAL: &str = "Goal ";
/// Why3's valid verdict marker.
pub const VERDICT_VALID: &str = "Valid";
/// Hint appended when both IR parsing and the compiler step fail.
pub const MSG_INPUT_CONTRACT: &str =
    "input is neither printed IR nor compilable DSL source (@pl.jit kernel)";
/// Message when why3 is not installed.
pub const MSG_NO_WHY3: &str =
    "why3 is not installed; --spmd check needs it (why3 config detect with Z3)";
/// Default interpreter that has pypto importable (overridable via
/// PYPTO_SMT_PYTHON).
pub const PYPTO_PYTHON: &str = "/root/pypto/.venv/bin/python";
/// Default PYTHONPATH for the compiler step (overridable via
/// PYPTO_SMT_PATH).
pub const PYPTO_PATH: &str =
    "/root/pypto:/root/pypto-lib:/root/pypto/runtime:/root/pypto/runtime/python";
/// The interpreter's run-from-string flag.
pub const PY_ARG_CODE: &str = "-c";
/// Empty positional argument for the driver's optional kernel name.
pub const EMPTY_ARG: &str = "";
/// PYTHONPATH env name for the compiler step.
pub const ENV_PYTHONPATH: &str = "PYTHONPATH";
/// Runtime pin check bypass for the compiler step.
pub const ENV_SKIP_PIN_KEY: &str = "PYPTO_SKIP_RUNTIME_PIN_CHECK";
/// Value disabling the runtime pin check.
pub const ENV_SKIP_PIN_VAL: &str = "1";
/// Env-var name overriding the interpreter.
pub const ENV_PYTHON: &str = "PYPTO_SMT_PYTHON";
/// Env-var name overriding PYTHONPATH.
pub const ENV_PATH: &str = "PYPTO_SMT_PATH";
/// Preflight line for a successful compiler step.
pub const MSG_COMPILED: &str = "compiled via pypto specialize:";
/// Line prefix naming the analyzed kernel (DSL inputs).
pub const MSG_KERNEL: &str = "kernel:";
/// The driver's kernel-name stderr tag.
pub const KERNEL_TAG: &str = "kernel:";
/// Message when the compiler interpreter cannot be spawned.
pub const MSG_NO_COMPILER: &str = "cannot run the pypto interpreter";
/// Note when a program contains no spmd bodies.
pub const MSG_NO_SITES: &str = "no spmd sites in this program (symir analyzes orchestration-level task graphs)";
/// The first spmd site index.
pub const FIRST_SITE: usize = 0;
/// Provenance line for emitted models.
pub const MSG_PROVENANCE: &str = "source: printed PyPTO IR (pre-pass), first spmd site";
/// Verdict-line markers parsed by the why3 end-to-end tests.
pub const GOAL_MARKER: &str = "Goal waw_out_n1.";
pub const VERDICT_MARKER: &str = "Prover result";
/// Goal-line marker of the exhaustiveness (coverage) goal.
pub const COVERAGE_MARKER: &str = "Goal covered_";
/// The unordered-axes distinctness clause head in emitted pair goals.
/// Worker-axis distinctness term in pair goals.
pub const WORKER_DISTINCT: &str = "w_a <> w_b";
/// The unordered-axes distinctness clause head in emitted pair goals.
pub const CLAUSE_UNORDERED_AXES: &str = "w_a <> w_b \\/ (0 + 1 * k0_a) <> (0 + 1 * k0_b)";
/// The author-contract lemma injected by the conditional-race test: the
/// table maps distinct blocks to distinct rows (a data fact, stated as an
/// axiom-style lemma for the demonstration).
pub const TABLE_AXIOM_LEMMA: &str = "  lemma table_distinct : forall i j. 0 <= i < 4 -> 0 <= j < 4 -> i <> j ->\n    (read_table i) <> (read_table j)\n\n  predicate overlap";
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
