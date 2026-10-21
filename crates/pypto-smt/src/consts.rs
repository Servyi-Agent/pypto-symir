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
pub const MSG_USAGE: &str =
    "usage: pypto-smt <printed-ir.py>  (lists spmd sites and checks the stride-partition lemma)";
/// File-read failure prose.
pub const MSG_READ_FAILURE: &str = "cannot read";
/// argv[0] is the program name: exactly one more argument is expected.
pub const EXPECTED_ARG_COUNT: usize = 2;
/// argv[1] is the input path.
pub const PATH_ARG_INDEX: usize = 1;
/// Report line for one spmd site.
pub const MSG_SITE_LINE: &str = "spmd site";
/// Lemma verdicts.
pub const MSG_LEMMA_HOLDS: &str = "stride-partition lemma holds: distinct workers never share a work item";
pub const MSG_LEMMA_BROKEN: &str =
    "stride-partition lemma BROKEN at these parameters: workers can collide";
