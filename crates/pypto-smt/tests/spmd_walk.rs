//! The SPMD body walk: access log on the real DeepSeek indexer dump,
//! legality refusals for the un-modellable class, and the emitted WhyML
//! model (solver runs gated on why3 being installed).

// Test expectations read better inline; the servyi `inline_literal`
// policy sanctions silencing sites with an explicit allow.
#![allow(unknown_lints)] // stable toolchain: the servyi lints exist only under the driver
#![allow(inline_literal)]
// Tests are the sanctioned exemption for panic/unwrap (servyi
// `non_test_panic_allow`); this file is an integration-test crate.
#![allow(clippy::unwrap_used, clippy::panic)]

use pypto_ir::parse_program;
use pypto_smt::consts;
use pypto_smt::term::Term;
use pypto_smt::walk::{walk_spmd, Dir};

const INDEXER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../symir/tests/fixtures/deepseek_indexer_pre.py"
);

fn indexer_body() -> pypto_smt::walk::SpmdBody {
    let src = std::fs::read_to_string(INDEXER).unwrap();
    walk_spmd(&parse_program(&src).unwrap(), 0).unwrap()
}

/// The real score body: two guarded stores to `output`, reads of the
/// metadata plane and the payload, and the worker-strided work loop.
#[test]
fn access_log_matches_real_score_body() {
    let body = indexer_body();
    assert_eq!(body.width, 32);
    let stores: Vec<_> = body
        .accesses
        .iter()
        .filter(|a| a.dir == Dir::Write && a.region == "output")
        .collect();
    assert_eq!(stores.len(), 2, "the if/else store pair");
    for store in &stores {
        assert_eq!(store.offsets.len(), 2);
        assert_eq!(store.shapes.len(), 2);
        assert!(
            !matches!(store.guards, pypto_smt::term::Bool::B(true)),
            "stores are guarded by metadata conditions"
        );
        assert!(store.coords.len() >= 3, "worker + work/pair loops in scope");
    }
    let regions: Vec<&str> = body.accesses.iter().map(|a| a.region.as_str()).collect();
    for metadata in ["lengths", "block_table", "request_ids"] {
        assert!(regions.contains(&metadata), "{metadata} is read");
    }
    assert!(regions.contains(&"query"), "payload is loaded");
}

/// The work loop is the worker-strided partition: lo = get_block_idx(),
/// step = 32 = width.
#[test]
fn work_loop_is_worker_strided() {
    let body = indexer_body();
    let work = body
        .loops
        .iter()
        .find(|l| matches!(&l.lo, Term::BlockIdx))
        .expect("the work loop starts at the block index");
    assert_eq!(work.step, pypto_smt::term::Term::Int(body.width));
}

/// A body containing `while` must be refused by name — phantom iterations
/// would fabricate accesses (unsound for coverage), unlike ignored sync
/// which only loses ordering.
#[test]
fn while_body_is_refused() {
    let src = "\
V = pl.dynamic(\"V\")

@pl.program
class W:
    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32]):
        with pl.spmd(2, name_hint=\"while_spmd\"):
            i: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX)
            while i < 8:
                t: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [i, 0], [1, 8], [1, 8], target_memory=pl.Mem.Vec)
                i = i + 1
        return x
";
    let program = parse_program(src).unwrap();
    let refused = walk_spmd(&program, 0).unwrap_err();
    assert_eq!(refused.construct, "while");
}

/// The emitted model contains the partition lemma and both check kinds.
#[test]
fn emitted_model_has_lemma_and_goals() {
    let body = indexer_body();
    let mlw = pypto_smt::mlw::emit_mlw(&body, "indexer score (pre-pass)");
    assert!(mlw.contains("lemma stride_partition"));
    assert!(mlw.contains("goal waw_output"), "the Q2 goal for output");
    assert!(
        mlw.contains("goal covered_output") || mlw.contains("SKIPPED — body carries state"),
        "the Q3 goal for output, or the documented carry-skip"
    );
    assert!(mlw.contains("function read_lengths"), "metadata plane declared");
}

/// All three spmd sites of the real indexer walk clean and emit goals.
#[test]
fn all_indexer_spmd_sites_walk() {
    let src = std::fs::read_to_string(INDEXER).unwrap();
    let program = parse_program(&src).unwrap();
    for site in 0..3 {
        let body = walk_spmd(&program, site).unwrap();
        assert!(
            body.accesses.iter().any(|a| a.dir == Dir::Write),
            "site {site} stores something"
        );
        let mlw = pypto_smt::mlw::emit_mlw(&body, "indexer");
        assert!(mlw.contains("goal waw_"), "site {site} has a Q2 goal");
    }
}

/// End to end through why3 itself: Q2 must come back Valid on the real
/// kernel. Ignored by default — CI has no why3; run locally with
/// `cargo test -p pypto-smt -- --ignored`.
#[test]
#[ignore = "requires why3 (present in the dev environment, not in CI)"]
fn why3_proves_output_waw_disjoint() {
    let body = indexer_body();
    let mlw = pypto_smt::mlw::emit_mlw(&body, "indexer score (pre-pass)");
    let dir = std::env::temp_dir().join("pypto-smt-why3");
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join("score.mlw");
    std::fs::write(&file, mlw).unwrap();
    let out = std::process::Command::new("why3")
        .arg("prove")
        .arg("-P")
        .arg("Z3,4.16.0")
        .arg("-t")
        .arg("60")
        .arg(&file)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(
        text.contains("Goal waw_output_n1.") && text.contains("Valid"),
        "Q2 must be proved; why3 said:\n{text}"
    );
}

/// The flagship spmd example (pypto's own 09_paged_attention_spmd,
/// frontend dump committed under fixtures/): five dispatch-form sites at
/// width 4. All five walk under the havoc-context policy — carried state
/// (yield_/init_values/loop-exit accumulations) quantified freely, InCore
/// callees inlined, runtime stride from get_block_num.
#[test]
fn paged_attention_spmd_all_sites_walk() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/paged_attention_spmd.py"
    ))
    .unwrap();
    let program = parse_program(&src).unwrap();
    for site in 0..5 {
        let body = walk_spmd(&program, site).unwrap();
        assert_eq!(body.width, 4);
        assert!(!body.havocs.is_empty(), "carried state is havoced");
        assert!(
            body.accesses.iter().any(|a| a.dir == Dir::Write),
            "site {site} writes memory"
        );
        let mlw = pypto_smt::mlw::emit_mlw(&body, "paged attention spmd");
        assert!(mlw.contains("goal waw_") || mlw.contains("goal raw_"));
        assert!(mlw.contains("SKIPPED — body carries state"), "Q3 refused");
    }
}

/// End to end through why3 on the flagship example: every pair goal of
/// site 0 must prove. Ignored by default — CI has no why3.
#[test]
#[ignore = "requires why3 (present in the dev environment, not in CI)"]
fn why3_proves_paged_attention_spmd() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/paged_attention_spmd.py"
    ))
    .unwrap();
    let body = walk_spmd(&parse_program(&src).unwrap(), 0).unwrap();
    let mlw = pypto_smt::mlw::emit_mlw(&body, "paged attention spmd");
    let dir = std::env::temp_dir().join("pypto-smt-why3-pa");
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join("pa.mlw");
    std::fs::write(&file, mlw).unwrap();
    let out = std::process::Command::new("why3")
        .arg("prove")
        .arg("-P")
        .arg("Z3,4.16.0")
        .arg("-t")
        .arg("120")
        .arg(&file)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    let valid = text.matches("Valid").count();
    assert!(valid >= 30, "expected every goal proved, got {valid}:\n{text}");
}

/// A valid program with a genuine intra-spmd WAW race: every block
/// stores the same window of `out`. The compiler accepts it (fixture is
/// real specialize output); the self-pair goal must exist.
#[test]
fn race_fixture_emits_self_pair_goal() {
    for name in ["spmd_race", "spmd_partitioned"] {
        let path = format!("{}/tests/fixtures/{}.py", env!("CARGO_MANIFEST_DIR"), name);
        let program = parse_program(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let body = walk_spmd(&program, 0).unwrap();
        assert_eq!(body.width, 4);
        let mlw = pypto_smt::mlw::emit_mlw(&body, name);
        assert!(
            mlw.contains("goal waw_out_n1"),
            "{name}: the store races with itself across blocks"
        );
    }
}

/// End to end: z3 finds the race (goal not provable) and proves the
/// stride-partitioned twin disjoint. Ignored without why3.
#[test]
#[ignore = "requires why3 (present in the dev environment, not in CI)"]
fn why3_separates_race_from_partitioned() {
    for (name, must_be_valid) in [("spmd_race", false), ("spmd_partitioned", true)] {
        let path = format!("{}/tests/fixtures/{}.py", env!("CARGO_MANIFEST_DIR"), name);
        let program = parse_program(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let body = walk_spmd(&program, 0).unwrap();
        let mlw = pypto_smt::mlw::emit_mlw(&body, name);
        let dir = std::env::temp_dir().join("pypto-smt-why3-race");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join(format!("{name}.mlw"));
        std::fs::write(&file, mlw).unwrap();
        let out = std::process::Command::new("why3")
            .arg("prove")
            .arg("-P")
            .arg("Z3,4.16.0")
            .arg("-t")
            .arg("60")
            .arg(&file)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        let mut verdict: Option<String> = None;
        let mut at_goal = false;
        for line in text.lines() {
            if line.contains(consts::GOAL_MARKER) {
                at_goal = true;
            } else if at_goal && line.contains(consts::VERDICT_MARKER) {
                verdict = Some(line.to_owned());
                break;
            }
        }
        let valid = verdict.as_deref().is_some_and(|v| v.contains("Valid"));
        assert_eq!(
            valid, must_be_valid,
            "{name}: waw verdict {:?} (expected valid={must_be_valid})",
            verdict.unwrap_or_else(|| "(no verdict)".to_owned())
        );
    }
}

/// A *conditional* race: overlap depends on tensor contents. Block w
/// stores at rows read from table[w]; the tool must refuse to prove it
/// (z3 exhibits colliding table values), and the same goal must become
/// Valid once the author's contract (distinct slots, distinct rows) is
/// stated as a lemma — the lemma itself stays an axiom (sat), because it
/// is a fact about runtime data, not a theorem.
#[test]
fn conditional_race_fixture_walks() {
    let path = format!("{}/tests/fixtures/spmd_cond_race.py", env!("CARGO_MANIFEST_DIR"));
    let program = parse_program(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let body = walk_spmd(&program, 0).unwrap();
    assert_eq!(body.width, 4);
    let mlw = pypto_smt::mlw::emit_mlw(&body, "conditional race");
    assert!(mlw.contains("function read_table"), "the metadata plane is read");
    assert!(mlw.contains("goal waw_out_n1"), "the conditional WAW pair");
}

#[test]
#[ignore = "requires why3 (present in the dev environment, not in CI)"]
fn why3_conditional_race_needs_the_data_axiom() {
    let path = format!("{}/tests/fixtures/spmd_cond_race.py", env!("CARGO_MANIFEST_DIR"));
    let program = parse_program(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let body = walk_spmd(&program, 0).unwrap();
    let mlw = pypto_smt::mlw::emit_mlw(&body, "conditional race");
    let augmented = mlw.replacen(
        "  predicate overlap",
        consts::TABLE_AXIOM_LEMMA,
        1,
    );
    assert!(augmented.len() > mlw.len(), "axiom injected");
    for (model, must_be_valid) in [(mlw.as_str(), false), (augmented.as_str(), true)] {
        let dir = std::env::temp_dir().join("pypto-smt-why3-cond");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("cond.mlw");
        std::fs::write(&file, model).unwrap();
        let out = std::process::Command::new("why3")
            .arg("prove")
            .arg("-P")
            .arg("Z3,4.16.0")
            .arg("-t")
            .arg("60")
            .arg(&file)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        let mut verdict: Option<String> = None;
        let mut at_goal = false;
        for line in text.lines() {
            if line.contains(consts::GOAL_MARKER) {
                at_goal = true;
            } else if at_goal && line.contains(consts::VERDICT_MARKER) {
                verdict = Some(line.to_owned());
                break;
            }
        }
        let valid = verdict.as_deref().is_some_and(|v| v.contains("Valid"));
        assert_eq!(
            valid, must_be_valid,
            "verdict {:?} (expected valid={must_be_valid})",
            verdict.unwrap_or_else(|| "(none)".to_owned())
        );
    }
}
