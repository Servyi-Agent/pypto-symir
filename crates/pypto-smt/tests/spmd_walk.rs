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
    assert_eq!(work.step, body.width);
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
    assert!(mlw.contains("goal covered_output"), "the Q3 goal for output");
    assert!(mlw.contains("function read_lengths"), "metadata plane declared");
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
        .arg("Z3,4.16.0,noBV")
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
