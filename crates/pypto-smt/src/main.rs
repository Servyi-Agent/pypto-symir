//! CLI: scan a printed PyPTO IR file for spmd sites and check the
//! strided-partition lemma for each site's width.

use std::env;
use std::fs;
use std::process::ExitCode;

use pypto_smt::consts;
use pypto_smt::mlw;
use pypto_smt::walk;
use pypto_smt::{find_spmd, stride_partition_is_disjoint};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < consts::EXPECTED_ARG_COUNT {
        eprintln!("{}", consts::MSG_USAGE);
        return ExitCode::FAILURE;
    }
    let src = match fs::read_to_string(&args[consts::PATH_ARG_INDEX]) {
        Ok(src) => src,
        Err(e) => {
            eprintln!(
                "{} {}: {e}",
                consts::MSG_READ_FAILURE,
                args[consts::PATH_ARG_INDEX]
            );
            return ExitCode::FAILURE;
        }
    };
    let program = match pypto_ir::parse_program(&src) {
        Ok(program) => program,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(mode) = flag_value(&args, consts::FLAG_SPMD) {
        let site = flag_value(&args, consts::FLAG_SITE)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(consts::FIRST_SITE);
        return spmd_mode(&program, &mode, site);
    }
    for site in find_spmd(&program) {
        let disjoint = stride_partition_is_disjoint(site.width, site.width);
        println!(
            "{}: fn={} hint={:?} width={} body_len={} => {}",
            consts::MSG_SITE_LINE,
            site.function,
            site.hint,
            site.width,
            site.body_len,
            if disjoint {
                consts::MSG_LEMMA_HOLDS
            } else {
                consts::MSG_LEMMA_BROKEN
            }
        );
    }
    ExitCode::SUCCESS
}

/// `--flag value` extractor.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|pos| args.get(pos + consts::PATH_ARG_INDEX))
        .cloned()
}

/// `--spmd <mlw|check>`: walk a body and act on it.
fn spmd_mode(program: &pypto_ir::ast::Program, mode: &str, site: usize) -> ExitCode {
    if mode == consts::MODE_MLW || mode == consts::MODE_CHECK {
        let body = match walk::walk_spmd(program, site) {
            Ok(body) => body,
            Err(e) => {
                eprintln!("{}: {}: {}", consts::MSG_REFUSED, e.construct, e.detail);
                return ExitCode::FAILURE;
            }
        };
        if mode == consts::MODE_MLW {
            println!("{}", mlw::emit_mlw(&body, consts::MSG_PROVENANCE));
            return ExitCode::SUCCESS;
        }
        return check_site(&body, site);
    }
    eprintln!("{}", consts::MSG_USAGE);
    ExitCode::FAILURE
}

/// One command: emit the model and discharge it with why3/z3, printing
/// one verdict line per goal. Exit FAILURE when any goal is not proved.
fn check_site(body: &walk::SpmdBody, site: usize) -> ExitCode {
    let model = mlw::emit_mlw(body, consts::MSG_PROVENANCE);
    let dir = std::env::temp_dir().join(consts::CHECK_TEMP_DIR);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let file = dir.join(format!("site{site}.mlw"));
    if let Err(e) = std::fs::write(&file, model) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let output = match std::process::Command::new(consts::WHY3_BIN)
        .arg(consts::WHY3_ARG_PROVE)
        .arg(consts::WHY3_ARG_PROVER)
        .arg(consts::WHY3_PROVER)
        .arg(consts::WHY3_ARG_TIME)
        .arg(consts::WHY3_TIMELIMIT)
        .arg(&file)
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            eprintln!("{} ({e})", consts::MSG_NO_WHY3);
            return ExitCode::FAILURE;
        }
    };
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    let mut all_proved = true;
    let mut at_goal: Option<String> = None;
    for line in text.lines() {
        if line.contains(consts::VERDICT_GOAL) {
            if let Some(goal) = line.split_whitespace().last() {
                at_goal = Some(goal.to_owned());
            }
        } else if line.contains(consts::VERDICT_MARKER) {
            if let Some(goal) = at_goal.take() {
                let proved = line.contains(consts::VERDICT_VALID);
                if !proved {
                    all_proved = false;
                }
                println!("site {site}  {goal:<24} {}", line.trim());
            }
        }
    }
    if all_proved {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
