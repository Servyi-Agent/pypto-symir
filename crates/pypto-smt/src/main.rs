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

/// `--spmd <index|mlw>`: walk a body and act on it.
fn spmd_mode(program: &pypto_ir::ast::Program, mode: &str, site: usize) -> ExitCode {
    if mode == consts::MODE_MLW {
        match walk::walk_spmd(program, site) {
            Ok(body) => {
                println!(
                    "{}",
                    mlw::emit_mlw(&body, consts::MSG_PROVENANCE)
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{}: {}: {}", consts::MSG_REFUSED, e.construct, e.detail);
                ExitCode::FAILURE
            }
        }
    } else {
        eprintln!("{}", consts::MSG_USAGE);
        ExitCode::FAILURE
    }
}
