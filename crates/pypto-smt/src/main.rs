//! CLI: scan a printed PyPTO IR file for spmd sites and check the
//! strided-partition lemma for each site's width.

use std::env;
use std::fs;
use std::process::ExitCode;

use pypto_smt::consts;
use pypto_smt::{find_spmd, stride_partition_is_disjoint};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != consts::EXPECTED_ARG_COUNT {
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
