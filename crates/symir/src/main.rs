//! CLI: parse a printed PyPTO IR file and emit the symbolic task DAG.

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use symir::consts;
use symir::emit;

/// Which emitter to use.
enum Format {
    Dot,
    Json,
    Report,
    Svg,
    Png,
}

/// Exit status for usage errors (kept out of `main` for testability).
const EXIT_USAGE: u8 = 2;

fn main() -> ExitCode {
    let mut input: Option<String> = None;
    let mut format = Format::Report;
    let mut entry: Option<String> = None;
    let mut output: Option<String> = None;
    let args: Vec<String> = env::args().skip(consts::SKIP_PROGRAM_NAME).collect();
    let mut idx = usize::default();
    while let Some(arg) = args.get(idx) {
        if arg == consts::FLAG_EMIT {
            idx += consts::LINE_ADVANCE;
            match args.get(idx).map(|s| s.as_str()) {
                Some(v) if v == consts::FORMAT_DOT => format = Format::Dot,
                Some(v) if v == consts::FORMAT_JSON => format = Format::Json,
                Some(v) if v == consts::FORMAT_REPORT => format = Format::Report,
                Some(v) if v == consts::FORMAT_SVG => format = Format::Svg,
                Some(v) if v == consts::FORMAT_PNG => format = Format::Png,
                _ => {
                    eprintln!("{}", consts::MSG_USAGE);
                    return ExitCode::from(EXIT_USAGE);
                }
            }
        } else if arg == consts::FLAG_ENTRY {
            idx += consts::LINE_ADVANCE;
            match args.get(idx) {
                Some(v) => entry = Some(v.clone()),
                None => {
                    eprintln!("{}", consts::MSG_USAGE);
                    return ExitCode::from(EXIT_USAGE);
                }
            }
        } else if arg == consts::FLAG_OUTPUT {
            idx += consts::LINE_ADVANCE;
            match args.get(idx) {
                Some(v) => output = Some(v.clone()),
                None => {
                    eprintln!("{}", consts::MSG_USAGE);
                    return ExitCode::from(EXIT_USAGE);
                }
            }
        } else {
            input = Some(arg.clone());
        }
        idx += consts::LINE_ADVANCE;
    }
    let Some(input) = input else {
        eprintln!("{}", consts::MSG_USAGE);
        return ExitCode::from(EXIT_USAGE);
    };
    let src = match fs::read_to_string(&input) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("{} {input}: {e}", consts::MSG_READ_FAILURE);
            return ExitCode::FAILURE;
        }
    };
    let dag = match symir::extract(&src, entry.as_deref()) {
        Ok(dag) => dag,
        Err(e) => {
            eprintln!("{} {input}: {e}", consts::MSG_PIPELINE_FAILURE);
            return ExitCode::FAILURE;
        }
    };
    match format {
        Format::Svg | Format::Png => {
            let Some(path) = output else {
                eprintln!("{}", consts::MSG_USAGE);
                return ExitCode::from(EXIT_USAGE);
            };
            let task_graph = symir::graph::build(&dag);
            let render_format = match format {
                Format::Svg => consts::FORMAT_SVG,
                _ => consts::FORMAT_PNG,
            };
            if let Err(e) = task_graph.render(&dag.program, render_format, Path::new(&path)) {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
        _ => {
            let rendered = match format {
                Format::Dot => symir::graph::build(&dag).to_dot(&dag.program),
                Format::Json => match serde_json::to_string_pretty(&dag) {
                    Ok(text) => text,
                    Err(e) => {
                        eprintln!("{} {e}", consts::MSG_JSON_FAILURE);
                        return ExitCode::FAILURE;
                    }
                },
                Format::Report => emit::to_report(&dag),
                Format::Svg | Format::Png => unreachable_render(),
            };
            match output {
                Some(path) => {
                    if let Err(e) = fs::write(&path, rendered) {
                        eprintln!("{} {path}: {e}", consts::MSG_WRITE_FAILURE);
                        return ExitCode::FAILURE;
                    }
                }
                None => print!("{rendered}"),
            }
        }
    }
    ExitCode::SUCCESS
}

/// The svg/png arms are handled before this point.
fn unreachable_render() -> String {
    String::new()
}
