//! The compiler step: when the input is DSL source (a `@pl.jit` kernel),
//! run the real pypto compiler (`kernel.specialize(...).as_python()`)
//! through the configured interpreter and analyze its printed IR.
//!
//! Argument synthesis mirrors the pypto examples' own pattern: meta
//! tensors built from the kernel's annotations, dynamic dims replaced by
//! a small literal — the compiler still produces the IR; nothing here
//! interprets the DSL itself.

use std::env;
use std::process::Command;

use crate::consts;

/// The embedded driver: imports the user's module, picks the kernel
/// (by name, or the first module-level JIT kernel), synthesizes meta
/// arguments from annotations, and prints the specialized IR.
#[allow(inline_literal)] // the driver script is a single necessity literal
pub const DRIVER: &str = r#"
import importlib.util
import inspect
import os
import sys

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else None

import torch

from pypto.jit.decorator import JITFunction, _resolve_annotation, _torch_dtype_to_pypto
from pypto.jit.specializer import func_name_lookup

P_TO_TORCH = {}
for dt in (torch.float16, torch.float32, torch.bfloat16, torch.int8, torch.int16,
           torch.int32, torch.int64, torch.uint8, torch.bool):
    P_TO_TORCH[_torch_dtype_to_pypto(dt)] = dt

SYNTH_DIM = 4

def synth(annotation, lookup):
    from pypto.language import Tensor
    try:
        ann = _resolve_annotation(annotation, lookup)
    except Exception:
        return SYNTH_DIM
    if ann is Tensor or isinstance(ann, Tensor):
        raw = getattr(ann, "shape", None)
        shape = [d if isinstance(d, int) and d > 0 else SYNTH_DIM for d in raw] if raw else [SYNTH_DIM, SYNTH_DIM]
        dtype = P_TO_TORCH.get(getattr(ann, "dtype", None), torch.float32)
        return torch.empty(shape, dtype=dtype, device="meta")
    return SYNTH_DIM

import contextlib
spec = importlib.util.spec_from_file_location("pypto_smt_user_module", path)
module = importlib.util.module_from_spec(spec)
# The module may print at import time; keep stdout for the IR alone.
with contextlib.redirect_stdout(sys.stderr):
    spec.loader.exec_module(module)
kernels = {n: v for n, v in vars(module).items() if isinstance(v, JITFunction)}
if not kernels:
    sys.exit("no module-level @pl.jit kernel found")
name = want if want in kernels else sorted(kernels)[0]
kernel = kernels[name]
sig = inspect.signature(kernel._func)
lookup = func_name_lookup(kernel._func)
args = [synth(p.annotation, lookup) for p in sig.parameters.values()]
sys.stderr.write("kernel: %s\n" % name)
print(kernel.specialize(*args).as_python())
"#;

/// Runs the compiler step on a DSL source file. On success, returns the
/// printed IR. On failure, returns the compiler's stderr for reporting.
pub fn specialize_to_ir(path: &str, kernel: Option<&str>) -> Result<String, String> {
    let python = env::var(consts::ENV_PYTHON).unwrap_or_else(|_| consts::PYPTO_PYTHON.to_owned());
    let path_var = env::var(consts::ENV_PATH).unwrap_or_else(|_| consts::PYPTO_PATH.to_owned());
    let mut cmd = Command::new(&python);
    let _ = cmd
        .arg(consts::PY_ARG_CODE)
        .arg(DRIVER)
        .arg(path)
        .arg(kernel.unwrap_or(consts::EMPTY_ARG));
    let _ = cmd
        .env(consts::ENV_PYTHONPATH, path_var)
        .env(consts::ENV_SKIP_PIN_KEY, consts::ENV_SKIP_PIN_VAL);
    let output = cmd
        .output()
        .map_err(|e| format!("{} {python}: {e}", consts::MSG_NO_COMPILER))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}
