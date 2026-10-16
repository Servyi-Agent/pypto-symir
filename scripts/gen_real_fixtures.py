"""Generate REAL fixtures: every fixture is a dump produced by the actual
PyPTO compiler (00_frontend.py of a compile with dump_passes=True).
No handwritten IR. Run from a pypto checkout's venv:

    .venv/bin/python scripts/gen_real_fixtures.py /workspace/pypto-symir
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/pypto-symir")
FIX = REPO / "crates" / "symir" / "tests" / "fixtures"

import pypto.ir
import pypto.language as pl
from pypto.backend import BackendType
from pypto.runtime import RunConfig

def dump(name, program, **kw):
    compiled = pypto.ir.compile(
        program, skip_ptoas=True, dump_passes=True,
        backend_type=BackendType.Ascend950, **kw,
    )
    src = (Path(compiled.output_dir) / "passes_dump" / "00_frontend.py").read_text()
    (FIX / name).write_text(src)
    print(name, "<-", compiled.output_dir, f"({len(src.splitlines())} lines)")


# --- deps_tasks: explicit submit + deps=, real kernels with returns -------
@pl.program
class Pipeline:
    @pl.function(type=pl.FunctionType.InCore)
    def stage_a(self, x: pl.Tensor[[4, 8], pl.FP32], mid: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        mid[:] = pl.mul(x, 2.0)
        return mid

    @pl.function(type=pl.FunctionType.InCore)
    def stage_b(self, mid: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        out[:] = pl.mul(mid, 3.0)
        return out

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], mid: pl.Out[pl.Tensor[[4, 8], pl.FP32]], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        mid_r, a_tid = pl.submit(self.stage_a, x, mid)
        out_r, b_tid = pl.submit(self.stage_b, mid, out, deps=[a_tid])
        return out_r


dump("deps_tasks.py", Pipeline)

# --- parallel_carry: parallel loop threading an InOut carried buffer ------
@pl.program
class ParCarry:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], acc: pl.InOut[pl.Tensor[[4, 8], pl.FP32]], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        a = acc
        for i, (a_2,) in pl.parallel(4, init_values=(a,)):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.tensor.add(a_2, x)
            a = pl.yield_(pl.tensor.assemble(a_2, t, [0, 0]))
        out[:] = a
        return out


dump("parallel_carry.py", ParCarry)

# --- while_loop: natural while over a scalar condition --------------------
@pl.program
class WhileProg:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, n: pl.Scalar[pl.INDEX], x: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        i: pl.Scalar[pl.INDEX] = 0
        acc = x
        while i < n:
            with pl.at(level=pl.Level.CORE_GROUP):
                acc = pl.tensor.add(acc, x)
            i = i + 1
        out[:] = acc
        return out


dump("while_loop.py", WhileProg)


# --- range_carry: sequential loop with a carried accumulator (compilable) --
@pl.jit
def range_carry(
    x: pl.Tensor[[4, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
):
    acc = x
    for i, (a_1,) in pl.range(4, init_values=(acc,)):
        with pl.at(level=pl.Level.CORE_GROUP):
            t = pl.tensor.add(a_1, x)
        a = pl.yield_(pl.tensor.assemble(a_1, t, [0, 0]))
    out[:] = a
    return out


@pl.jit
def parallel_task(
    x: pl.Tensor[[4, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
):
    for i in pl.parallel(4):
        with pl.at(level=pl.Level.CORE_GROUP):
            out[:] = pl.mul(x, 2.0)
    return out


import torch  # noqa: E402

_x = torch.ones(4, 8, dtype=torch.float32)
_out = torch.zeros(4, 8, dtype=torch.float32)

for name, kernel in (("range_carry_pre.py", range_carry), ("range_carry_post.py", range_carry),
                     ("parallel_task_post.py", parallel_task)):
    stage = "specialize" if "pre" in name else "lower"
    if stage == "specialize":
        src = kernel.specialize(_x, _out).as_python()
    else:
        src = kernel.lower(_x, _out, config=RunConfig(arch=BackendType.Ascend950)).as_python()
    (FIX / name).write_text(src)
    # prove the underlying program compiles end to end before keeping it
    compiled = kernel.compile(_x, _out, config=RunConfig(arch=BackendType.Ascend950))
    print(name, f"<- {stage}() of a kernel that compiles ({compiled.output_dir})")
