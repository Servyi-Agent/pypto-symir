# pypto.program: _jit_loop_range
import pypto.language as pl

@pl.program
class _jit_loop_range:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def loop_range(self, x: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        acc: pl.Tensor[[4, 8], pl.FP32] = x
        for i, (a_1,) in pl.range(4, init_values=(acc,)):
            a: pl.Tensor[[4, 8], pl.FP32] = pl.yield_(pl.tensor.add(a_1, x))
        out: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out, a, [0, 0])
        return out