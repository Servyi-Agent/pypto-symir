# pypto.program: _jit_range_carry
import pypto.language as pl

@pl.program
class _jit_range_carry:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def range_carry(self, x: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        acc: pl.Tensor[[4, 8], pl.FP32] = x
        for i, (a_1,) in pl.range(4, init_values=(acc,)):
            with pl.at(level=pl.Level.CORE_GROUP):
                t: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(a_1, x)
            a: pl.Tensor[[4, 8], pl.FP32] = pl.yield_(pl.tensor.assemble(a_1, t, [0, 0]))
        out: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out, a, [0, 0])
        return out