# pypto.program: ParCarry
import pypto.language as pl

@pl.program
class ParCarry:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], acc: pl.InOut[pl.Tensor[[4, 8], pl.FP32]], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        a: pl.Tensor[[4, 8], pl.FP32] = acc
        for i, (a_2,) in pl.parallel(4, init_values=(a,)):
            with pl.at(level=pl.Level.CORE_GROUP):
                t: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(a_2, x)
            a_1: pl.Tensor[[4, 8], pl.FP32] = pl.yield_(pl.tensor.assemble(a_2, t, [0, 0]))
        out: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out, a_1, [0, 0])
        return out
