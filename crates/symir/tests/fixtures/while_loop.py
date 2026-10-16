# pypto.program: WhileProg
import pypto.language as pl

@pl.program
class WhileProg:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, n: pl.Scalar[pl.INDEX], x: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        i: pl.Scalar[pl.INDEX] = 0
        acc: pl.Tensor[[4, 8], pl.FP32] = x
        while i < n:
            with pl.at(level=pl.Level.CORE_GROUP):
                acc: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(acc, x)
            i: pl.Scalar[pl.INDEX] = i + 1
        out: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out, acc, [0, 0])
        return out
