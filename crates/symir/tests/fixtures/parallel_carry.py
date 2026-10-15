import pypto.language as pl

@pl.program
class ParCarry:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        for i, (a,) in pl.parallel(4, init_values=(x,)):
            t = pl.tensor.add(a, x)
            a = pl.yield_(pl.tensor.assemble(a, t, [0, 0]))
        out_r = pl.tensor.assemble(out, a, [0, 0])
        return out_r
