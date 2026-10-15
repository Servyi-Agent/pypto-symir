import pypto.language as pl

@pl.program
class WhileProg:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        for (c,) in pl.while_(init_values=(x,)):
            pl.cond(pl.tensor.dim(c, 0) > 1)
            c = pl.yield_(pl.tensor.assemble(c, pl.tensor.add(c, x), [0, 0]))
        out_r = pl.tensor.assemble(out, c, [0, 0])
        return out_r
