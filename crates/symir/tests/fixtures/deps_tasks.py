import pypto.language as pl

@pl.program
class Pipeline:
    @pl.function(type=pl.FunctionType.InCore)
    def stage_a(self, x: pl.Tensor[[4, 8], pl.FP32], mid: pl.Out[pl.Tensor[[4, 8], pl.FP32]]):
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def stage_b(self, mid: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]):
        pass

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], mid: pl.Out[pl.Tensor[[4, 8], pl.FP32]], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        (mid_r, a_tid) = pl.submit(self.stage_a, x, mid)
        (out_r, b_tid) = pl.submit(self.stage_b, mid, out, deps=[a_tid])
        return out_r
