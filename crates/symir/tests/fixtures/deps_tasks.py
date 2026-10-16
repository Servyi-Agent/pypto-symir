# pypto.program: Pipeline
import pypto.language as pl

@pl.program
class Pipeline:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def stage_a(x: pl.Tensor[[4, 8], pl.FP32], mid: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        mid: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(mid, pl.tensor.muls(x, 2.0), [0, 0])
        return mid
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def stage_b(mid: pl.Tensor[[4, 8], pl.FP32], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        out: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out, pl.tensor.muls(mid, 3.0), [0, 0])
        return out
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], mid: pl.Out[pl.Tensor[[4, 8], pl.FP32]], out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        _submit_tmp: pl.Tuple[pl.Tensor[[4, 8], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.stage_a, x, mid)
        mid_r: pl.Tensor[[4, 8], pl.FP32] = _submit_tmp[0]
        a_tid: pl.Scalar[pl.TASK_ID] = _submit_tmp[1]
        _submit_tmp_1: pl.Tuple[pl.Tensor[[4, 8], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.stage_b, mid, out, deps=[a_tid])
        out_r: pl.Tensor[[4, 8], pl.FP32] = _submit_tmp_1[0]
        b_tid: pl.Scalar[pl.TASK_ID] = _submit_tmp_1[1]
        return out_r
