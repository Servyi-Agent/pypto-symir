# pypto.program: Probe0
import pypto.language as pl

@pl.program
class Probe0:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def k1(x: pl.Tensor[[4, 8], pl.FP32], m1: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        m1: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(m1, pl.tensor.muls(x, 2.0), [0, 0])
        return m1
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def k2(m1: pl.Tensor[[4, 8], pl.FP32], m2: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        m2: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(m2, pl.tensor.muls(m1, 3.0), [0, 0])
        return m2
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def entry(self, x: pl.Tensor[[4, 8], pl.FP32], m1: pl.Out[pl.Tensor[[4, 8], pl.FP32]], m2: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        with pl.scope():
            _submit_tmp: pl.Tuple[pl.Tensor[[4, 8], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.k1, x, m1)
            m1_r: pl.Tensor[[4, 8], pl.FP32] = _submit_tmp[0]
            t1: pl.Scalar[pl.TASK_ID] = _submit_tmp[1]
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                _submit_tmp_1: pl.Tuple[pl.Tensor[[4, 8], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.k2, m1, m2)
                m2_r: pl.Tensor[[4, 8], pl.FP32] = _submit_tmp_1[0]
                t2: pl.Scalar[pl.TASK_ID] = _submit_tmp_1[1]
            return m2_r
    @pl.function(type=pl.FunctionType.Inline)
    def manual_helper(self, m1: pl.Tensor[[4, 8], pl.FP32], m2: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            _submit_tmp: pl.Tuple[pl.Tensor[[4, 8], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.k2, m1, m2)
            m2_r: pl.Tensor[[4, 8], pl.FP32] = _submit_tmp[0]
            t2: pl.Scalar[pl.TASK_ID] = _submit_tmp[1]
        return m2_r
