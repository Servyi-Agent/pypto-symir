# pypto.program: _jit_slicing
import pypto.language as pl

@pl.program
class _jit_slicing:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def slicing(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            t__tmp_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.slice(x__ssa_v0, [2, 8], [0, 0])
            t__tmp_v1: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(t__tmp_v0, 2.0)
            out__ssa_v1: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out__ssa_v0, t__tmp_v1, [0, 0])
        with pl.at(level=pl.Level.CORE_GROUP):
            t__tmp_v2: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.slice(x__ssa_v0, [2, 8], [2, 0])
            t__tmp_v3: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(t__tmp_v2, 3.0)
            out__ssa_v2: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out__ssa_v1, t__tmp_v3, [2, 0])
        return out__ssa_v2
