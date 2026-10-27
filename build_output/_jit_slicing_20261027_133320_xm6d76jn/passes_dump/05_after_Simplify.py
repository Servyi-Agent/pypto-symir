# pypto.program: _jit_slicing
import pypto.language as pl

@pl.program
class _jit_slicing:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def slicing(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out__ssa_v1: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out__ssa_v0, pl.tensor.muls(pl.tensor.slice(x__ssa_v0, [2, 8], [0, 0]), 2.0), [0, 0])
        with pl.at(level=pl.Level.CORE_GROUP):
            out__ssa_v2: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.assemble(out__ssa_v1, pl.tensor.muls(pl.tensor.slice(x__ssa_v0, [2, 8], [2, 0]), 3.0), [2, 0])
        return out__ssa_v2
