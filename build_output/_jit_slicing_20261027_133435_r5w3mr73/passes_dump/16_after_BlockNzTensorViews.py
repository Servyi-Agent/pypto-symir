# pypto.program: _jit_slicing
import pypto.language as pl

@pl.program
class _jit_slicing:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def slicing_incore_0(x__ssa_v0: pl.Tensor[[4, 8], pl.FP32], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        t__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x__ssa_v0, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
        t__tile_1: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile, 2.0)
        out__tile: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(t__tile_1, [0, 0], out__ssa_v0)
        return out__ssa_v0
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def slicing_incore_1(x__ssa_v0: pl.Tensor[[4, 8], pl.FP32], out__ssa_v1: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        t__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x__ssa_v0, [2, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
        t__tile_1: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile, 3.0)
        out__tile: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(t__tile_1, [2, 0], out__ssa_v1)
        return out__ssa_v1
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def slicing(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        out__ssa_v1: pl.Tensor[[4, 8], pl.FP32] = self.slicing_incore_0(x__ssa_v0, out__ssa_v0)
        out__ssa_v2: pl.Tensor[[4, 8], pl.FP32] = self.slicing_incore_1(x__ssa_v0, out__ssa_v1)
        return out__ssa_v2
