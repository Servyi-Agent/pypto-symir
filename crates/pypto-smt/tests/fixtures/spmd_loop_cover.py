# pypto.program: _jit_spmd_loop_cover
import pypto.language as pl

@pl.program
class _jit_spmd_loop_cover:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def spmd_loop_cover(self, x: pl.Tensor[[16, 8], pl.FP32], out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]) -> pl.Tensor[[16, 8], pl.FP32]:
        for w in pl.spmd(4, name_hint="loop_cover_spmd"):
            for i in pl.range(2):
                t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
                o: pl.Tensor[[16, 8], pl.FP32] = pl.tile.store(t, [w * 4 + i * 2, 0], out, [2, 8])
        return o