# pypto.program: _jit_spmd_if_hole
import pypto.language as pl

@pl.program
class _jit_spmd_if_hole:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def spmd_if_hole(self, x: pl.Tensor[[16, 8], pl.FP32], out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]) -> pl.Tensor[[16, 8], pl.FP32]:
        for w in pl.spmd(4, name_hint="if_hole_spmd"):
            t: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [4, 8], [4, 8], target_memory=pl.Mem.Vec)
            if w < 2:
                o: pl.Tensor[[16, 8], pl.FP32] = pl.tile.store(t, [w * 4, 0], out, [4, 8])
        return o