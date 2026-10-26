# pypto.program: _jit_spmd_par_race
import pypto.language as pl

@pl.program
class _jit_spmd_par_race:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def spmd_par_race(self, x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
        for w in pl.spmd(4, name_hint="par_race_spmd"):
            for p in pl.parallel(2):
                t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
                o: pl.Tensor[[8, 8], pl.FP32] = pl.tile.store(t, [w * 2, 0], out, [2, 8])
        return o