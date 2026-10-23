# pypto.program: _jit_spmd_race
import pypto.language as pl

@pl.program
class _jit_spmd_race:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def spmd_race(self, x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
        # Every block loads rows [0,2) and stores rows [0,2) — the four
        #         concurrent block instances write the same window: a WAW race.
        for w in pl.spmd(4, name_hint="spmd_race_spmd"):
            t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
            o: pl.Tensor[[8, 8], pl.FP32] = pl.tile.store(t, [0, 0], out, [2, 8])
        return o