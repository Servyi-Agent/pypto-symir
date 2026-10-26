# pypto.program: _jit_spmd_cond_race
import pypto.language as pl

@pl.program
class _jit_spmd_cond_race:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def spmd_cond_race(self, table: pl.Tensor[[4], pl.INT32], x: pl.Tensor[[16, 8], pl.FP32], out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]) -> pl.Tensor[[16, 8], pl.FP32]:
        for w in pl.spmd(4, name_hint="spmd_cond_race_spmd"):
            r: pl.Scalar[pl.INT32] = pl.tensor.read(table, [w])
            row: pl.Scalar[pl.INDEX] = pl.cast(r, pl.INDEX) * 2
            t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
            o: pl.Tensor[[16, 8], pl.FP32] = pl.tile.store(t, [row, 0], out, [2, 8])
        return o