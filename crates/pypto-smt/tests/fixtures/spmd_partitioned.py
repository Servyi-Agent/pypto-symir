# pypto.program: _jit_spmd_partitioned
import pypto.language as pl

@pl.program
class _jit_spmd_partitioned:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def spmd_partitioned(self, x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
        # The same kernel with the stride-partitioned store: block w writes
        #         rows [2w, 2w+2) — disjoint by construction.
        for w in pl.spmd(4, name_hint="spmd_partitioned_spmd"):
            t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
            o: pl.Tensor[[8, 8], pl.FP32] = pl.tile.store(t, [w * 2, 0], out, [2, 8])
        return o