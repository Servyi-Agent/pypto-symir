"""DSL source fixture: the racy spmd kernel in source form (for the
compiler-step tests; every other fixture is printed IR)."""

import pypto.language as pl


@pl.jit
def spmd_race(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
    with pl.spmd(4, name_hint="spmd_race"):
        w: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
        o: pl.Tensor[[8, 8], pl.FP32] = pl.tile.store(t, [0, 0], out, [2, 8])
    return out
