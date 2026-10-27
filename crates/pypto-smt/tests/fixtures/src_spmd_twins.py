"""A two-kernel spmd fixture in DSL source form.

- partitioned: block w writes rows [2w, 2w+2) — disjoint by
  construction. All goals prove; exit 0.
- racy: every block writes rows [0,2) — a cross-tile WAW race plus an
  unwritten-tensor coverage hole. Both show as sat with witnesses.

Run:
    pypto-smt spmd_fixture.py --kernel partitioned --spmd check
    pypto-smt spmd_fixture.py --kernel racy --spmd check
"""

import pypto.language as pl


@pl.jit
def partitioned(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
    with pl.spmd(4, name_hint="partitioned"):
        w: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
        o: pl.Tensor[[8, 8], pl.FP32] = pl.tile.store(t, [w * 2, 0], out, [2, 8])
    return out


@pl.jit
def racy(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
    with pl.spmd(4, name_hint="racy"):
        w: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Vec)
        o: pl.Tensor[[8, 8], pl.FP32] = pl.tile.store(t, [0, 0], out, [2, 8])
    return o
