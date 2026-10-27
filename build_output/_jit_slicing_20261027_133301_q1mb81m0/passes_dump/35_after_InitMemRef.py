# pypto.program: _jit_slicing
import pypto.language as pl

@pl.program
class _jit_slicing:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def slicing_incore_0(x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        t__tile: pl.Tile[[2, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl._dump.call("tile.load", [x__ssa_v0, [0, 0], [2, 8], [2, 8]], pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec})
        t__tile_1: pl.Tile[[2, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl._dump.call("tile.muls", [t__tile, 2.0], pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec])
        out__tile: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)] = pl._dump.call("tile.store", [t__tile_1, [0, 0], out__ssa_v0], pl.Tensor[[4, 8], pl.FP32])
        return out__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def slicing_incore_1(x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__ssa_v1: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        t__tile: pl.Tile[[2, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl._dump.call("tile.load", [x__ssa_v0, [2, 0], [2, 8], [2, 8]], pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec})
        t__tile_1: pl.Tile[[2, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl._dump.call("tile.muls", [t__tile, 3.0], pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec])
        out__tile: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)] = pl._dump.call("tile.store", [t__tile_1, [2, 0], out__ssa_v1], pl.Tensor[[4, 8], pl.FP32])
        return out__ssa_v1
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def slicing(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        out__ssa_v1: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 128)] = pl._dump.call(self.slicing_incore_0, [x__ssa_v0, out__ssa_v0], pl.Tensor[[4, 8], pl.FP32])
        out__ssa_v2: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 128)] = pl._dump.call(self.slicing_incore_1, [x__ssa_v0, out__ssa_v1], pl.Tensor[[4, 8], pl.FP32])
        return out__ssa_v2
