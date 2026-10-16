# pypto.program: _jit_range_carry
import pypto.language as pl

@pl.program
class _jit_range_carry:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def range_carry_incore_0(a_1: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)], ret0__out: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[8, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True, "buffer_ir_selected": False})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        a_1__tile: pl.Tile[[4, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl._dump.call("tile.load", [a_1, [0, 0], [4, 8], [4, 8]], pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec})
        x__tile: pl.Tile[[4, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(128, pl.INT64), 128), pl.Mem.Vec] = pl._dump.call("tile.load", [x__ssa_v0, [0, 0], [4, 8], [4, 8]], pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec})
        t__tile: pl.Tile[[4, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl._dump.call("tile.add", [a_1__tile, x__tile], pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec])
        ret0__store: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[8, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 128)] = pl._dump.call("tile.store", [t__tile, [0, 0], ret0__out], pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[8, 1], layout=pl.TensorLayout.ND)])
        return ret0__out
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def range_carry(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        with pl.scope():
            acc__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 128)] = x__ssa_v0
            for i__idx_v0 in pl.range(4):
                with pl.scope():
                    ret0__out: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 128)] = pl._dump.call("tensor.slice", [acc__ssa_v0, [4, 8], [0, 0]], pl.Tensor[[4, 8], pl.FP32])
                    t__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 128)] = pl._dump.call(self.range_carry_incore_0, [acc__ssa_v0, x__ssa_v0, ret0__out], pl.Tensor[[4, 8], pl.FP32], attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout]})
            out__ssa_v1: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)] = pl._dump.call("tensor.assemble", [out__ssa_v0, acc__ssa_v0, [0, 0]], pl.Tensor[[4, 8], pl.FP32])
            return out__ssa_v1