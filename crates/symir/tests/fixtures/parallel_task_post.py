# pypto.program: _jit_parallel_task
import pypto.language as pl

@pl.program
class _jit_parallel_task:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def parallel_task_incore_0(x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__iter_v1: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True, "buffer_ir_selected": False})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        x__tile: pl.Tile[[4, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl._dump.call("tile.load", [x__ssa_v0, [0, 0], [4, 8], [4, 8]], pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec})
        t__tile: pl.Tile[[4, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl._dump.call("tile.muls", [x__tile, 2.0], pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec])
        out__tile: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)] = pl._dump.call("tile.store", [t__tile, [0, 0], out__iter_v1], pl.Tensor[[4, 8], pl.FP32])
        return out__iter_v1
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def parallel_task(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        with pl.scope():
            for i__idx_v0, (out__iter_v1,) in pl.parallel(4, init_values=(out__ssa_v0,), attrs={"iter_arg_rebind_0": False}):
                with pl.scope():
                    out__ssa_v3: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 128)] = pl._dump.call(self.parallel_task_incore_0, [x__ssa_v0, out__iter_v1], pl.Tensor[[4, 8], pl.FP32], attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                    out__rv_v2: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 128)] = pl.yield_(out__ssa_v3)
            return out__rv_v2