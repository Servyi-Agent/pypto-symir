# pypto.program: _jit_loop_parallel
import pypto.language as pl
import pypto

@pl.program
class _jit_loop_parallel:
    # ir_stage: Buffer
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def loop_parallel_incore_0(x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__iter_v1: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True, "buffer_ir_selected": True})
        mem_vec_2_buffer: pypto.ir.BufferType([4, 8], pypto.ir.DataType.FP32, pypto.ir.MemorySpace.Vec, valid_shape=[4, 8], blayout=pypto.ir.TileLayout.row_major, slayout=pypto.ir.TileLayout.none_box, fractal=512, pad=pypto.ir.PadValue.null, compact=pypto.ir.CompactMode.null) = pl.buffer.alloc([], 0)
        pl._dump.call("buffer.load", [x__ssa_v0, [0, 0], [4, 8], mem_vec_2_buffer], pypto.ir.VoidType())
        pl.buffer.muls(mem_vec_2_buffer, 2.0, mem_vec_2_buffer)
        pl._dump.call("buffer.store", [mem_vec_2_buffer, [0, 0], [4, 8], out__iter_v1], pypto.ir.VoidType())
        return out__iter_v1
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def loop_parallel(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        with pl.scope():
            for i__idx_v0, (out__iter_v1,) in pl.parallel(4, init_values=(out__ssa_v0,), attrs={"iter_arg_rebind_0": False}):
                with pl.scope():
                    out__ssa_v3: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 128)] = pl._dump.call(self.loop_parallel_incore_0, [x__ssa_v0, out__iter_v1], pl.Tensor[[4, 8], pl.FP32], attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                    out__rv_v2: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 128)] = pl.yield_(out__ssa_v3)
            return out__rv_v2