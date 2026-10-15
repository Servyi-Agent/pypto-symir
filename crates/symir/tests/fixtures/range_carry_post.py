# pypto.program: _jit_loop_range
import pypto.language as pl

@pl.program
class _jit_loop_range:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def loop_range(self, x__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 128)], out__ssa_v0: pl.Out[pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)]]) -> pl.Tensor[[4, 8], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        with pl.scope():
            acc__ssa_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 128)] = x__ssa_v0
            for i__idx_v0, (a_1,) in pl.range(4, init_values=(acc__ssa_v0,), attrs={"iter_arg_rebind_0": True}):
                with pl.scope():
                    t__tmp_v0: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 128)] = pl._dump.call("tensor.add", [a_1, x__ssa_v0], pl.Tensor[[4, 8], pl.FP32])
                    a: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 128)] = pl.yield_(t__tmp_v0)
            out__ssa_v1: pl.Tensor[[4, 8], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 128)] = pl._dump.call("tensor.assemble", [out__ssa_v0, a, [0, 0]], pl.Tensor[[4, 8], pl.FP32])
            return out__ssa_v1