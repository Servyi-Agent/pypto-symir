# pypto.program: _jit_gate_test
import pypto.language as pl

@pl.program
class _jit_gate_test:
    @pl.function(type=pl.FunctionType.Inline)
    def route_topk_row(self, score_row: pl.Tensor[[1, 384], pl.FP32], idx_init: pl.Tensor[[1, 384], pl.UINT32]):
        # Sort one Pro router row and return its leading expert ids.
        sorted_32: pl.Tensor[[1, 768], pl.FP32] = pl.tensor.sort32(score_row, idx_init)
        sorted_64: pl.Tensor[[1, 768], pl.FP32] = pl.tensor.mrgsort_format1(sorted_32, pl.const(64, pl.INT32))
        sorted_pairs: pl.Tensor[[1, 768], pl.FP32] = pl.tensor.mrgsort_format2(pl.tensor.slice(sorted_64, [1, 256], [0, 0]), pl.tensor.slice(sorted_64, [1, 256], [0, 256]), pl.tensor.slice(sorted_64, [1, 256], [0, 512]), exhausted=False)
        return pl.tensor.gather_mask(pl.tensor.slice(sorted_pairs, [1, 16], [0, 0]), mask_pattern=2, output_dtype=pl.INT32)
    @pl.function(type=pl.FunctionType.Inline)
    def gate_normalized(self, x_normed: pl.Tensor[[80, 5120], pl.BF16], gate_w: pl.Tensor[[384, 5120], pl.FP32], gate_bias: pl.Tensor[[384], pl.FP32], num_tokens: pl.Scalar[pl.INT32], x_norm_mx: pl.Tensor[[80, 5120], pl.FP8E4M3FN], x_norm_scale: pl.Tensor[[1, 12800], pl.FP8E8M0], indices: pl.Tensor[[80, 6], pl.INT32], weights: pl.Tensor[[80, 6], pl.FP32]) -> pl.Tensor[[80, 6], pl.FP32]:
        active_tokens: pl.Scalar[pl.INDEX] = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        if active_tokens > 80:
            active_tokens: pl.Scalar[pl.INDEX] = pl.cast(80, pl.INDEX)
        task_tokens: pl.Scalar[pl.INDEX] = pl.max(active_tokens, pl.cast(1, pl.INDEX))
        active_mask: pl.Scalar[pl.FP32] = pl.cast(pl.min(pl.max(pl.cast(num_tokens, pl.INDEX), 0), 1), pl.FP32)
        task_gate_tiles: pl.Scalar[pl.INDEX] = (task_tokens + 16 - 1) // 16
        xg_buf: pl.Tensor[[80, 5120], pl.FP32] = pl.tensor.create([80, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for init_idx in pl.spmd(50, name_hint="gate_normalized_zero_spmd"):
            init_t: pl.Scalar[pl.INDEX] = init_idx // 10 * 16
            init_k: pl.Scalar[pl.INDEX] = init_idx % 10 * 512
            xg_buf: pl.Tensor[[80, 5120], pl.FP32] = pl.tensor.assemble(xg_buf, pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0), [init_t, init_k])
        for tok in pl.spmd(task_tokens, name_hint="gate_normalized_input_spmd"):
            normalized: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.load(x_normed, [tok, 0], [1, 5120], [1, 5120]), target_type=pl.FP32, mode='round')
            normalized_v1: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec] = pl.tile.muls(normalized, active_mask)
            pl.tile.store(normalized_v1, [tok, 0], xg_buf, [1, 5120])
        for quant_idx in pl.spmd(25, name_hint="x_norm_mx_quant_spmd"):
            tile_idx: pl.Scalar[pl.INDEX] = quant_idx // 5
            task_chunk_idx: pl.Scalar[pl.INDEX] = quant_idx % 5
            t0: pl.Scalar[pl.INDEX] = tile_idx * 16
            for task_chunk in pl.range(4):
                chunk_idx: pl.Scalar[pl.INDEX] = task_chunk_idx * 4 + task_chunk
                k0: pl.Scalar[pl.INDEX] = chunk_idx * 256
                x_norm_chunk: pl.Tile[[16, 256], pl.FP32] = pl.tile.load(xg_buf, [t0, k0], [16, 256], [16, 256])
                quant_groups: pl.Tile[[128, 32], pl.FP32] = pl.tile.reshape(x_norm_chunk, [128, 32])
                quant_tmp: pl.Tile[[128, 32], pl.FP32] = pl.tile.create([128, 32], dtype=pl.FP32)
                quant_maximum: pl.Tile[[128, 1], pl.FP32, pl.Mem.Vec] = pl.tile.maximums(pl.tile.row_max(pl.tile.abs(quant_groups), quant_tmp), 0.0001)
                quant_bits: pl.Tile[[128, 1], pl.INT32, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.muls(quant_maximum, 0.002232142857142857), dtype=pl.INT32)
                quant_exponent: pl.Tile[[128, 1], pl.INT32, pl.Mem.Vec] = pl.tile.shrs(pl.tile.adds(quant_bits, pl.const(8388607, pl.INT32)), pl.const(23, pl.INT32))
                quant_scale: pl.Tile[[128, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.shls(quant_exponent, pl.const(23, pl.INT32)), dtype=pl.FP32)
                quant_normalized: pl.Tile[[128, 32], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(quant_groups, quant_scale)
                quant_clipped: pl.Tile[[128, 32], pl.FP32, pl.Mem.Vec] = pl.tile.minimums(pl.tile.maximums(quant_normalized, -448.0), 448.0)
                quant_payload: pl.Tile[[128, 32], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.cast(quant_clipped, target_type=pl.FP8E4M3FN, mode='rint')
                x_quant: pl.Tile[[16, 256], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.reshape(quant_payload, [16, 256])
                quant_signed: pl.Tile[[128, 1], pl.INT32, pl.Mem.Vec] = pl.tile.sub(quant_exponent, pl.tile.muls(pl.tile.shrs(quant_exponent, pl.const(7, pl.INT32)), pl.const(256, pl.INT32)))
                quant_codes: pl.Tile[[128, 1], pl.UINT8, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.cast(quant_signed, target_type=pl.INT8, mode='round'), dtype=pl.UINT8)
                quant_flat: pl.Tile[[1, 128], pl.UINT8, pl.Mem.Vec] = pl.tile.reshape(quant_codes, [1, 128])
                quant_pack_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                quant_packed: pl.Tile[[16, 8], pl.UINT8, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.tmov_x2zz(quant_flat, quant_pack_tmp, group_axis=1, dst_rows=16, dst_cols=8)
                scale_quant: pl.Tile[[16, 8], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.reinterpret_view(quant_packed, dtype=pl.FP8E8M0)
                x_norm_mx: pl.Tensor[[80, 5120], pl.FP8E4M3FN] = pl.tile.store(x_quant, [t0, k0], x_norm_mx)
                scale_offset: pl.Scalar[pl.INDEX] = t0 * 160 + chunk_idx * 16 * 8
                x_norm_scale: pl.Tensor[[1, 12800], pl.FP8E8M0] = pl.tile.store(pl.tile.reshape(scale_quant, [1, 128]), [0, scale_offset], x_norm_scale)
        biased_scores_buf: pl.Tensor[[80, 384], pl.FP32] = pl.tensor.create([80, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
            for zt in pl.range(80):
                if zt >= active_tokens:
                    for zk in pl.range(6):
                        zero_index: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                        pl.tensor.write(indices, [zt, zk], zero_index)
                        zero_weight: pl.Scalar[pl.FP32] = pl.cast(0.0, pl.FP32)
                        pl.tensor.write(weights, [zt, zk], zero_weight)
            if pl.const(384, pl.INDEX) < pl.const(384, pl.INDEX):
                biased_pad: pl.Tensor[[80, 0], pl.FP32] = pl.tensor.full([80, 0], dtype=pl.FP32, value=-3.4028234663852886e+38)
                biased_scores_buf: pl.Tensor[[80, 384], pl.FP32] = pl.tensor.assemble(biased_scores_buf, biased_pad, [0, 384])
        route_scores_buf: pl.Tensor[[80, 384], pl.FP32] = pl.tensor.create([80, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for gb_idx in pl.spmd(task_gate_tiles * 24, name_hint="gate_spmd"):
            tg: pl.Scalar[pl.INDEX] = gb_idx // 24
            nb: pl.Scalar[pl.INDEX] = gb_idx % 24
            t1: pl.Scalar[pl.INDEX] = tg * 16
            n0: pl.Scalar[pl.INDEX] = nb * 16
            gate_logits_tile: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for kb in pl.pipeline(10, stage=2):
                gd_kd: pl.Scalar[pl.INDEX] = kb * 512
                gd_x: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(xg_buf, [16, 512], [t1, gd_kd])
                gd_w: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(gate_w, [16, 512], [n0, gd_kd])
                if gd_kd == 0:
                    gate_logits_tile: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.matmul(gd_x, gd_w, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                else:
                    gate_logits_tile: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.matmul_acc(gate_logits_tile, gd_x, gd_w, a_trans=False, b_trans=True)
            gp_relu: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.maximum(gate_logits_tile, 0.0)
            gp_abs: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.abs(gate_logits_tile)
            gp_neg_abs: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.neg(gp_abs)
            gp_exp_abs: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.exp(gp_neg_abs)
            gp_exp_plus: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.adds(gp_exp_abs, 1.0)
            gp_softplus_tail: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.log(gp_exp_plus)
            gp_softplus_log: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.add(gp_relu, gp_softplus_tail)
            gp_neg_logits: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.neg(gate_logits_tile)
            gp_neg_shift: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.subs(gp_neg_logits, 10.0)
            gp_neg_mask_floor: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.maximum(gp_neg_shift, 0.0)
            gp_neg_floor_mask: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.minimum(gp_neg_mask_floor, 1.0)
            gp_logits_floor: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.minimum(gate_logits_tile, 0.0)
            gp_neg_exp: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.exp(gp_logits_floor)
            gp_neg_floor: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.mul(gp_neg_floor_mask, gp_neg_exp)
            gp_softplus: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.maximum(gp_softplus_log, gp_neg_floor)
            gp_score: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.sqrt(gp_softplus)
            route_scores_buf: pl.Tensor[[80, 384], pl.FP32] = pl.tensor.assemble(route_scores_buf, gp_score, [t1, n0])
            gp_bias_row: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(gate_bias, [16], [n0]), [1, 16])
            gp_biased: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.col_expand_add(gp_score, gp_bias_row)
            biased_scores_buf: pl.Tensor[[80, 384], pl.FP32] = pl.tensor.assemble(biased_scores_buf, gp_biased, [t1, n0])
        task_route_tiles: pl.Scalar[pl.INDEX] = (task_tokens + 8 - 1) // 8
        for ts_idx in pl.spmd(task_route_tiles, name_hint="route_sort_spmd"):
            t1: pl.Scalar[pl.INDEX] = ts_idx * 8
            topk_idx_tile: pl.Tensor[[8, 8], pl.INT32] = pl.tensor.create([8, 8], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            sr_idx_init: pl.Tensor[[1, 384], pl.UINT32] = pl.tensor.create([1, 384], dtype=pl.UINT32, layout=pl.TensorLayout.ND)
            sr_index_range: pl.Tensor[[1, 384], pl.UINT32] = pl.tensor.ci(pl.const(0, pl.UINT32), [1, 384], dtype=pl.UINT32, descending=False)
            sr_idx_init: pl.Tensor[[1, 384], pl.UINT32] = pl.tensor.assemble(sr_idx_init, sr_index_range, [0, 0])
            for sr_tt in pl.range(8):
                sr_t: pl.Scalar[pl.INDEX] = t1 + sr_tt
                sr_row: pl.Tensor[[1, 384], pl.FP32] = pl.tensor.slice(biased_scores_buf, [1, 384], [sr_t, 0])
                sr_i: pl.Tensor[[1, 8], pl.INT32] = self.route_topk_row(sr_row, sr_idx_init)
                topk_idx_tile: pl.Tensor[[8, 8], pl.INT32] = pl.tensor.assemble(topk_idx_tile, sr_i, [sr_tt, 0])
            local_scores: pl.Tensor[[8, 384], pl.FP32] = pl.tensor.create([8, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            local_scores: pl.Tensor[[8, 384], pl.FP32] = pl.tensor.assemble(local_scores, pl.tensor.slice(route_scores_buf, [8, 384], [t1, 0]), [0, 0])
            gather_all: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.gather(local_scores, topk_idx_tile, dim=-1)
            gather_valid: pl.Tensor[[8, 8], pl.FP32, pl.TensorView(valid_shape=[8, 6], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(gather_all, 8, 6)
            topk_vals_pad: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            topk_idx_read: pl.Tensor[[8, 8], pl.INT32] = pl.tensor.create([8, 8], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            topk_idx_read: pl.Tensor[[8, 8], pl.INT32] = pl.tensor.assemble(topk_idx_read, pl.tensor.slice(topk_idx_tile, [8, 8], [0, 0]), [0, 0])
            topk_sum: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(topk_vals_pad)
            denom: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(topk_sum, [8, 1])
            topk_normalized: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.row_expand_div(topk_vals_pad, denom)
            normalized_weights: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(topk_normalized, 1.5)
            for wt_tt in pl.range(8):
                wt_out_t: pl.Scalar[pl.INDEX] = t1 + wt_tt
                if wt_out_t < active_tokens:
                    for wt_k in pl.range(6):
                        wt_out_idx: pl.Scalar[pl.INT32] = pl.tensor.read(topk_idx_read, [wt_tt, wt_k])
                        pl.tensor.write(indices, [wt_out_t, wt_k], wt_out_idx)
                        wt_out_weight: pl.Scalar[pl.FP32] = pl.tensor.read(normalized_weights, [wt_tt, wt_k])
                        pl.tensor.write(weights, [wt_out_t, wt_k], wt_out_weight)
        return weights
    @pl.function(type=pl.FunctionType.Inline)
    def rms_norm_tiled(self, x: pl.Tensor[[80, 5120], pl.BF16], norm_w: pl.Tensor[[5120], pl.BF16], x_normed: pl.Tensor[[80, 5120], pl.BF16], block: pl.Scalar[pl.INT32], num_tokens: pl.Scalar[pl.INT32]) -> pl.Tensor[[80, 5120], pl.BF16]:
        # Normalize one eight-row block using 512-column accumulation.
        t0: pl.Scalar[pl.INDEX] = pl.cast(block, pl.INDEX) * 8
        valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, pl.cast(num_tokens, pl.INDEX) - t0)
        sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(10, stage=2):
            k0: pl.Scalar[pl.INDEX] = kb * 512
            source: pl.Tensor[[8, 512], pl.BF16, pl.TensorView(valid_shape=[valid_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x, [8, 512], [t0, k0], [valid_rows, 512])
            source_v1: pl.Tensor[[8, 512], pl.BF16] = pl.tensor.set_validshape(pl.tensor.fillpad(source, pad_value=pl.PadValue.zero), 8, 512)
            value: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(source_v1, target_type=pl.FP32, mode='round')
            sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.reshape(pl.tensor.row_sum(pl.tensor.mul(value, value)), [1, 8]))
        inv_rms: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum, 0.00019531250000000001), 9.9999999999999995e-07), high_precision=True), [8, 1])
        for kb_1 in pl.pipeline(10, stage=2):
            k0: pl.Scalar[pl.INDEX] = kb_1 * 512
            source: pl.Tensor[[8, 512], pl.BF16, pl.TensorView(valid_shape=[valid_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x, [8, 512], [t0, k0], [valid_rows, 512])
            source_v2: pl.Tensor[[8, 512], pl.BF16] = pl.tensor.set_validshape(pl.tensor.fillpad(source, pad_value=pl.PadValue.zero), 8, 512)
            value: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(source_v2, target_type=pl.FP32, mode='round')
            gamma: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(norm_w, [512], [k0]), target_type=pl.FP32, mode='round'), [1, 512])
            normalized: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(value, inv_rms), gamma)
            x_normed: pl.Tensor[[80, 5120], pl.BF16] = pl.tensor.assemble(x_normed, pl.tensor.set_validshape(pl.tensor.cast(normalized, target_type=pl.BF16, mode='rint'), valid_rows, 512), [t0, k0])
        return x_normed
    @pl.function(type=pl.FunctionType.Inline)
    def gate(self, x_mixed: pl.Tensor[[80, 5120], pl.BF16], norm_w: pl.Tensor[[5120], pl.BF16], gate_w: pl.Tensor[[384, 5120], pl.FP32], gate_bias: pl.Tensor[[384], pl.FP32], num_tokens: pl.Scalar[pl.INT32], x_norm_mx: pl.Tensor[[80, 5120], pl.FP8E4M3FN], x_norm_scale: pl.Tensor[[1, 12800], pl.FP8E8M0], indices: pl.Tensor[[80, 6], pl.INT32], weights: pl.Tensor[[80, 6], pl.FP32]):
        # Keep the standalone gate ABI while making FFN RMSNorm explicit.
        x_normed: pl.Tensor[[80, 5120], pl.BF16] = pl.tensor.create([80, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        norm_tokens: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_mixed, 0)
        for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm_spmd"):
            self.rms_norm_tiled(x_mixed, norm_w, x_normed, norm_block, norm_tokens)
        return self.gate_normalized(x_normed, gate_w, gate_bias, num_tokens, x_norm_mx, x_norm_scale, indices, weights)
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def gate_test(self, x_mixed: pl.Tensor[[80, 5120], pl.BF16], norm_w: pl.Tensor[[5120], pl.BF16], gate_w: pl.Tensor[[384, 5120], pl.FP32], gate_bias: pl.Tensor[[384], pl.FP32], layer_id: pl.Scalar[pl.INT32], num_tokens: pl.Scalar[pl.INT32], tid2eid: pl.Tensor[[129280, 6], pl.INT32], input_ids: pl.Tensor[[80], pl.INT64], x_norm_mx: pl.Out[pl.Tensor[[80, 5120], pl.FP8E4M3FN]], x_norm_scale: pl.Out[pl.Tensor[[1, 12800], pl.FP8E8M0]], indices: pl.Out[pl.Tensor[[80, 6], pl.INT32]], weights: pl.Out[pl.Tensor[[80, 6], pl.FP32]]) -> tuple[pl.Tensor[[80, 5120], pl.FP8E4M3FN], pl.Tensor[[1, 12800], pl.FP8E8M0], pl.Tensor[[80, 6], pl.INT32], pl.Tensor[[80, 6], pl.FP32]]:
        self.gate(x_mixed, norm_w, gate_w, gate_bias, num_tokens, x_norm_mx, x_norm_scale, indices, weights)
        return x_norm_mx, x_norm_scale, indices, weights
