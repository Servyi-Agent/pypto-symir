# pypto.program: SpmdPagedAttentionProgram
import pypto.language as pl

@pl.program
class SpmdPagedAttentionProgram:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def SpmdNormalize(oi_batch: pl.Tensor[[64, 128], pl.FP32], li_batch: pl.Tensor[[64, 1], pl.FP32], out_tensor: pl.Out[pl.Tensor[[64, 128], pl.FP32]], batch_count: pl.Scalar[pl.INDEX], q_offset: pl.Scalar[pl.INDEX], num_heads_param: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64, 128], pl.FP32]:
        # Normalize accumulators and scatter to output (VECTOR).
        spmd_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        spmd_block_num: pl.Scalar[pl.INDEX] = pl.tile.get_block_num()
        for b in pl.range(spmd_idx, batch_count, spmd_block_num):
            oi_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(oi_batch, [b * 16, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
            li_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(li_batch, [b * 16, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
            dst_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(oi_tile, li_tile)
            dst_row: pl.Scalar[pl.INDEX] = b * num_heads_param + q_offset
            out_tensor: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(dst_tile, [dst_row, 0], out_tensor, [16, 128])
        return out_tensor
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def SpmdOnlineUpdate(mij_batch: pl.Tensor[[64, 1], pl.FP32], lij_batch: pl.Tensor[[64, 1], pl.FP32], oi_new_batch: pl.Tensor[[64, 128], pl.FP32], mi_batch: pl.InOut[pl.Tensor[[64, 1], pl.FP32]], li_batch: pl.InOut[pl.Tensor[[64, 1], pl.FP32]], oi_batch: pl.InOut[pl.Tensor[[64, 128], pl.FP32]], is_first: pl.Scalar[pl.INDEX], batch_count: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[64, 1], pl.FP32], pl.Tensor[[64, 1], pl.FP32], pl.Tensor[[64, 128], pl.FP32]]:
        # noqa: PLR0913
        # Online softmax accumulator update (VECTOR).
        spmd_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        spmd_block_num: pl.Scalar[pl.INDEX] = pl.tile.get_block_num()
        if is_first != 0:
            for b in pl.range(spmd_idx, batch_count, spmd_block_num):
                mij_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(mij_batch, [b * 16, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                lij_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(lij_batch, [b * 16, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                oi_new_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(oi_new_batch, [b * 16, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
                mi_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(mij_tile, [b * 16, 0], mi_batch, [16, 1])
                li_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(lij_tile, [b * 16, 0], li_batch, [16, 1])
                oi_batch: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(oi_new_tile, [b * 16, 0], oi_batch, [16, 128])
        else:
            for b_1 in pl.range(spmd_idx, batch_count, spmd_block_num):
                mij_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(mij_batch, [b_1 * 16, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                lij_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(lij_batch, [b_1 * 16, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                oi_new_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(oi_new_batch, [b_1 * 16, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
                mi_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(mi_batch, [b_1 * 16, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                li_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(li_batch, [b_1 * 16, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                oi_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(oi_batch, [b_1 * 16, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
                mi_tile_nd: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(mi_tile, [1, 16])
                mij_tile_nd: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(mij_tile, [1, 16])
                li_tile_nd: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(li_tile, [1, 16])
                lij_tile_nd: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(lij_tile, [1, 16])
                mi_new: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(mi_tile_nd, mij_tile_nd)
                alpha: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(mi_tile_nd, mi_new))
                beta: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(pl.tile.sub(mij_tile_nd, mi_new))
                li_updated: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(pl.tile.mul(alpha, li_tile_nd), pl.tile.mul(beta, lij_tile_nd))
                mi_new_dn: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(mi_new, [16, 1])
                li_updated_dn: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(li_updated, [16, 1])
                mi_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(mi_new_dn, [b_1 * 16, 0], mi_batch, [16, 1])
                li_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(li_updated_dn, [b_1 * 16, 0], li_batch, [16, 1])
                alpha_dn: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(alpha, [16, 1])
                beta_dn: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(beta, [16, 1])
                oi_scaled: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(oi_tile, alpha_dn)
                oi_new_scaled: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(oi_new_tile, beta_dn)
                oi_updated: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(oi_scaled, oi_new_scaled)
                oi_batch: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(oi_updated, [b_1 * 16, 0], oi_batch, [16, 128])
        return mi_batch, li_batch, oi_batch
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def SpmdPvMatmul(pij_batch: pl.Tensor[[64, 128], pl.BF16], value_cache: pl.Tensor[[131072, 128], pl.BF16], oi_new_batch: pl.Out[pl.Tensor[[64, 128], pl.FP32]], block_table: pl.Tensor[[1024], pl.INT32], context_lens: pl.Tensor[[4], pl.INT32], batch_count: pl.Scalar[pl.INDEX], block_idx_kv: pl.Scalar[pl.INDEX], block_num: pl.Scalar[pl.INDEX], block_size_param: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64, 128], pl.FP32]:
        # PV matmul with SPMD stride partitioning over batch (CUBE).
        spmd_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        spmd_block_num: pl.Scalar[pl.INDEX] = pl.tile.get_block_num()
        for b in pl.range(spmd_idx, batch_count, spmd_block_num):
            cur_seq: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens, [b])
            start: pl.Scalar[pl.INDEX] = block_idx_kv * block_size_param
            remaining: pl.Scalar[pl.INDEX] = pl.cast(cur_seq, pl.INDEX) - start
            valid_len: pl.Scalar[pl.INDEX] = pl.max(pl.min(remaining, block_size_param), 0)
            if valid_len > 0:
                phys_block_raw: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [b * block_num + block_idx_kv])
                phys_block_idx: pl.Scalar[pl.INDEX] = pl.cast(phys_block_raw, pl.INDEX)
                vj_row: pl.Scalar[pl.INDEX] = phys_block_idx * block_size_param
                pij_l1: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(pij_batch, [b * 16, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat)
                vj_l1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(value_cache, [vj_row, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
                pij_l0a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(pij_l1, target_memory=pl.Mem.Left)
                vj_l0b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(vj_l1, target_memory=pl.Mem.Right)
                oi_l0c: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(pij_l0a, vj_l0b)
                oi_new_batch: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(oi_l0c, [b * 16, 0], oi_new_batch)
        return oi_new_batch
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def SpmdQkMatmul(query: pl.Tensor[[64, 128], pl.BF16], key_cache: pl.Tensor[[131072, 128], pl.BF16], sij_batch: pl.Out[pl.Tensor[[64, 128], pl.FP32]], block_table: pl.Tensor[[1024], pl.INT32], context_lens: pl.Tensor[[4], pl.INT32], batch_count: pl.Scalar[pl.INDEX], block_idx_kv: pl.Scalar[pl.INDEX], q_offset: pl.Scalar[pl.INDEX], block_num: pl.Scalar[pl.INDEX], block_size_param: pl.Scalar[pl.INDEX], num_heads_param: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64, 128], pl.FP32]:
        # noqa: PLR0913
        # QK matmul with SPMD stride partitioning over batch (CUBE).
        spmd_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        spmd_block_num: pl.Scalar[pl.INDEX] = pl.tile.get_block_num()
        for b in pl.range(spmd_idx, batch_count, spmd_block_num):
            qi_row: pl.Scalar[pl.INDEX] = b * num_heads_param + q_offset
            cur_seq: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens, [b])
            start: pl.Scalar[pl.INDEX] = block_idx_kv * block_size_param
            remaining: pl.Scalar[pl.INDEX] = pl.cast(cur_seq, pl.INDEX) - start
            valid_len: pl.Scalar[pl.INDEX] = pl.max(pl.min(remaining, block_size_param), 0)
            if valid_len > 0:
                phys_block_raw: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [b * block_num + block_idx_kv])
                phys_block_idx: pl.Scalar[pl.INDEX] = pl.cast(phys_block_raw, pl.INDEX)
                kj_row: pl.Scalar[pl.INDEX] = phys_block_idx * block_size_param
                qi_l1: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(query, [qi_row, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat)
                kj_nat: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(key_cache, [kj_row, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
                kj_l1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(kj_nat)
                qi_l0a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(qi_l1, target_memory=pl.Mem.Left)
                kj_l0b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(kj_l1, target_memory=pl.Mem.Right)
                sij_l0c: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(qi_l0a, kj_l0b)
                sij_batch: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(sij_l0c, [b * 16, 0], sij_batch)
        return sij_batch
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def SpmdSoftmaxPrepare(sij_batch: pl.Tensor[[64, 128], pl.FP32], pij_batch: pl.Out[pl.Tensor[[64, 128], pl.BF16]], mij_batch: pl.Out[pl.Tensor[[64, 1], pl.FP32]], lij_batch: pl.Out[pl.Tensor[[64, 1], pl.FP32]], scale_value: pl.Scalar[pl.FP32], context_lens: pl.Tensor[[4], pl.INT32], batch_count: pl.Scalar[pl.INDEX], block_idx_kv: pl.Scalar[pl.INDEX], block_size_param: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[64, 128], pl.BF16], pl.Tensor[[64, 1], pl.FP32], pl.Tensor[[64, 1], pl.FP32]]:
        # Softmax prepare with SPMD stride over batch (VECTOR).
        spmd_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        spmd_block_num: pl.Scalar[pl.INDEX] = pl.tile.get_block_num()
        for b in pl.range(spmd_idx, batch_count, spmd_block_num):
            cur_seq: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens, [b])
            start: pl.Scalar[pl.INDEX] = block_idx_kv * block_size_param
            remaining: pl.Scalar[pl.INDEX] = pl.cast(cur_seq, pl.INDEX) - start
            valid_len: pl.Scalar[pl.INDEX] = pl.max(pl.min(remaining, block_size_param), 0)
            if valid_len == 0:
                zero_pij_f32_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
                zero_pij_tile: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(zero_pij_f32_tile, target_type=pl.BF16, mode='round')
                zero_tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                zero_li_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(zero_pij_f32_tile, zero_tmp_tile)
                neg_inf_scores_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=pl.const(-1e+30, pl.FP32))
                neg_inf_tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                neg_inf_mi_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(neg_inf_scores_tile, neg_inf_tmp_tile)
                pij_batch: pl.Tensor[[64, 128], pl.BF16] = pl.tile.store(zero_pij_tile, [b * 16, 0], pij_batch, [16, 128])
                mij_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(neg_inf_mi_tile, [b * 16, 0], mij_batch, [16, 1])
                lij_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(zero_li_tile, [b * 16, 0], lij_batch, [16, 1])
            else:
                s_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[16, valid_len])] = pl.tile.load(sij_batch, [b * 16, 0], [16, 128], [16, valid_len], target_memory=pl.Mem.Vec)
                s_padded: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
                scaled: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.muls(s_padded, scale_value)
                tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                mi_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(scaled, tmp_tile)
                sij_centered: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(scaled, mi_tile)
                exp_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(sij_centered)
                pij_tile_f16: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.cast(exp_tile, target_type=pl.BF16, mode='round')
                pij_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.cast(pij_tile_f16, target_type=pl.FP32, mode='round')
                li_tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(pij_tile, tmp_tile)
                pij_batch: pl.Tensor[[64, 128], pl.BF16] = pl.tile.store(pij_tile_f16, [b * 16, 0], pij_batch, [16, 128])
                mij_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(mi_tile, [b * 16, 0], mij_batch, [16, 1])
                lij_batch: pl.Tensor[[64, 1], pl.FP32] = pl.tile.store(li_tile, [b * 16, 0], lij_batch, [16, 1])
        return pij_batch, mij_batch, lij_batch
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def paged_attention_spmd(self, query: pl.Tensor[[64, 128], pl.BF16], key_cache: pl.Tensor[[131072, 128], pl.BF16], value_cache: pl.Tensor[[131072, 128], pl.BF16], block_table: pl.Tensor[[1024], pl.INT32], context_lens: pl.Tensor[[4], pl.INT32], scale: pl.Tensor[[1], pl.FP32], out: pl.Out[pl.Tensor[[64, 128], pl.FP32]], config: pl.Tensor[[7], pl.INT64]) -> pl.Tensor[[64, 128], pl.FP32]:
        # SPMD paged attention orchestration.
        # 
        #         Outer loop over q_tile groups, inner loop over KV blocks.
        #         Each kernel stage dispatched as SPMD across multiple hardware blocks.
        #         Config: [batch, num_heads, kv_head_num, head_dim,
        #                  block_size, block_num_capacity, active_block_num]
        #         
        batch_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
        num_heads_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
        block_size_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [4])
        block_num_capacity_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [5])
        active_block_num_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [6])
        scale_value: pl.Scalar[pl.FP32] = pl.tensor.read(scale, [0])
        zero_bn_cfg: pl.Scalar[pl.INT64] = pl.const(0, pl.INT64)
        if batch_cfg == 0:
            max_bn_cfg: pl.Scalar[pl.INT64] = pl.yield_(zero_bn_cfg)
        else:
            first_seq: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens, [0])
            first_bn_cfg: pl.Scalar[pl.INT64] = (pl.cast(first_seq, pl.INT64) + block_size_cfg - 1) // block_size_cfg
            max_bn_cfg: pl.Scalar[pl.INT64] = pl.yield_(first_bn_cfg)
        for b in pl.range(1, batch_cfg):
            cur_seq_b: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens, [b])
            bn_b: pl.Scalar[pl.INT64] = (pl.cast(cur_seq_b, pl.INT64) + block_size_cfg - 1) // block_size_cfg
            max_bn_cfg: pl.Scalar[pl.INT64] = pl.max(max_bn_cfg, bn_b)
        max_bn_cfg: pl.Scalar[pl.INT64] = pl.min(max_bn_cfg, active_block_num_cfg)
        q_loop_cfg: pl.Scalar[pl.INT64] = (num_heads_cfg + 16 - 1) // 16
        for q_idx in pl.range(q_loop_cfg):
            q_offset: pl.Scalar[pl.INDEX] = q_idx * 16
            # Per-batch accumulators for online softmax across KV blocks.
            mi_a: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.create([64, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            li_a: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.create([64, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            oi_a: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.create([64, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for bn in pl.range(max_bn_cfg):
                # Temporary tensors for this KV block iteration.
                sij_b: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.create([64, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                pij_b: pl.Tensor[[64, 128], pl.BF16] = pl.tensor.create([64, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                mij_b: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.create([64, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                lij_b: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.create([64, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                oi_new_b: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.create([64, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                # sync_start ensures all HW blocks begin together before
                # the cooperative SPMD wave.
                with pl.spmd(4, sync_start=True):
                    sij_b: pl.Tensor[[64, 128], pl.FP32] = self.SpmdQkMatmul(query, key_cache, sij_b, block_table, context_lens, batch_cfg, bn, q_offset, block_num_capacity_cfg, block_size_cfg, num_heads_cfg)
                with pl.cluster():
                    with pl.spmd(4):
                        _tuple_tmp: pl.Tuple[pl.Tensor[[64, 128], pl.BF16], pl.Tensor[[64, 1], pl.FP32], pl.Tensor[[64, 1], pl.FP32]] = self.SpmdSoftmaxPrepare(sij_b, pij_b, mij_b, lij_b, scale_value, context_lens, batch_cfg, bn, block_size_cfg)
                        pij_b: pl.Tensor[[64, 128], pl.BF16] = _tuple_tmp[0]
                        mij_b: pl.Tensor[[64, 1], pl.FP32] = _tuple_tmp[1]
                        lij_b: pl.Tensor[[64, 1], pl.FP32] = _tuple_tmp[2]
                with pl.spmd(4):
                    oi_new_b: pl.Tensor[[64, 128], pl.FP32] = self.SpmdPvMatmul(pij_b, value_cache, oi_new_b, block_table, context_lens, batch_cfg, bn, block_num_capacity_cfg, block_size_cfg)
                if bn == 0:
                    is_first: pl.Scalar[pl.INDEX] = pl.yield_(1)
                else:
                    is_first: pl.Scalar[pl.INDEX] = pl.yield_(0)
                with pl.cluster():
                    with pl.spmd(4):
                        _tuple_tmp_1: pl.Tuple[pl.Tensor[[64, 1], pl.FP32], pl.Tensor[[64, 1], pl.FP32], pl.Tensor[[64, 128], pl.FP32]] = self.SpmdOnlineUpdate(mij_b, lij_b, oi_new_b, mi_a, li_a, oi_a, is_first, batch_cfg)
                        mi_a: pl.Tensor[[64, 1], pl.FP32] = _tuple_tmp_1[0]
                        li_a: pl.Tensor[[64, 1], pl.FP32] = _tuple_tmp_1[1]
                        oi_a: pl.Tensor[[64, 128], pl.FP32] = _tuple_tmp_1[2]
            # Final normalization: out = oi / li, scattered to output tensor.
            with pl.spmd(4):
                out: pl.Tensor[[64, 128], pl.FP32] = self.SpmdNormalize(oi_a, li_a, out, batch_cfg, q_offset, num_heads_cfg)
        return out
