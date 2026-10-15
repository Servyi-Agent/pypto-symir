# pypto.program: _jit_entry
import pypto.language as pl

V41_B_DYN = pl.dynamic("V41_B_DYN")
V41_CMP_POSITIONS_DYN = pl.dynamic("V41_CMP_POSITIONS_DYN")
V41_INDEX_BLOCKS_DYN = pl.dynamic("V41_INDEX_BLOCKS_DYN")
V41_TABLE_DYN = pl.dynamic("V41_TABLE_DYN")
V41_T_DYN = pl.dynamic("V41_T_DYN")

@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.Inline)
    def _index_key_decode_lut(self):
        # Map packed FP4 nibbles to exact FP8 carrier bits.
        codes: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False)
        magnitude: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.ands(codes, pl.const(7, pl.INT32))
        values: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.shls(magnitude, pl.const(2, pl.INT32)), pl.const(48, pl.INT32))
        values_v1: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.select(pl.tile.cmps(magnitude, pl.const(0, pl.INT32), cmp_type=0), pl.const(0, pl.INT32), values)
        values_v2: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.select(pl.tile.cmps(magnitude, pl.const(1, pl.INT32), cmp_type=0), pl.const(48, pl.INT32), values_v1)
        values_v3: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.or_(values_v2, pl.tile.shls(pl.tile.ands(codes, pl.const(8, pl.INT32)), pl.const(4, pl.INT32)))
        return pl.tile.cast(values_v3, target_type=pl.INT16, mode='round')
    @pl.function(type=pl.FunctionType.Inline)
    def score(self, query: pl.Tensor[[V41_T_DYN, 4096], pl.FP8E4M3FN], query_scale: pl.Tensor[[V41_T_DYN, 128], pl.FP8E8M0], weights: pl.Tensor[[V41_T_DYN, 32], pl.BF16], index_cache: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 64], pl.UINT8], index_cache_scale: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 4], pl.FP8E8M0], block_table: pl.Tensor[[V41_B_DYN, V41_TABLE_DYN], pl.INT32], request_ids: pl.Tensor[[V41_T_DYN], pl.INT32], lengths: pl.Tensor[[V41_T_DYN], pl.INT32], candidates: pl.Tensor[[V41_T_DYN, V41_CMP_POSITIONS_DYN], pl.UINT8], output: pl.Tensor[[8, 1024], pl.FP32], num_tokens: pl.Scalar[pl.INT32], cache_ready: pl.Scalar[pl.TASK_ID], query_ready: pl.Scalar[pl.TASK_ID], weights_ready: pl.Scalar[pl.TASK_ID]):
        width: pl.Scalar[pl.INDEX] = pl.tensor.dim(output, 1)
        cache_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(index_cache, 0) * 128
        cache_flat: pl.Tensor[[cache_rows, 64], pl.UINT8] = pl.tensor.reshape(index_cache, [cache_rows, 64])
        scale_flat: pl.Tensor[[1, cache_rows * pl.const(4, pl.INDEX)], pl.FP8E8M0] = pl.tensor.reshape(index_cache_scale, [1, cache_rows * 4])
        query_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(query, 0) * 32
        query_flat: pl.Tensor[[query_rows, 128], pl.FP8E4M3FN] = pl.tensor.reshape(query, [query_rows, 128])
        macro_tiles: pl.Scalar[pl.INDEX] = (width + 512 - 1) // 512
        work_items: pl.Scalar[pl.INDEX] = pl.cast(num_tokens, pl.INDEX) // 2 * macro_tiles
        with pl.spmd(32, name_hint="c1a_index_score_paired_block_mxfp8_spmd", optimizations=[pl.cross_core_slot(slot_num=2)], deps=[cache_ready, query_ready, weights_ready]) as score_tid:
            worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            lut: pl.Tile[[1, 16], pl.INT16, pl.Mem.Vec] = self._index_key_decode_lut()
            for work in pl.range(worker, work_items, 32):
                first_token: pl.Scalar[pl.INDEX] = work // macro_tiles * 2
                macro_index: pl.Scalar[pl.INDEX] = work % macro_tiles
                logical_begin: pl.Scalar[pl.INDEX] = macro_index * 512
                second_token: pl.Scalar[pl.INDEX] = first_token + 1
                first_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [first_token])
                first_visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [first_token]), pl.INDEX), width)
                second_visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [second_token]), pl.INDEX), width)
                maximum_visible: pl.Scalar[pl.INDEX] = pl.max(first_visible, second_visible)
                query_pair: pl.Tile[[64, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.load(query_flat, [first_token * 32, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec)
                pair_scale_raw: pl.Tile[[2, 128], pl.UINT8, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.load(query_scale, [first_token, 0], [2, 128], [2, 128], target_memory=pl.Mem.Vec), dtype=pl.UINT8)
                pair_scale_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                pair_scale_zz: pl.Tile[[64, 4], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.reinterpret_view(pl.tile.tmov_x2zz(pl.tile.reshape(pair_scale_raw, [1, 256]), pair_scale_tmp, group_axis=1, dst_rows=64, dst_cols=4), dtype=pl.FP8E8M0)
                query_mat: pl.Tile[[64, 128], pl.FP8E4M3FN, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.move(query_pair, target_memory=pl.Mem.Mat, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major)
                query_left: pl.Tile[[64, 128], pl.FP8E4M3FN, pl.Mem.Left] = pl.tile.move(query_mat, target_memory=pl.Mem.Left)
                pair_scale_mat: pl.Tile[[64, 4], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.move(pair_scale_zz, target_memory=pl.Mem.Mat, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major)
                pair_scale_left: pl.Tile[[64, 4], pl.FP8E8M0, pl.Mem.LeftScale] = pl.tile.move(pair_scale_mat, target_memory=pl.Mem.LeftScale)
                for half_index in pl.unroll(2):
                    half_begin: pl.Scalar[pl.INDEX] = logical_begin + half_index * 256
                    if half_begin < maximum_visible:
                        packed: pl.Tile[[256, 64], pl.INT8, pl.Mem.Vec] = pl.tile.full([256, 64], dtype=pl.INT8, value=pl.const(0, pl.INT8))
                        raw_scale: pl.Tile[[1, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.full([1, 1024], dtype=pl.INT8, value=pl.const(127, pl.INT8))
                        for page_offset in pl.unroll(2):
                            page_begin: pl.Scalar[pl.INDEX] = half_begin + page_offset * 128
                            if page_begin < maximum_visible:
                                page_id: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [pl.cast(first_request, pl.INDEX), page_begin // 128])
                                if pl.cast(page_id, pl.INDEX) >= 0:
                                    page_row: pl.Scalar[pl.INDEX] = pl.cast(page_id, pl.INDEX) * 128
                                    page_packed: pl.Tile[[128, 64], pl.INT8, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.load(cache_flat, [page_row, 0], [128, 64], [128, 64], target_memory=pl.Mem.Vec), dtype=pl.INT8)
                                    packed: pl.Tile[[256, 64], pl.INT8, pl.Mem.Vec] = pl.tile.assemble(packed, page_packed, [page_offset * 128, 0])
                                    page_scale: pl.Tile[[1, 512], pl.INT8, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.reinterpret_view(pl.tile.load(scale_flat, [0, page_row * 4], [1, 512], [1, 512], target_memory=pl.Mem.Vec), dtype=pl.UINT8), dtype=pl.INT8)
                                    raw_scale: pl.Tile[[1, 1024], pl.INT8, pl.Mem.Vec] = pl.tile.assemble(raw_scale, page_scale, [0, page_offset * 128 * 4])
                        byte_values: pl.Tile[[256, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(pl.tile.cast(packed, target_type=pl.INT16, mode='round'), pl.const(255, pl.INT16))
                        low: pl.Tile[[256, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(byte_values, pl.const(15, pl.INT16))
                        high: pl.Tile[[256, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(pl.tile.shrs(byte_values, pl.const(4, pl.INT16)), pl.const(15, pl.INT16))
                        lut_tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                        low_bytes: pl.Tile[[256, 64], pl.INT16, pl.Mem.Vec] = pl.tile.gather(lut, low, lut_tmp)
                        high_bytes: pl.Tile[[256, 64], pl.INT16, pl.Mem.Vec] = pl.tile.gather(lut, high, lut_tmp)
                        interleaved_words: pl.Tile[[256, 64], pl.INT16, pl.Mem.Vec] = pl.tile.or_(low_bytes, pl.tile.shls(high_bytes, pl.const(8, pl.INT16)))
                        keys: pl.Tile[[256, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.reinterpret_view(interleaved_words, dtype=pl.INT8, shape=[256, 128]), dtype=pl.FP8E4M3FN)
                        key_scale_tmp: pl.Tile[[1, 128], pl.UINT8] = pl.tile.create([1, 128], dtype=pl.UINT8)
                        key_scale_zz: pl.Tile[[256, 4], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.reinterpret_view(pl.tile.tmov_x2zz(pl.tile.reinterpret_view(raw_scale, dtype=pl.UINT8), key_scale_tmp, group_axis=1, dst_rows=256, dst_cols=4), dtype=pl.FP8E8M0)
                        key_scale_nn: pl.Tile[[4, 256], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.transpose_view(key_scale_zz)
                        keys_kn: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.transpose(pl.tile.reinterpret_view(keys, dtype=pl.INT8), 0, 1), dtype=pl.FP8E4M3FN)
                        dot_bf16: pl.Tile[[64, 256], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pl.tile.matmul_mx(query_left, pair_scale_left, keys_kn, key_scale_nn), target_type=pl.BF16, mode='rint')
                        for panel_index in pl.unroll(2):
                            output_begin: pl.Scalar[pl.INDEX] = half_begin + panel_index * 128
                            if output_begin < width:
                                output_count: pl.Scalar[pl.INDEX] = pl.min(128, width - output_begin)
                                for pair_offset in pl.unroll(2):
                                    token: pl.Scalar[pl.INDEX] = first_token + pair_offset
                                    visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [token]), pl.INDEX), width)
                                    negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                    masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32)
                                    if output_begin < visible:
                                        valid_count: pl.Scalar[pl.INDEX] = pl.min(128, visible - output_begin)
                                        page_id: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [pl.cast(first_request, pl.INDEX), output_begin // 128])
                                        if pl.cast(page_id, pl.INDEX) >= 0:
                                            head_dot: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.slice(dot_bf16, [32, 128], [pair_offset * 32, panel_index * 128])
                                            relu: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.maximums(pl.tile.cast(head_dot, target_type=pl.FP32, mode='round'), 0.0)
                                            head_weight: pl.Tile[[1, 32], pl.BF16, pl.Mem.Vec] = pl.tile.load(weights, [token, 0], [1, 32], [1, 32], target_memory=pl.Mem.Vec)
                                            head_weight_v1: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(head_weight, target_type=pl.FP32, mode='round'), 0.015625)
                                            rounded_weight: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(head_weight_v1, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                            weighted: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.row_expand_mul(relu, pl.tile.reshape(rounded_weight, [32, 1]))
                                            weighted_v1: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pl.tile.cast(weighted, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                            total: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.col_sum(weighted_v1), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                            if False:
                                                candidate: pl.Tile[[1, 128], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.reinterpret_view(pl.tile.load(candidates, [token, output_begin], [1, 128], [1, valid_count], target_memory=pl.Mem.Vec), dtype=pl.INT8)
                                                for lane in pl.range(valid_count):
                                                    if pl.cast(pl.tile.read(candidate, [0, lane]), pl.INDEX) != 0:
                                                        pl.tile.write(masked, [0, lane], pl.tile.read(total, [0, lane]))
                                            else:
                                                lanes: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                                                in_range: pl.Tile[[1, 32], pl.UINT8, pl.Mem.Vec, pl.TileView(valid_shape=[1, 16])] = pl.tile.cmps(lanes, pl.cast(valid_count, pl.INT32), cmp_type=2)
                                                masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.select(in_range, total, masked)
                                    pl.tile.store(pl.tile.set_validshape(masked, 1, output_count), [token, output_begin], output)
                    else:
                        for panel_index_1 in pl.unroll(2):
                            output_begin: pl.Scalar[pl.INDEX] = half_begin + panel_index_1 * 128
                            if output_begin < width:
                                output_count: pl.Scalar[pl.INDEX] = pl.min(128, width - output_begin)
                                for pair_offset_1 in pl.unroll(2):
                                    token: pl.Scalar[pl.INDEX] = first_token + pair_offset_1
                                    negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                    pl.tile.store(pl.tile.set_validshape(pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32), 1, output_count), [token, output_begin], output)
        return score_tid
    @pl.function(type=pl.FunctionType.Inline)
    def score__2(self, query: pl.Tensor[[V41_T_DYN, 4096], pl.FP8E4M3FN], query_scale: pl.Tensor[[V41_T_DYN, 128], pl.FP8E8M0], weights: pl.Tensor[[V41_T_DYN, 32], pl.BF16], index_cache: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 64], pl.UINT8], index_cache_scale: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 4], pl.FP8E8M0], block_table: pl.Tensor[[V41_B_DYN, V41_TABLE_DYN], pl.INT32], request_ids: pl.Tensor[[V41_T_DYN], pl.INT32], lengths: pl.Tensor[[V41_T_DYN], pl.INT32], candidates: pl.Tensor[[V41_T_DYN, V41_CMP_POSITIONS_DYN], pl.UINT8], output: pl.Tensor[[8, 1024], pl.FP32], num_tokens: pl.Scalar[pl.INT32], cache_ready: pl.Scalar[pl.TASK_ID], query_ready: pl.Scalar[pl.TASK_ID], weights_ready: pl.Scalar[pl.TASK_ID]):
        width: pl.Scalar[pl.INDEX] = pl.tensor.dim(output, 1)
        cache_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(index_cache, 0) * 128
        cache_flat: pl.Tensor[[cache_rows, 64], pl.UINT8] = pl.tensor.reshape(index_cache, [cache_rows, 64])
        scale_flat: pl.Tensor[[1, cache_rows * pl.const(4, pl.INDEX)], pl.FP8E8M0] = pl.tensor.reshape(index_cache_scale, [1, cache_rows * 4])
        query_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(query, 0) * 32
        query_flat: pl.Tensor[[query_rows, 128], pl.FP8E4M3FN] = pl.tensor.reshape(query, [query_rows, 128])
        tiles: pl.Scalar[pl.INDEX] = (width + 128 - 1) // 128
        work_items: pl.Scalar[pl.INDEX] = (pl.cast(num_tokens, pl.INDEX) + 1) // 2 * tiles
        with pl.spmd(16, name_hint="c1a_index_score_page_mxfp8_spmd", deps=[cache_ready, query_ready, weights_ready]) as score_tid:
            lut: pl.Tile[[1, 16], pl.INT16, pl.Mem.Vec] = self._index_key_decode_lut()
            worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            for work in pl.range(worker, work_items, 16):
                first_token: pl.Scalar[pl.INDEX] = work // tiles * 2
                panel: pl.Scalar[pl.INDEX] = work % tiles
                logical_begin: pl.Scalar[pl.INDEX] = panel * 128
                output_count: pl.Scalar[pl.INDEX] = pl.min(128, width - logical_begin)
                first_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [first_token])
                second_token: pl.Scalar[pl.INDEX] = pl.min(first_token + 1, pl.cast(num_tokens, pl.INDEX) - 1)
                second_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [second_token])
                for anchor_offset in pl.range(2):
                    anchor: pl.Scalar[pl.INDEX] = first_token + anchor_offset
                    if anchor < pl.cast(num_tokens, pl.INDEX):
                        if anchor_offset == 0 or first_request != second_request:
                            anchor_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [anchor])
                            maximum_visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [anchor]), pl.INDEX), width)
                            if first_request == second_request:
                                maximum_visible: pl.Scalar[pl.INDEX] = pl.max(maximum_visible, pl.min(pl.cast(pl.tensor.read(lengths, [second_token]), pl.INDEX), width))
                            if logical_begin < maximum_visible:
                                physical_page_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [pl.cast(anchor_request, pl.INDEX), logical_begin // 128])
                                if pl.cast(physical_page_i32, pl.INDEX) >= 0:
                                    physical_page: pl.Scalar[pl.INDEX] = pl.cast(physical_page_i32, pl.INDEX)
                                    physical_row: pl.Scalar[pl.INDEX] = physical_page * 128 + logical_begin % 128
                                    packed: pl.Tile[[128, 64], pl.UINT8] = pl.tile.load(cache_flat, [physical_row, 0], [128, 64], [128, 64])
                                    byte_values: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(pl.tile.cast(pl.tile.reinterpret_view(packed, dtype=pl.INT8), target_type=pl.INT16, mode='round'), pl.const(255, pl.INT16))
                                    low: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(byte_values, pl.const(15, pl.INT16))
                                    high: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(pl.tile.shrs(byte_values, pl.const(4, pl.INT16)), pl.const(15, pl.INT16))
                                    lut_tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                                    low_bytes: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.gather(lut, low, lut_tmp)
                                    high_bytes: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.gather(lut, high, lut_tmp)
                                    interleaved_words: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.or_(low_bytes, pl.tile.shls(high_bytes, pl.const(8, pl.INT16)))
                                    keys: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.reinterpret_view(interleaved_words, dtype=pl.INT8, shape=[128, 128]), dtype=pl.FP8E4M3FN)
                                    raw_scale: pl.Tile[[1, 512], pl.UINT8] = pl.tile.reinterpret_view(pl.tile.load(scale_flat, [0, physical_row * 4], [1, 512], [1, 512]), dtype=pl.UINT8)
                                    scale_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                                    key_scale_zz: pl.Tile[[128, 4], pl.UINT8, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.tmov_x2zz(raw_scale, scale_tmp, group_axis=1, dst_rows=128, dst_cols=4)
                                    key_scale_nn: pl.Tile[[4, 128], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.transpose_view(pl.tile.reinterpret_view(key_scale_zz, dtype=pl.FP8E8M0))
                                    keys_kn: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.transpose(pl.tile.reinterpret_view(keys, dtype=pl.INT8), 0, 1), dtype=pl.FP8E4M3FN)
                                    keys_mat: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Mat] = pl.tile.move(keys_kn, target_memory=pl.Mem.Mat, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)
                                    keys_right: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Right] = pl.tile.move(keys_mat, target_memory=pl.Mem.Right)
                                    key_scale_mat: pl.Tile[[4, 128], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.move(key_scale_nn, target_memory=pl.Mem.Mat, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major)
                                    key_scale_right: pl.Tile[[4, 128], pl.FP8E8M0, pl.Mem.RightScale] = pl.tile.move(key_scale_mat, target_memory=pl.Mem.RightScale)
                                    if second_token > first_token and first_request == second_request:
                                        query_pair: pl.Tile[[64, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.load(query_flat, [first_token * 32, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec)
                                        pair_scale_raw: pl.Tile[[2, 128], pl.UINT8] = pl.tile.reinterpret_view(pl.tile.load(query_scale, [first_token, 0], [2, 128], [2, 128]), dtype=pl.UINT8)
                                        pair_scale_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                                        pair_scale_zz: pl.Tile[[64, 4], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.reinterpret_view(pl.tile.tmov_x2zz(pl.tile.reshape(pair_scale_raw, [1, 256]), pair_scale_tmp, group_axis=1, dst_rows=64, dst_cols=4), dtype=pl.FP8E8M0)
                                        pair_dot: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_mx(query_pair, pair_scale_zz, keys_right, key_scale_right)
                                        pair_dot_v1: pl.Tile[[64, 128], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pair_dot, target_type=pl.BF16, mode='rint')
                                        for pair_query_offset in pl.unroll(2):
                                            pair_token: pl.Scalar[pl.INDEX] = first_token + pair_query_offset
                                            pair_negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                            pair_masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(pair_negative_bits, dtype=pl.FP32)
                                            pair_visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [pair_token]), pl.INDEX), width)
                                            if logical_begin < pair_visible:
                                                pair_valid_count: pl.Scalar[pl.INDEX] = pl.min(128, pair_visible - logical_begin)
                                                pair_head_dot: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.slice(pair_dot_v1, [32, 128], [pair_query_offset * 32, 0])
                                                pair_relu: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.maximums(pl.tile.cast(pair_head_dot, target_type=pl.FP32, mode='round'), 0.0)
                                                pair_head_weight: pl.Tile[[1, 32], pl.BF16] = pl.tile.load(weights, [pair_token, 0], [1, 32], [1, 32])
                                                pair_head_weight_v1: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pair_head_weight, target_type=pl.FP32, mode='round'), 0.015625)
                                                pair_rounded_weight: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pair_head_weight_v1, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                pair_weighted: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.row_expand_mul(pair_relu, pl.tile.reshape(pair_rounded_weight, [32, 1]))
                                                pair_weighted_v1: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pl.tile.cast(pair_weighted, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                pair_total: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.col_sum(pair_weighted_v1), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                if False:
                                                    pair_candidate: pl.Tile[[1, 128], pl.UINT8, pl.TileView(valid_shape=[1, pair_valid_count])] = pl.tile.load(candidates, [pair_token, logical_begin], [1, 128], [1, pair_valid_count])
                                                    pair_candidate_v1: pl.Tile[[1, 128], pl.INT8, pl.TileView(valid_shape=[1, pair_valid_count])] = pl.tile.reinterpret_view(pair_candidate, dtype=pl.INT8)
                                                    for pair_lane in pl.range(pair_valid_count):
                                                        if pl.cast(pl.tile.read(pair_candidate_v1, [0, pair_lane]), pl.INDEX) != 0:
                                                            pl.tile.write(pair_masked, [0, pair_lane], pl.tile.read(pair_total, [0, pair_lane]))
                                                else:
                                                    pair_valid_count_i32: pl.Scalar[pl.INT32] = pl.cast(pair_valid_count, pl.INT32)
                                                    pair_lanes_i32: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                                                    pair_predicate: pl.Tile[[1, 32], pl.UINT8, pl.Mem.Vec, pl.TileView(valid_shape=[1, 16])] = pl.tile.cmps(pair_lanes_i32, pair_valid_count_i32, cmp_type=2)
                                                    pair_masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.select(pair_predicate, pair_total, pair_masked)
                                            pl.tile.store(pl.tile.set_validshape(pair_masked, 1, output_count), [pair_token, logical_begin], output)
                                    else:
                                        for query_offset in pl.range(2):
                                            token: pl.Scalar[pl.INDEX] = first_token + query_offset
                                            if token < pl.cast(num_tokens, pl.INDEX):
                                                query_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token])
                                                if query_request == anchor_request:
                                                    negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                                    masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32)
                                                    visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [token]), pl.INDEX), width)
                                                    if logical_begin < visible:
                                                        valid_count: pl.Scalar[pl.INDEX] = pl.min(128, visible - logical_begin)
                                                        query_panel: pl.Tile[[32, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.load(query_flat, [token * 32, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec)
                                                        raw_query_scale: pl.Tile[[1, 128], pl.UINT8] = pl.tile.reinterpret_view(pl.tile.load(query_scale, [token, 0], [1, 128], [1, 128]), dtype=pl.UINT8)
                                                        query_scale_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                                                        query_scale_zz: pl.Tile[[32, 4], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.reinterpret_view(pl.tile.tmov_x2zz(raw_query_scale, query_scale_tmp, group_axis=1, dst_rows=32, dst_cols=4), dtype=pl.FP8E8M0)
                                                        dot: pl.Tile[[32, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_mx(query_panel, query_scale_zz, keys_right, key_scale_right)
                                                        dot_v1: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(dot, target_type=pl.BF16, mode='rint')
                                                        relu: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.maximums(pl.tile.cast(dot_v1, target_type=pl.FP32, mode='round'), 0.0)
                                                        head_weight: pl.Tile[[1, 32], pl.BF16] = pl.tile.load(weights, [token, 0], [1, 32], [1, 32])
                                                        head_weight_v1: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(head_weight, target_type=pl.FP32, mode='round'), 0.015625)
                                                        rounded_weight: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(head_weight_v1, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                        weighted: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.row_expand_mul(relu, pl.tile.reshape(rounded_weight, [32, 1]))
                                                        weighted_v1: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pl.tile.cast(weighted, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                        total: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.col_sum(weighted_v1), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                        if False:
                                                            candidate: pl.Tile[[1, 128], pl.UINT8, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(candidates, [token, logical_begin], [1, 128], [1, valid_count])
                                                            candidate_v1: pl.Tile[[1, 128], pl.INT8, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.reinterpret_view(candidate, dtype=pl.INT8)
                                                            for lane in pl.range(valid_count):
                                                                if pl.cast(pl.tile.read(candidate_v1, [0, lane]), pl.INDEX) != 0:
                                                                    pl.tile.write(masked, [0, lane], pl.tile.read(total, [0, lane]))
                                                        else:
                                                            valid_count_i32: pl.Scalar[pl.INT32] = pl.cast(valid_count, pl.INT32)
                                                            lanes_i32: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                                                            predicate: pl.Tile[[1, 32], pl.UINT8, pl.Mem.Vec, pl.TileView(valid_shape=[1, 16])] = pl.tile.cmps(lanes_i32, valid_count_i32, cmp_type=2)
                                                            masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.select(predicate, total, masked)
                                                    pl.tile.store(pl.tile.set_validshape(masked, 1, output_count), [token, logical_begin], output)
                                else:
                                    for query_offset_1 in pl.range(2):
                                        token: pl.Scalar[pl.INDEX] = first_token + query_offset_1
                                        if token < pl.cast(num_tokens, pl.INDEX):
                                            query_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token])
                                            if query_request == anchor_request:
                                                negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                                masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32)
                                                pl.tile.store(pl.tile.set_validshape(masked, 1, output_count), [token, logical_begin], output)
                            else:
                                for query_offset_2 in pl.range(2):
                                    token: pl.Scalar[pl.INDEX] = first_token + query_offset_2
                                    if token < pl.cast(num_tokens, pl.INDEX):
                                        query_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token])
                                        if query_request == anchor_request:
                                            negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                            masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32)
                                            pl.tile.store(pl.tile.set_validshape(masked, 1, output_count), [token, logical_begin], output)
        return score_tid
    @pl.function(type=pl.FunctionType.Inline)
    def score__3(self, query: pl.Tensor[[V41_T_DYN, 4096], pl.FP8E4M3FN], query_scale: pl.Tensor[[V41_T_DYN, 128], pl.FP8E8M0], weights: pl.Tensor[[V41_T_DYN, 32], pl.BF16], index_cache: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 64], pl.UINT8], index_cache_scale: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 4], pl.FP8E8M0], block_table: pl.Tensor[[V41_B_DYN, V41_TABLE_DYN], pl.INT32], request_ids: pl.Tensor[[V41_T_DYN], pl.INT32], lengths: pl.Tensor[[V41_T_DYN], pl.INT32], candidates: pl.Tensor[[V41_T_DYN, V41_CMP_POSITIONS_DYN], pl.UINT8], output: pl.Tensor[[8, 1024], pl.FP32], num_tokens: pl.Scalar[pl.INT32], cache_ready: pl.Scalar[pl.TASK_ID], query_ready: pl.Scalar[pl.TASK_ID], weights_ready: pl.Scalar[pl.TASK_ID]):
        width: pl.Scalar[pl.INDEX] = pl.tensor.dim(output, 1)
        cache_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(index_cache, 0) * 128
        cache_flat: pl.Tensor[[cache_rows, 64], pl.UINT8] = pl.tensor.reshape(index_cache, [cache_rows, 64])
        scale_flat: pl.Tensor[[1, cache_rows * pl.const(4, pl.INDEX)], pl.FP8E8M0] = pl.tensor.reshape(index_cache_scale, [1, cache_rows * 4])
        query_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(query, 0) * 32
        query_flat: pl.Tensor[[query_rows, 128], pl.FP8E4M3FN] = pl.tensor.reshape(query, [query_rows, 128])
        tiles: pl.Scalar[pl.INDEX] = (width + 128 - 1) // 128
        work_items: pl.Scalar[pl.INDEX] = (pl.cast(num_tokens, pl.INDEX) + 1) // 2 * tiles
        with pl.spmd(32, name_hint="c1a_index_score_page_mxfp8_spmd", deps=[cache_ready, query_ready, weights_ready]) as score_tid:
            lut: pl.Tile[[1, 16], pl.INT16, pl.Mem.Vec] = self._index_key_decode_lut()
            worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            for work in pl.range(worker, work_items, 32):
                first_token: pl.Scalar[pl.INDEX] = work // tiles * 2
                panel: pl.Scalar[pl.INDEX] = work % tiles
                logical_begin: pl.Scalar[pl.INDEX] = panel * 128
                output_count: pl.Scalar[pl.INDEX] = pl.min(128, width - logical_begin)
                first_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [first_token])
                second_token: pl.Scalar[pl.INDEX] = pl.min(first_token + 1, pl.cast(num_tokens, pl.INDEX) - 1)
                second_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [second_token])
                for anchor_offset in pl.range(2):
                    anchor: pl.Scalar[pl.INDEX] = first_token + anchor_offset
                    if anchor < pl.cast(num_tokens, pl.INDEX):
                        if anchor_offset == 0 or first_request != second_request:
                            anchor_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [anchor])
                            maximum_visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [anchor]), pl.INDEX), width)
                            if first_request == second_request:
                                maximum_visible: pl.Scalar[pl.INDEX] = pl.max(maximum_visible, pl.min(pl.cast(pl.tensor.read(lengths, [second_token]), pl.INDEX), width))
                            if logical_begin < maximum_visible:
                                physical_page_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [pl.cast(anchor_request, pl.INDEX), logical_begin // 128])
                                if pl.cast(physical_page_i32, pl.INDEX) >= 0:
                                    physical_page: pl.Scalar[pl.INDEX] = pl.cast(physical_page_i32, pl.INDEX)
                                    physical_row: pl.Scalar[pl.INDEX] = physical_page * 128 + logical_begin % 128
                                    packed: pl.Tile[[128, 64], pl.UINT8] = pl.tile.load(cache_flat, [physical_row, 0], [128, 64], [128, 64])
                                    byte_values: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(pl.tile.cast(pl.tile.reinterpret_view(packed, dtype=pl.INT8), target_type=pl.INT16, mode='round'), pl.const(255, pl.INT16))
                                    low: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(byte_values, pl.const(15, pl.INT16))
                                    high: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.ands(pl.tile.shrs(byte_values, pl.const(4, pl.INT16)), pl.const(15, pl.INT16))
                                    lut_tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                                    low_bytes: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.gather(lut, low, lut_tmp)
                                    high_bytes: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.gather(lut, high, lut_tmp)
                                    interleaved_words: pl.Tile[[128, 64], pl.INT16, pl.Mem.Vec] = pl.tile.or_(low_bytes, pl.tile.shls(high_bytes, pl.const(8, pl.INT16)))
                                    keys: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.reinterpret_view(interleaved_words, dtype=pl.INT8, shape=[128, 128]), dtype=pl.FP8E4M3FN)
                                    raw_scale: pl.Tile[[1, 512], pl.UINT8] = pl.tile.reinterpret_view(pl.tile.load(scale_flat, [0, physical_row * 4], [1, 512], [1, 512]), dtype=pl.UINT8)
                                    scale_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                                    key_scale_zz: pl.Tile[[128, 4], pl.UINT8, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.tmov_x2zz(raw_scale, scale_tmp, group_axis=1, dst_rows=128, dst_cols=4)
                                    key_scale_nn: pl.Tile[[4, 128], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.transpose_view(pl.tile.reinterpret_view(key_scale_zz, dtype=pl.FP8E8M0))
                                    keys_kn: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.reinterpret_view(pl.tile.transpose(pl.tile.reinterpret_view(keys, dtype=pl.INT8), 0, 1), dtype=pl.FP8E4M3FN)
                                    keys_mat: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Mat] = pl.tile.move(keys_kn, target_memory=pl.Mem.Mat, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)
                                    keys_right: pl.Tile[[128, 128], pl.FP8E4M3FN, pl.Mem.Right] = pl.tile.move(keys_mat, target_memory=pl.Mem.Right)
                                    key_scale_mat: pl.Tile[[4, 128], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.move(key_scale_nn, target_memory=pl.Mem.Mat, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major)
                                    key_scale_right: pl.Tile[[4, 128], pl.FP8E8M0, pl.Mem.RightScale] = pl.tile.move(key_scale_mat, target_memory=pl.Mem.RightScale)
                                    if second_token > first_token and first_request == second_request:
                                        query_pair: pl.Tile[[64, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.load(query_flat, [first_token * 32, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec)
                                        pair_scale_raw: pl.Tile[[2, 128], pl.UINT8] = pl.tile.reinterpret_view(pl.tile.load(query_scale, [first_token, 0], [2, 128], [2, 128]), dtype=pl.UINT8)
                                        pair_scale_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                                        pair_scale_zz: pl.Tile[[64, 4], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.reinterpret_view(pl.tile.tmov_x2zz(pl.tile.reshape(pair_scale_raw, [1, 256]), pair_scale_tmp, group_axis=1, dst_rows=64, dst_cols=4), dtype=pl.FP8E8M0)
                                        pair_dot: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_mx(query_pair, pair_scale_zz, keys_right, key_scale_right)
                                        pair_dot_v1: pl.Tile[[64, 128], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pair_dot, target_type=pl.BF16, mode='rint')
                                        for pair_query_offset in pl.unroll(2):
                                            pair_token: pl.Scalar[pl.INDEX] = first_token + pair_query_offset
                                            pair_negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                            pair_masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(pair_negative_bits, dtype=pl.FP32)
                                            pair_visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [pair_token]), pl.INDEX), width)
                                            if logical_begin < pair_visible:
                                                pair_valid_count: pl.Scalar[pl.INDEX] = pl.min(128, pair_visible - logical_begin)
                                                pair_head_dot: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.slice(pair_dot_v1, [32, 128], [pair_query_offset * 32, 0])
                                                pair_relu: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.maximums(pl.tile.cast(pair_head_dot, target_type=pl.FP32, mode='round'), 0.0)
                                                pair_head_weight: pl.Tile[[1, 32], pl.BF16] = pl.tile.load(weights, [pair_token, 0], [1, 32], [1, 32])
                                                pair_head_weight_v1: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pair_head_weight, target_type=pl.FP32, mode='round'), 0.015625)
                                                pair_rounded_weight: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pair_head_weight_v1, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                pair_weighted: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.row_expand_mul(pair_relu, pl.tile.reshape(pair_rounded_weight, [32, 1]))
                                                pair_weighted_v1: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pl.tile.cast(pair_weighted, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                pair_total: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.col_sum(pair_weighted_v1), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                if False:
                                                    pair_candidate: pl.Tile[[1, 128], pl.UINT8, pl.TileView(valid_shape=[1, pair_valid_count])] = pl.tile.load(candidates, [pair_token, logical_begin], [1, 128], [1, pair_valid_count])
                                                    pair_candidate_v1: pl.Tile[[1, 128], pl.INT8, pl.TileView(valid_shape=[1, pair_valid_count])] = pl.tile.reinterpret_view(pair_candidate, dtype=pl.INT8)
                                                    for pair_lane in pl.range(pair_valid_count):
                                                        if pl.cast(pl.tile.read(pair_candidate_v1, [0, pair_lane]), pl.INDEX) != 0:
                                                            pl.tile.write(pair_masked, [0, pair_lane], pl.tile.read(pair_total, [0, pair_lane]))
                                                else:
                                                    pair_valid_count_i32: pl.Scalar[pl.INT32] = pl.cast(pair_valid_count, pl.INT32)
                                                    pair_lanes_i32: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                                                    pair_predicate: pl.Tile[[1, 32], pl.UINT8, pl.Mem.Vec, pl.TileView(valid_shape=[1, 16])] = pl.tile.cmps(pair_lanes_i32, pair_valid_count_i32, cmp_type=2)
                                                    pair_masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.select(pair_predicate, pair_total, pair_masked)
                                            pl.tile.store(pl.tile.set_validshape(pair_masked, 1, output_count), [pair_token, logical_begin], output)
                                    else:
                                        for query_offset in pl.range(2):
                                            token: pl.Scalar[pl.INDEX] = first_token + query_offset
                                            if token < pl.cast(num_tokens, pl.INDEX):
                                                query_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token])
                                                if query_request == anchor_request:
                                                    negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                                    masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32)
                                                    visible: pl.Scalar[pl.INDEX] = pl.min(pl.cast(pl.tensor.read(lengths, [token]), pl.INDEX), width)
                                                    if logical_begin < visible:
                                                        valid_count: pl.Scalar[pl.INDEX] = pl.min(128, visible - logical_begin)
                                                        query_panel: pl.Tile[[32, 128], pl.FP8E4M3FN, pl.Mem.Vec] = pl.tile.load(query_flat, [token * 32, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec)
                                                        raw_query_scale: pl.Tile[[1, 128], pl.UINT8] = pl.tile.reinterpret_view(pl.tile.load(query_scale, [token, 0], [1, 128], [1, 128]), dtype=pl.UINT8)
                                                        query_scale_tmp: pl.Tile[[1, 96], pl.UINT8] = pl.tile.create([1, 96], dtype=pl.UINT8)
                                                        query_scale_zz: pl.Tile[[32, 4], pl.FP8E8M0, pl.Mem.Vec, pl.TileView(slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.reinterpret_view(pl.tile.tmov_x2zz(raw_query_scale, query_scale_tmp, group_axis=1, dst_rows=32, dst_cols=4), dtype=pl.FP8E8M0)
                                                        dot: pl.Tile[[32, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_mx(query_panel, query_scale_zz, keys_right, key_scale_right)
                                                        dot_v1: pl.Tile[[32, 128], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(dot, target_type=pl.BF16, mode='rint')
                                                        relu: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.maximums(pl.tile.cast(dot_v1, target_type=pl.FP32, mode='round'), 0.0)
                                                        head_weight: pl.Tile[[1, 32], pl.BF16] = pl.tile.load(weights, [token, 0], [1, 32], [1, 32])
                                                        head_weight_v1: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(head_weight, target_type=pl.FP32, mode='round'), 0.015625)
                                                        rounded_weight: pl.Tile[[1, 32], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(head_weight_v1, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                        weighted: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.row_expand_mul(relu, pl.tile.reshape(rounded_weight, [32, 1]))
                                                        weighted_v1: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.cast(pl.tile.cast(weighted, target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                        total: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.col_sum(weighted_v1), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                                                        if False:
                                                            candidate: pl.Tile[[1, 128], pl.UINT8, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(candidates, [token, logical_begin], [1, 128], [1, valid_count])
                                                            candidate_v1: pl.Tile[[1, 128], pl.INT8, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.reinterpret_view(candidate, dtype=pl.INT8)
                                                            for lane in pl.range(valid_count):
                                                                if pl.cast(pl.tile.read(candidate_v1, [0, lane]), pl.INDEX) != 0:
                                                                    pl.tile.write(masked, [0, lane], pl.tile.read(total, [0, lane]))
                                                        else:
                                                            valid_count_i32: pl.Scalar[pl.INT32] = pl.cast(valid_count, pl.INT32)
                                                            lanes_i32: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 128], dtype=pl.INT32, descending=False)
                                                            predicate: pl.Tile[[1, 32], pl.UINT8, pl.Mem.Vec, pl.TileView(valid_shape=[1, 16])] = pl.tile.cmps(lanes_i32, valid_count_i32, cmp_type=2)
                                                            masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.select(predicate, total, masked)
                                                    pl.tile.store(pl.tile.set_validshape(masked, 1, output_count), [token, logical_begin], output)
                                else:
                                    for query_offset_1 in pl.range(2):
                                        token: pl.Scalar[pl.INDEX] = first_token + query_offset_1
                                        if token < pl.cast(num_tokens, pl.INDEX):
                                            query_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token])
                                            if query_request == anchor_request:
                                                negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                                masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32)
                                                pl.tile.store(pl.tile.set_validshape(masked, 1, output_count), [token, logical_begin], output)
                            else:
                                for query_offset_2 in pl.range(2):
                                    token: pl.Scalar[pl.INDEX] = first_token + query_offset_2
                                    if token < pl.cast(num_tokens, pl.INDEX):
                                        query_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token])
                                        if query_request == anchor_request:
                                            negative_bits: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.INT32, value=pl.const(-8388608, pl.INT32))
                                            masked: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reinterpret_view(negative_bits, dtype=pl.FP32)
                                            pl.tile.store(pl.tile.set_validshape(masked, 1, output_count), [token, logical_begin], output)
        return score_tid
    @pl.function(type=pl.FunctionType.Inline)
    def score__4(self, query: pl.Tensor[[V41_T_DYN, 4096], pl.FP8E4M3FN], query_scale: pl.Tensor[[V41_T_DYN, 128], pl.FP8E8M0], weights: pl.Tensor[[V41_T_DYN, 32], pl.BF16], index_cache: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 64], pl.UINT8], index_cache_scale: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 4], pl.FP8E8M0], block_table: pl.Tensor[[V41_B_DYN, V41_TABLE_DYN], pl.INT32], request_ids: pl.Tensor[[V41_T_DYN], pl.INT32], lengths: pl.Tensor[[V41_T_DYN], pl.INT32], candidates: pl.Tensor[[V41_T_DYN, V41_CMP_POSITIONS_DYN], pl.UINT8], output: pl.Tensor[[8, 1024], pl.FP32], num_tokens: pl.Scalar[pl.INT32], cache_ready: pl.Scalar[pl.TASK_ID], query_ready: pl.Scalar[pl.TASK_ID], weights_ready: pl.Scalar[pl.TASK_ID]):
        width: pl.Scalar[pl.INDEX] = pl.tensor.dim(output, 1)
        panels: pl.Scalar[pl.INDEX] = (width + 128 - 1) // 128
        paired_panels: pl.Scalar[pl.INDEX] = (pl.cast(num_tokens, pl.INDEX) + 1) // 2 * panels
        complete_pairs: pl.Scalar[pl.INDEX] = 0
        if width >= 4096:
            if pl.cast(num_tokens, pl.INDEX) % 2 == 0:
                complete_pairs: pl.Scalar[pl.INDEX] = 1
                for token in pl.range(0, num_tokens, 2):
                    first_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token])
                    second_request: pl.Scalar[pl.INT32] = pl.tensor.read(request_ids, [token + 1])
                    if first_request != second_request:
                        complete_pairs: pl.Scalar[pl.INDEX] = 0
        if complete_pairs != 0:
            result: pl.Scalar[pl.TASK_ID] = self.score(query, query_scale, weights, index_cache, index_cache_scale, block_table, request_ids, lengths, candidates, output, num_tokens, cache_ready, query_ready, weights_ready)
        else:
            if paired_panels <= 16:
                result: pl.Scalar[pl.TASK_ID] = self.score__2(query, query_scale, weights, index_cache, index_cache_scale, block_table, request_ids, lengths, candidates, output, num_tokens, cache_ready, query_ready, weights_ready)
            else:
                result: pl.Scalar[pl.TASK_ID] = self.score__3(query, query_scale, weights, index_cache, index_cache_scale, block_table, request_ids, lengths, candidates, output, num_tokens, cache_ready, query_ready, weights_ready)
        return result
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(self, query: pl.Tensor[[V41_T_DYN, 4096], pl.FP8E4M3FN], query_scale: pl.Tensor[[V41_T_DYN, 128], pl.FP8E8M0], weights: pl.Tensor[[V41_T_DYN, 32], pl.BF16], index_cache: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 64], pl.UINT8], index_cache_scale: pl.Tensor[[V41_INDEX_BLOCKS_DYN, 128, 1, 4], pl.FP8E8M0], block_table: pl.Tensor[[V41_B_DYN, V41_TABLE_DYN], pl.INT32], request_ids: pl.Tensor[[V41_T_DYN], pl.INT32], lengths: pl.Tensor[[V41_T_DYN], pl.INT32], candidates: pl.Tensor[[V41_T_DYN, V41_CMP_POSITIONS_DYN], pl.UINT8], output: pl.Out[pl.Tensor[[8, 1024], pl.FP32]], num_tokens: pl.Scalar[pl.INT32], cache_ready: pl.Scalar[pl.TASK_ID], query_ready: pl.Scalar[pl.TASK_ID], weights_ready: pl.Scalar[pl.TASK_ID]) -> pl.Tensor[[8, 1024], pl.FP32]:
        self.score__4(query, query_scale, weights, index_cache, index_cache_scale, block_table, request_ids, lengths, candidates, output, num_tokens, cache_ready, query_ready, weights_ready)
        return output