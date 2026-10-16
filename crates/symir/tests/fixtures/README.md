# Fixture provenance

**Every fixture is a dump produced by the real PyPTO compiler.** No
handwritten IR. Each one round-trips: it re-parses and fully recompiles
(parser + pass pipeline + codegen) — verified by `scripts/validate_fixtures.py`
(run from a pypto checkout's venv), which fails if any fixture is rejected.

Regenerate with `scripts/gen_real_fixtures.py` (also from a pypto venv).

| Fixture | Origin |
|---|---|
| `deps_tasks.py` | `00_frontend.py` dump of `Pipeline`: two InCore kernels, `pl.submit` with `deps=` |
| `parallel_carry.py` | `00_frontend.py` dump of `ParCarry`: `pl.parallel` loop carrying an `InOut` buffer |
| `while_loop.py` | `00_frontend.py` dump of `WhileProg`: natural `while i < n:` over tensors |
| `range_carry_pre.py` / `range_carry_post.py` | `specialize()` / `lower()` output of the `range_carry` kernel (which fully compiles) |
| `parallel_task_post.py` | `lower()` output of the `parallel_task` kernel |
| `slicing.py` / `slicing_post.py` | pre-pass and pass-59 dumps of the `slicing` kernel (disjoint partial writes through `tensor.slice`/`assemble`/`tile.store`). The post-pass dump is real compiler output whose *re-import* PyPTO itself rejects (tile auto-names use the reserved `__` delimiter) — documented in the validator |
| `deepseek_indexer_pre.py` | pre-pass dump of the real DeepSeek-V4.1 decode indexer (`pypto-lib`, width-1024 specialization) |

The parser unit tests in `tests/parse_unit.rs` use small inline snippets;
those are parser-level inputs (some intentionally invalid) and are not
presented as compiler output anywhere.
