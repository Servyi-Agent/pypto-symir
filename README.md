# symir

A symbolic interpreter over the printed [PyPTO](https://github.com/hw-native-sys/pypto)
IR, extracting a **DAG of symbolic tasks**.

It parses the DSL's printed form — the `.as_python()` output of
`specialize()` / `lower()` / the per-pass `passes_dump/` snapshots — and
interprets one function symbolically to produce the task graph the code
establishes.

## The encoding

* **Loops introduce an uninterpreted bounded variable** (`mu0`, `mu1`, ...):
  the induction variable is symbolic, and the bounds are symbolic terms
  (runtime scalars, tensor reads, arithmetic over other variables).
* **Sequential loops** (`pl.range`, `pl.unroll`, `pl.pipeline`, `pl.while_`)
  produce **one node** carrying `muN` with a `Bounds` constraint
  `lo <= muN < hi`.
* **Parallel loops** (`pl.parallel`) produce **two nodes** — the body
  evaluated at `muNa` and `muNb` — with a `Distinct` constraint
  `muNa != muNb`, expressing that two arbitrary iterations may be observed
  unordered.
* **All edges the explicit code establishes** appear in the DAG:
  * `Dep` — explicit `deps=[...]` on submits and spmd scopes (direct,
    even across loop boundaries);
  * `Raw` / `Waw` / `War` — region-level dataflow derived from declared
    argument directions and op effects; cross-boundary edges run through
    the loop summary nodes (the classic hierarchical task graph).
* A region **both read and written** inside a parallel loop is a carried
  dependency — since `muNa != muNb` are unordered, that is reported as a
  **race warning**.

## Install and run

```bash
cargo build --release
./target/release/symir kernel_dump.py --emit report   # constraints + warnings
./target/release/symir kernel_dump.py --emit dot -o dag.dot
./target/release/symir kernel_dump.py --emit json
```

Producing input from a PyPTO checkout:

```python
program = kernel.specialize(*samples)          # pre-pass IR
print(program.as_python())                     # -> feed this to symir
```

## Example

On the real DeepSeek-V4.1 decode indexer dump (428 lines, depth-14 guard
cascade, `deepseek_v4_1_flash/decode_indexer.py` specialized for width
1024):

```
program: _jit_entry  entry: entry
nodes: 29  edges: 23
constraint: Bounds { loop_id: 1, var: "mu1",
    lo: "t29=get_block_idx()",
    hi: "t28=*(t27=//(t26=cast(param.num_tokens(), pl.INDEX), 2), ...)" }
...
n4 [label="mu2: [0, 2)\n  ...\n  iter: mu1\n  when: false && ..."]
```

The worker-strided SPMD loop shows `lo = get_block_idx()` — the uninterpreted
core index — and the guards carry the folded branch conditions (a statically
dead arm shows as `when: ... && false`).

## Semantics

| Construct | Handling |
|---|---|
| op calls | uninterpreted applications `tN=op(args...)`; concrete scalar algebra folds |
| tensors / buffers / allocations | region identities; effects default to reads (the C++ `ArgEffect::Read` default), with a writer table for `assemble`/`store`/allocs |
| task submits (`pl.submit`, `pl.spmd_submit`, kernel calls) | one `Task` node; reads/writes from the callee signature's `pl.In`/`pl.Out`/`pl.InOut` directions or the explicit `arg_directions` attr |
| `Inline` / `Opaque` function calls | spliced into the caller's context (no task node), regions flowing through |
| `with pl.spmd(...)` | a `Task` node summarizing the body; `deps=` produce direct edges |
| `if` / `elif` / `else` | both arms explored; guards recorded on nodes; writers/readers merged (phi) |
| module-level `pl.dynamic("X")` | symbolic dimension constant `X` |

### Limitations (documented, by design)

* Effects are **region-granular**: no intra-tensor offsets or slicing
  precision — a RAW edge means "same region", not "same bytes".
* `Scalar[TASK_ID]` *parameters* stay symbolic terms; only task ids produced
  inside the interpreted function yield concrete `Dep` edges.
* Uninterpreted values flowing through SSA renames are terms, not regions;
  dependencies through pure value flow (no region write) are not edges.

## Lint policy

This project runs the [servyi/lints](https://github.com/servyi/lints)
policy in CI: clippy under the shared flag list plus the `servyi-lint`
driver (`unsafe_scope`, `workspace_lints_table`, `non_test_panic_allow`,
`unsound_constructor`, `custom_parser`, `inline_literal`).

Compliance notes:

* **No hand-rolled parsing** — the grammar is [pest](https://pest.rs), an
  off-the-shelf PEG generator; block structure is recovered by comparing
  indent widths (arithmetic only). The `custom_parser`-banned string
  primitives are not used anywhere.
* **Literals live in named consts** (`consts.rs`, `msgs.rs`), including
  operator tokens and error prose; the few sanctioned inline positions
  (message macros, trivial numerics) are used as the policy allows.
* **No lint tables in manifests** — the flag list the CI runs is the single
  source of truth.
* No `unsafe`, no `unwrap`/`panic` outside the sanctioned test exemptions.

## License

MIT OR Apache-2.0.
