# symir — agent instructions

## The regression rule (read this first)

**Every strange behavior gets all three, in order:**

1. **A runtime assertion** that detects the class of the problem, not just
   the instance — for graph output that means structural invariants in
   `TaskDag::validate()` (edge uniqueness, dangling endpoints, acyclicity),
   run by `extract()` on every interpretation; add the new invariant there
   when the class is new.
2. **A reproducing test committed red** — write the test, demonstrate it
   failing against the pre-fix code (a `git worktree` at the parent commit
   is the standard way; record the failing output in the commit message),
   and only then land the fix that turns it green. A fix without a
   red-demonstrated test is not done.
3. **The fix**, with the commit message naming what was observed and what
   the red demonstration showed.

Observations made while exploring (weird parse positions, surprising
printer forms, effect-tracking gaps) follow the same rule at the *parser*
level: smallest input that reproduces, committed as a fixture or unit
test, red before the grammar/interpreter change.

## Branch and PR workflow (enforced on GitHub)

`main` is protected: direct pushes are rejected for everyone, admins
included, and force-pushes/deletions are off. Every change lands through
a pull request. There is no review requirement — a PR is merged
immediately once it has no conflicts, using **rebase merge** so history
stays linear.

The local shape of a change:

```sh
git checkout -b <branch>
# ... commits ...
git push <remote> <branch>
# open the PR (gh pr create --base main --head <branch>, or the REST API),
# then rebase-merge it as soon as it is conflict-free (gh pr merge
# --rebase, or PUT /repos/Servyi-Agent/pypto-symir/pulls/<n>/merge with
# {"merge_method": "rebase"}).
```

## Lint policy

[Servyi lints](https://github.com/servyi/lints) run in CI (see
`.github/workflows/ci.yml`). The local equivalents:

```sh
cargo clippy --workspace --all-targets -- -D warnings -W rust_2018_idioms \
  -D unused_results -D unused_must_use -D non_ascii_idents -D clippy::all \
  -D clippy::unwrap_used -D clippy::panic

RUSTC_WRAPPER=<servyi-lint binary> \
RUSTFLAGS="-D unsafe_scope -D workspace_lints_table -D non_test_panic_allow \
 -D unsound_constructor -W custom_parser -W inline_literal" \
cargo +nightly-2026-09-21 check --workspace --all-targets
```

Rules that shape this codebase:

- No hand-rolled parsing primitives (pest only); block structure by
  indent-width arithmetic.
- Literals live in named consts (`consts.rs`, `msgs.rs`) — including
  operator tokens and message prose. A const may not name its own value.
- No lint tables in manifests; the CI flag list is the single source of
  truth.
- No `unwrap`/`panic` outside tests; `expect` only with an argued reason.
- No `unsafe`.

## Graphs

DAGs are built with [petgraph](https://crates.io/crates/petgraph) and
drawn by libraries: `petgraph::dot::Dot` generates the layout
description, graphviz renders (`--emit svg|png`, needs the `dot` binary).
Do not hand-draw or hand-format graph output — when the output looks
wrong, that is a bug in the mapping, and it follows the regression rule.
