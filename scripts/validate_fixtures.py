"""Validate every fixture END TO END through the real PyPTO compiler:
import (real parser) + ir.compile (real pass pipeline + codegen).
A fixture that the compiler rejects is not a fixture — it is a forgery.

Run from a pypto checkout's venv:

    .venv/bin/python scripts/validate_fixtures.py /workspace/pypto-symir
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/pypto-symir")
FIX = REPO / "crates" / "symir" / "tests" / "fixtures"

sys.path.insert(0, str(FIX))

import pypto.ir  # noqa: E402
from pypto.backend import BackendType  # noqa: E402

CLASSES = {
    "deps_tasks.py": "Pipeline",
    "parallel_carry.py": "ParCarry",
    "while_loop.py": "WhileProg",
    "range_carry_pre.py": "_jit_range_carry",
    "range_carry_post.py": "_jit_range_carry",
    "parallel_task_post.py": "_jit_parallel_task",
    "deepseek_indexer_pre.py": "_jit_entry",
    "manual_in_auto.py": "Probe0",
    "manual_via_inline.py": "Probe3",
}

failures = []
for fname, cls in CLASSES.items():
    mod = __import__(fname.removesuffix(".py"))
    program = getattr(mod, cls)
    try:
        pypto.ir.compile(program, skip_ptoas=True, backend_type=BackendType.Ascend950)
        print(f"{fname:32} REAL: parser + passes + codegen OK")
    except Exception as e:  # noqa: BLE001
        first = next((ln for ln in str(e).splitlines() if ln.strip()), "")
        print(f"{fname:32} REJECTED: {first[:120]}")
        failures.append(fname)

if failures:
    sys.exit(1)
