# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Synthetic GUID conversion stress test; each run uses a fresh process.

Example:
    uv run --group benchmark python benchmarks/vision_guid_columns.py --baseline-checkout ../base --output results.json
Both checkouts use the same interpreter and dependencies. Input generation and conversion are measured separately.
No Excel I/O is included; real sheet loading is covered by the regression tests.

RSS sampling uses psutil on Linux, Windows and macOS. The default 4096 MiB virtual-address-space limit uses
Linux RLIMIT_AS, not an RSS cap. On Windows and macOS, pass --memory-limit-mib 0 to explicitly run without
a hard memory limit, and choose bounded --cases. The timeout still applies on every platform.
"""

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

CASES = {
    "small-repeated": (1_000, 16, 100),
    "small-unique": (1_000, 16, 1_000),
    "medium-repeated": (50_000, 105, 1_024),
    "medium-unique": (50_000, 105, 50_000),
    "large-repeated": (500_000, 105, 1_024),
    "large-unique": (100_000, 105, 100_000),
    "wide-repeated": (50_000, 210, 1_024),
}
MIB = 1024 * 1024
NUMERIC_COLUMNS = 20
CONTROLLED_INPUT_COLUMNS = 125
MATRIX_ROWS = (1_000, 20_000, 100_000)
MATRIX_GUID_COLUMNS = (1, 8, 32, 105)
MATRIX_CASES = {
    f"grid-r{rows}-g{guid_columns}-{distribution}": (
        rows,
        guid_columns,
        min(1_024, max(100, rows // 10)) if distribution == "repeated" else rows,
    )
    for rows in MATRIX_ROWS
    for guid_columns in MATRIX_GUID_COLUMNS
    for distribution in ("repeated", "unique")
}
ALL_CASES = {**CASES, **MATRIX_CASES}


def high_water_rss_bytes() -> int:
    if sys.platform == "win32":
        return int(psutil.Process().memory_info().peak_wset)
    resource = importlib.import_module("resource")
    unit = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * unit


def rss_bytes() -> int:
    """Current process resident memory, including native Windows working set."""
    return int(psutil.Process().memory_info().rss)


def set_memory_limit(memory_limit_mib: int) -> str:
    if memory_limit_mib == 0:
        return "none; explicitly disabled"
    if sys.platform != "linux":
        raise ValueError("RLIMIT_AS is only used on Linux; pass --memory-limit-mib 0 on this platform")
    resource = importlib.import_module("resource")
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_mib * MIB, memory_limit_mib * MIB))
    return "RLIMIT_AS virtual address space; not an RSS limit"


def measure(operation: Callable[[], Any]) -> tuple[Any, dict[str, float | int]]:
    start_rss = rss_bytes()
    samples = [start_rss]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.01):
            samples.append(rss_bytes())

    sampler = threading.Thread(target=sample)
    sampler.start()
    start = time.perf_counter()
    try:
        result = operation()
    finally:
        elapsed = time.perf_counter() - start
        samples.append(rss_bytes())
        stop.set()
        sampler.join()
    return result, {
        "seconds": elapsed,
        "start_rss_bytes": start_rss,
        "end_rss_bytes": samples[-1],
        "sampled_peak_rss_bytes": max(samples),
        "process_high_water_rss_bytes": high_water_rss_bytes(),
        "sampling_interval_seconds": 0.01,
    }


def worker(args: argparse.Namespace) -> None:
    memory_limit_kind = set_memory_limit(args.memory_limit_mib)
    sys.path.insert(0, str(args.candidate_checkout / "src"))
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    store_module = importlib.import_module("power_grid_model_io.data_stores.excel_file_store")
    pd.options.mode.copy_on_write = args.copy_on_write == "on"
    rows, guid_columns, unique_values = ALL_CASES[args.case]
    numeric_columns = CONTROLLED_INPUT_COLUMNS - guid_columns if args.case in MATRIX_CASES else NUMERIC_COLUMNS
    row_ids = np.arange(rows) % unique_values

    def guid_values(column: int):
        pool = np.array([f"{column:08x}-0000-4000-8000-{value:012x}" for value in range(unique_values)], dtype=object)
        return pool[row_ids]

    def generate():
        columns = {f"Field{i}GUID": guid_values(i) for i in range(guid_columns)}
        columns.update({f"Numeric{i}": np.arange(rows, dtype=np.int64) + i for i in range(numeric_columns)})
        return pd.DataFrame(columns, copy=False)

    data, generation = measure(generate)
    store = store_module.ExcelFileStore()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        result, conversion = measure(lambda: store._process_uuid_columns(data=data, sheet_name="Other"))  # noqa: SLF001

    # Expected numbers come from deterministic first-encounter IDs, not the implementation under test.
    for i in range(guid_columns):
        if not np.array_equal(result[f"Field{i}GUID"].to_numpy(), guid_values(i)):
            raise AssertionError(f"GUID values changed in column {i}")
        if not np.array_equal(result[f"Field{i}Number"].to_numpy(), i * unique_values + row_ids):
            raise AssertionError(f"Wrong derived numbers in column {i}")
    for i in range(numeric_columns):
        if not np.array_equal(result[f"Numeric{i}"].to_numpy(), np.arange(rows) + i):
            raise AssertionError(f"Numeric input changed in column {i}")
    if store._uuid_cvtr.get_size() != guid_columns * unique_values:  # noqa: SLF001
        raise AssertionError("Unexpected converter size")
    if result.shape != (rows, 2 * guid_columns + numeric_columns) or not result.columns.is_unique:
        raise AssertionError("Unexpected output shape or duplicate labels")

    print(
        json.dumps(
            {
                "case": args.case,
                "rows": rows,
                "guid_columns": guid_columns,
                "numeric_columns": numeric_columns,
                "unique_guid_ratio": unique_values / rows,
                "identifier_length": 36,
                "input_shape": [rows, guid_columns + numeric_columns],
                "output_shape": list(result.shape),
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "psutil": psutil.__version__,
                "rss_kind": "process working set" if sys.platform == "win32" else "process resident set",
                "copy_on_write": pd.options.mode.copy_on_write,
                "implementation_file": store_module.__file__,
                "generation": generation,
                "conversion": conversion,
                "performance_warning_count": sum(isinstance(w.message, pd.errors.PerformanceWarning) for w in captured),
                "validation": "all GUID values, derived numbers, numeric columns, shape and converter size passed",
                "memory_limit_mib": args.memory_limit_mib,
                "memory_limit_kind": memory_limit_kind,
                "whole_process_high_water_rss_bytes": high_water_rss_bytes(),
            }
        ),
        flush=True,
    )


def git_output(checkout: Path, *arguments: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("git is required to record source revisions")
    command = [executable, "-C", str(checkout)]
    marker = checkout / ".git"
    if marker.is_file() and sys.platform != "win32":
        # A worktree created by Windows Git has a Windows path that Linux Git cannot resolve directly.
        git_dir = marker.read_text().removeprefix("gitdir:").strip()
        if len(git_dir) > 1 and git_dir[1] == ":":
            mapper = shutil.which("wslpath")
            if mapper is None:
                raise RuntimeError("Windows worktree paths need wslpath; use a native Linux checkout otherwise")
            git_dir = subprocess.check_output([mapper, "-u", git_dir], text=True).strip()  # noqa: S603
        command = [executable, "--git-dir", str((checkout / git_dir).resolve()), "--work-tree", str(checkout)]
    return subprocess.check_output([*command, *arguments], text=True).strip()  # noqa: S603


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkout", type=Path)
    parser.add_argument("--candidate-checkout", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", choices=ALL_CASES, nargs="+")
    parser.add_argument("--matrix", action="store_true", help="Use the fixed-width rows by GUID-columns grid")
    parser.add_argument("--case", choices=ALL_CASES)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--memory-limit-mib", type=int, default=4096, help="Linux virtual address space cap; 0 explicitly disables it"
    )
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--copy-on-write", choices=["off", "on"], default="off")
    args = parser.parse_args()
    if args.memory_limit_mib < 0 or args.repeats < 1 or args.timeout_seconds < 1:
        parser.error("repeats and timeout must be positive; memory limit must be nonnegative")
    if args.memory_limit_mib and sys.platform != "linux":
        parser.error("RLIMIT_AS is only used on Linux; pass --memory-limit-mib 0 and select bounded --cases")
    if args.worker:
        if args.case is None:
            parser.error("--case is required for a worker")
        worker(args)
        return 0
    if args.baseline_checkout is None or args.output is None:
        parser.error("--baseline-checkout and --output are required")
    if args.matrix and args.cases is not None:
        parser.error("use either --matrix or --cases")
    cases = args.cases if args.cases is not None else list(MATRIX_CASES if args.matrix else CASES)

    runs: list[dict[str, Any]] = []
    report = {
        "platform": platform.platform(),
        "baseline_sha": git_output(args.baseline_checkout, "rev-parse", "HEAD"),
        "candidate_sha": git_output(args.candidate_checkout, "rev-parse", "HEAD"),
        "baseline_dirty": git_output(args.baseline_checkout, "status", "--porcelain").splitlines(),
        "candidate_dirty": git_output(args.candidate_checkout, "status", "--porcelain").splitlines(),
        "baseline_source_sha256": hashlib.sha256(
            (args.baseline_checkout / "src/power_grid_model_io/data_stores/excel_file_store.py").read_bytes()
        ).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(
            (args.candidate_checkout / "src/power_grid_model_io/data_stores/excel_file_store.py").read_bytes()
        ).hexdigest(),
        "cases": cases,
        "matrix_design": {
            "fixed_input_columns": CONTROLLED_INPUT_COLUMNS,
            "rows": MATRIX_ROWS,
            "guid_columns": MATRIX_GUID_COLUMNS,
            "distributions": ["repeated", "unique"],
            "case_count": len(MATRIX_CASES),
            "case_naming": "grid-r<rows>-g<guid-columns>-<distribution>",
            "trial_order": "baseline then candidate on odd repeats, candidate then baseline on even repeats",
        }
        if args.matrix
        else None,
        "runs": runs,
    }
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for case in cases:
        for repeat in range(args.repeats):
            implementations = [("baseline", args.baseline_checkout), ("candidate", args.candidate_checkout)]
            if repeat % 2:
                implementations.reverse()
            for label, checkout in implementations:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--candidate-checkout",
                    str(checkout),
                    "--memory-limit-mib",
                    str(args.memory_limit_mib),
                    "--copy-on-write",
                    args.copy_on_write,
                ]
                try:
                    completed = subprocess.run(  # noqa: S603
                        command, env=env, capture_output=True, text=True, timeout=args.timeout_seconds, check=False
                    )
                    run = json.loads(completed.stdout) if completed.returncode == 0 else {"stderr": completed.stderr}
                    run["exit_code"] = completed.returncode
                except subprocess.TimeoutExpired:
                    run = {"exit_code": "timeout"}
                run.update(
                    {
                        "case": case,
                        "implementation": label,
                        "repeat": repeat + 1,
                        "timeout_seconds": args.timeout_seconds,
                    }
                )
                runs.append(run)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"{case} {label} {repeat + 1}: {run['exit_code']}", flush=True)
                if run["exit_code"] != 0:
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
