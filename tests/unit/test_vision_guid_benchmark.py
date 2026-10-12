# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Platform accounting and failure reporting for the standalone benchmark, not timing assertions."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "vision_guid_columns.py"


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location("vision_guid_benchmark", BENCHMARK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("platform_name", "expected"), [("linux", 7 * 1024), ("darwin", 7), ("win32", 7000)])
def test_high_water_memory_units(benchmark, monkeypatch, platform_name, expected):
    monkeypatch.setattr(benchmark.sys, "platform", platform_name)
    resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=7))
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda _: resource)
    monkeypatch.setattr(
        benchmark.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(peak_wset=7000))
    )
    assert benchmark.high_water_rss_bytes() == expected


def test_rss_uses_resident_memory_not_virtual_memory(benchmark, monkeypatch):
    monkeypatch.setattr(
        benchmark.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=123, vms=999))
    )
    assert benchmark.rss_bytes() == 123


def test_memory_limit_is_explicit_and_linux_only(benchmark, monkeypatch):
    resource = SimpleNamespace(RLIMIT_AS=1, setrlimit=MagicMock())
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda _: resource)
    for platform_name in ("win32", "darwin"):
        monkeypatch.setattr(benchmark.sys, "platform", platform_name)
        assert benchmark.set_memory_limit(0) == "none; explicitly disabled"
        with pytest.raises(ValueError, match="only used on Linux"):
            benchmark.set_memory_limit(4096)
    resource.setrlimit.assert_not_called()
    monkeypatch.setattr(benchmark.sys, "platform", "linux")
    assert benchmark.set_memory_limit(64) == "RLIMIT_AS virtual address space; not an RSS limit"
    resource.setrlimit.assert_called_once_with(1, (64 * benchmark.MIB, 64 * benchmark.MIB))


def test_native_windows_worktree_does_not_need_wslpath(benchmark, monkeypatch, tmp_path):
    (tmp_path / ".git").write_text("gitdir: C:/repo/.git/worktrees/example\n", encoding="utf-8")
    monkeypatch.setattr(benchmark.sys, "platform", "win32")
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "git.exe" if name == "git" else None)
    check_output = MagicMock(return_value="abc\n")
    monkeypatch.setattr(benchmark.subprocess, "check_output", check_output)
    assert benchmark.git_output(tmp_path, "rev-parse", "HEAD") == "abc"
    check_output.assert_called_once_with(["git.exe", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True)


@pytest.mark.parametrize("case", ["small-repeated", "small-unique"])
def test_worker_runs_on_the_native_platform(case):
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(BENCHMARK_PATH),
            "--worker",
            "--case",
            case,
            "--candidate-checkout",
            str(BENCHMARK_PATH.parent.parent),
            "--memory-limit-mib",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["case"] == case
    assert report["memory_limit_mib"] == 0
    assert report["memory_limit_kind"] == "none; explicitly disabled"
    assert report["conversion"]["sampled_peak_rss_bytes"] > 0
    assert report["whole_process_high_water_rss_bytes"] > 0
    assert report["validation"].endswith("passed")


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_failed_worker_is_checkpointed_and_stops_the_grid(benchmark, monkeypatch, tmp_path, failure):
    checkout = tmp_path / "checkout"
    source = checkout / "src/power_grid_model_io/data_stores/excel_file_store.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixture source\n", encoding="utf-8")
    output = tmp_path / "results.json"
    monkeypatch.setattr(benchmark, "git_output", lambda *_: "abc")
    # platform.platform() may also invoke subprocess.run on an uncached Windows host.
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "test-platform")
    # Keep failure assertions from displaying the real inherited process environment.
    monkeypatch.setattr(benchmark.os, "environ", {})
    run = MagicMock()
    if failure == "timeout":
        run.side_effect = subprocess.TimeoutExpired("worker", 1)
    else:
        run.return_value = subprocess.CompletedProcess("worker", 1, stdout="", stderr="worker failed")
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BENCHMARK_PATH),
            "--baseline-checkout",
            str(checkout),
            "--candidate-checkout",
            str(checkout),
            "--output",
            str(output),
            "--cases",
            "small-repeated",
            "--memory-limit-mib",
            "0",
        ],
    )
    assert benchmark.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["runs"]) == 1
    assert report["runs"][0]["exit_code"] == ("timeout" if failure == "timeout" else 1)
    assert report["runs"][0]["implementation"] == "baseline"
    run.assert_called_once()
