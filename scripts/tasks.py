"""Portable task runner for the FireSentinel repository."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"
    )
    return environment


def _run(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, env=_environment(), check=False).returncode


def format_code(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "ruff", "format", "src", "tests", "scripts"])


def check_format(_: argparse.Namespace) -> int:
    return _run(
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"]
    )


def lint(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"])


def typecheck(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "mypy", "src", "tests", "scripts"])


def test(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "pytest"])


def download(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "firesentinel.data.download"])


def cache_inspect(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "firesentinel.data.download", "inspect"])


def cache_clean_case(arguments: argparse.Namespace) -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "firesentinel.data.download",
            "clean-case",
            "--case-id",
            arguments.case_id,
        ]
    )


def replay(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "firesentinel.agent.replay"])


def evaluate(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "firesentinel.evaluation.run"])


def firms_ingest(arguments: argparse.Namespace) -> int:
    command = [sys.executable, "-m", "firesentinel.evaluation.firms"]
    for source in arguments.source:
        command.extend(["--source", str(source)])
    if arguments.output_dir is not None:
        command.extend(["--output-dir", str(arguments.output_dir)])
    if arguments.maximum_distance_km is not None:
        command.extend(["--maximum-distance-km", str(arguments.maximum_distance_km)])
    if arguments.maximum_time_gap_minutes is not None:
        command.extend(
            ["--maximum-time-gap-minutes", str(arguments.maximum_time_gap_minutes)]
        )
    if arguments.overwrite:
        command.append("--overwrite")
    return _run(command)


def benchmark_build(arguments: argparse.Namespace) -> int:
    command = [sys.executable, "-m", "firesentinel.evaluation.benchmark"]
    for option in (
        "firms_labels",
        "observation_inventory",
        "output_dir",
        "cases_per_class",
        "random_seed",
    ):
        value = getattr(arguments, option)
        if value is not None:
            command.extend([f"--{option.replace('_', '-')}", str(value)])
    if arguments.overwrite:
        command.append("--overwrite")
    return _run(command)


def slice_replay(_: argparse.Namespace) -> int:
    return _run([sys.executable, "-m", "firesentinel.vision.real_event", "--verify"])


def ui(_: argparse.Namespace) -> int:
    return _run(
        [sys.executable, "-m", "streamlit", "run", "src/firesentinel/ui/app.py"]
    )


def clean(arguments: argparse.Namespace) -> int:
    targets = [
        ROOT / name
        for name in (".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist")
    ]
    if arguments.artifacts:
        targets.append(ROOT / "artifacts")
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"removed {target.relative_to(ROOT)}")
    if arguments.artifacts:
        (ROOT / "artifacts").mkdir(exist_ok=True)
        (ROOT / "artifacts" / ".gitkeep").touch()
    return 0


Task = Callable[[argparse.Namespace], int]
TASKS: dict[str, tuple[str, Task]] = {
    "format": ("Format source files in place.", format_code),
    "format-check": ("Check formatting without modifying files.", check_format),
    "lint": ("Run Ruff lint checks.", lint),
    "typecheck": ("Run strict mypy checks.", typecheck),
    "test": ("Run the test suite.", test),
    "download": ("Download datasets declared in manifests/datasets.json.", download),
    "cache-inspect": ("Inspect verified local source-cache contents.", cache_inspect),
    "cache-clean-case": (
        "Remove verified-cache references for one case only.",
        cache_clean_case,
    ),
    "replay": ("Validate and replay a JSONL event stream.", replay),
    "evaluate": ("Validate an evaluation JSONL file.", evaluate),
    "firms-ingest": (
        "Ingest local FIRMS CSV files into evaluation-only references.",
        firms_ingest,
    ),
    "benchmark-build": (
        "Build matched positive/control evaluation cases from pinned inputs.",
        benchmark_build,
    ),
    "slice": (
        "Recreate the pinned real-event OpenCV evidence from verified cached sources.",
        slice_replay,
    ),
    "ui": ("Launch the Streamlit UI shell.", ui),
    "clean": ("Remove generated tool caches.", clean),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    for name, (help_text, function) in TASKS.items():
        subparser = subparsers.add_parser(name, help=help_text)
        if name == "clean":
            subparser.add_argument(
                "--artifacts", action="store_true", help="also remove local artifacts"
            )
        if name == "cache-clean-case":
            subparser.add_argument("--case-id", required=True, help="case to remove")
        if name == "firms-ingest":
            subparser.add_argument(
                "--source",
                type=Path,
                action="append",
                required=True,
                help="local FIRMS CSV export; repeat for multiple sources",
            )
            subparser.add_argument(
                "--output-dir",
                type=Path,
                help="directory under evaluation-data/ for generated references",
            )
            subparser.add_argument(
                "--maximum-distance-km",
                type=float,
                help="event clustering distance in km",
            )
            subparser.add_argument(
                "--maximum-time-gap-minutes",
                type=int,
                help="event clustering time gap in minutes",
            )
            subparser.add_argument(
                "--overwrite",
                action="store_true",
                help="replace changed generated references",
            )
        if name == "benchmark-build":
            subparser.add_argument(
                "--firms-labels",
                type=Path,
                help="FIRMS labels under evaluation-data/",
            )
            subparser.add_argument(
                "--observation-inventory",
                type=Path,
                help="pinned observation-window inventory under evaluation-data/",
            )
            subparser.add_argument(
                "--output-dir",
                type=Path,
                help="directory under evaluation-data/ for the benchmark",
            )
            subparser.add_argument(
                "--cases-per-class",
                type=int,
                help="positive and control count; minimum 60",
            )
            subparser.add_argument(
                "--random-seed",
                type=int,
                help="deterministic control sampling seed",
            )
            subparser.add_argument(
                "--overwrite",
                action="store_true",
                help="replace changed benchmark files",
            )
        subparser.set_defaults(function=function)
    arguments = parser.parse_args(argv)
    task_name = arguments.task
    if task_name not in TASKS:
        parser.error(f"unknown task: {task_name}")
    return TASKS[task_name][1](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
