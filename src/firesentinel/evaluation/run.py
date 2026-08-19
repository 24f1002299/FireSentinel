"""Validate an evaluation JSONL file; metrics arrive with the first model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from firesentinel.config import load_settings
from firesentinel.logging import configure_logging

LOGGER = logging.getLogger("firesentinel.evaluation.run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL evaluation input")
    arguments = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings)
    if arguments.input is None:
        LOGGER.info("no evaluation input supplied; nothing to evaluate")
        return 0
    count = sum(
        1
        for line in arguments.input.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line) is not None
    )
    LOGGER.info("validated %s evaluation records", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
