"""Replay a JSONL event stream; the domain policy is added in later milestones."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from firesentinel.agent.label_boundary import runtime_input_path
from firesentinel.config import load_settings
from firesentinel.logging import configure_logging

LOGGER = logging.getLogger("firesentinel.agent.replay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL event file to replay")
    arguments = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings)
    if arguments.input is None:
        LOGGER.info("no replay input supplied; nothing to replay")
        return 0
    input_path = runtime_input_path(arguments.input, project_root=settings.root_dir)
    count = 0
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
            count += 1
    LOGGER.info("validated %s replay events", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
