"""Command-line entry point: ``yara ingest``, ``yara ask``, ``yara decide``.

argparse rather than click or typer: three subcommands and a handful of flags
don't justify a runtime dependency, and a stdlib parser means ``pip install
yara-research`` gives a working CLI with nothing extra.

Stdout carries the result (markdown or JSON) and nothing else; progress and
warnings go to stderr through logging. That split is what lets
``yara ask "..." > brief.md`` or ``yara decide task.json | jq`` work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .agents.decision_agent import DecisionAgent
from .config import Settings

logger = logging.getLogger("yara.cli")

# Exit codes. 2 is argparse's own code for a usage error, so a runtime
# failure gets 1 and scripts can tell "you called it wrong" from "it failed".
EXIT_OK = 0
EXIT_FAILURE = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yara",
        description="Multi-agent research assistant with citation-verified briefs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="where the index and run history live (default: $YARA_DATA_DIR or .yara)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini", "openai", "echo"],
        help="LLM provider (default: $YARA_PROVIDER or echo)",
    )
    parser.add_argument("--model", help="model name for the chosen provider")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for progress, -vv for debug"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    ingest = sub.add_parser("ingest", help="add files or directories to the corpus")
    ingest.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="research a question and print the brief")
    ask.add_argument("question")
    ask.add_argument("--tools", action="store_true", help="let agents call tools while analysing")
    ask.add_argument("--json", action="store_true", help="print the brief as JSON, not markdown")
    ask.add_argument("-o", "--output", type=Path, help="write to a file instead of stdout")
    ask.set_defaults(func=cmd_ask)

    decide = sub.add_parser("decide", help="rank options from a JSON decision task")
    decide.add_argument(
        "task",
        nargs="?",
        default="-",
        help="JSON file with the task, or - for stdin (default). "
        "A bare list is treated as the options.",
    )
    decide.add_argument(
        "-c",
        "--criterion",
        action="append",
        default=[],
        metavar="FIELD=WEIGHT",
        help="override the task's criteria; repeatable. Negative weight = cost.",
    )
    decide.add_argument("--top", type=int, default=3, help="how many options to return")
    decide.set_defaults(func=cmd_decide)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"yara: no such file or directory: {exc.filename or exc}", file=sys.stderr)
    except ImportError as exc:
        # Optional extras (pypdf, sentence-transformers) raise with an install hint.
        print(f"yara: {exc}", file=sys.stderr)
    except ValueError as exc:  # includes json.JSONDecodeError
        print(f"yara: {exc}", file=sys.stderr)
    return EXIT_FAILURE


# ---------- commands ----------


def cmd_ingest(args: argparse.Namespace) -> int:
    from .app import YARA

    # Check every path before ingesting any, so a typo in the third argument
    # doesn't leave the first two half-committed to the index.
    for path in args.paths:
        if not path.exists():
            raise FileNotFoundError(2, "not found", str(path))

    with YARA(_settings(args)) as yara:
        for path in args.paths:
            added = yara.ingest_path(path)
            print(f"{path}: {added} new chunk(s)")
        print(f"corpus: {yara.corpus_size} chunk(s) in {yara.settings.index_path}")
    return EXIT_OK


def cmd_ask(args: argparse.Namespace) -> int:
    from .app import YARA

    with YARA(_settings(args)) as yara:
        if yara.corpus_size == 0:
            # The facade only warns here, which suits a library. From the
            # shell an empty corpus is always a mistake, and a brief with no
            # sources would look like an answer.
            print(
                "yara: the corpus is empty - run `yara ingest <path>` first "
                f"(data dir: {yara.settings.data_dir})",
                file=sys.stderr,
            )
            return EXIT_FAILURE
        brief = yara.research(args.question, use_tools=args.tools)

    text = brief.model_dump_json(indent=2) if args.json else brief.to_markdown()
    _emit(text, args.output)
    return EXIT_OK


def cmd_decide(args: argparse.Namespace) -> int:
    # No YARA instance: ranking is pure computation, so there's no reason to
    # open the index, the database, or create a data directory for it.
    task = _load_task(args.task)
    if args.criterion:
        task["criteria"] = _parse_criteria(args.criterion)
    if args.top < 1:
        raise ValueError("--top must be at least 1")

    result = DecisionAgent(top_n=args.top).execute(task)
    _emit(json.dumps(result, indent=2, default=str), None)
    # An empty ranking comes back with a message (bad weights, everything
    # filtered out). Still printed, but signalled so scripts notice.
    return EXIT_FAILURE if not result.get("recommendations") else EXIT_OK


# ---------- helpers ----------


def _settings(args: argparse.Namespace) -> Settings:
    """Flags override env/.env; unset flags leave pydantic-settings in charge."""
    overrides: dict[str, Any] = {
        "data_dir": args.data_dir,
        "provider": args.provider,
        "model": args.model,
    }
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def _load_task(source: str) -> dict[str, Any]:
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"task is not valid JSON: {exc}") from exc

    if isinstance(data, list):
        return {"type": "decision", "data": data}
    if not isinstance(data, dict):
        raise ValueError("task must be a JSON object or a list of options")
    # The subcommand already says this is a decision; don't make the file
    # repeat it, but respect an explicit type if one is there.
    data.setdefault("type", "decision")
    return data


def _parse_criteria(items: list[str]) -> dict[str, float]:
    criteria = {}
    for item in items:
        name, sep, weight = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"criterion must look like FIELD=WEIGHT, got {item!r}")
        try:
            criteria[name.strip()] = float(weight)
        except ValueError as exc:
            raise ValueError(f"weight for {name.strip()!r} is not a number: {weight!r}") from exc
    return criteria


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
