"""The CLI is tested through main(argv), in-process.

Subprocesses would test the console-script wiring too, but they're slow and
hide tracebacks. main() returning an exit code instead of calling sys.exit is
what makes this possible.
"""

import io
import json
from pathlib import Path

import pytest

from yara.cli import EXIT_FAILURE, EXIT_OK, build_parser, main

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus"

OPTIONS = [
    {"name": "cheap", "price": 10, "quality": 0.5},
    {"name": "pricey", "price": 100, "quality": 0.9},
]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    # A developer's .env or YARA_PROVIDER must not turn these into live calls.
    monkeypatch.chdir(tmp_path)
    for var in ("YARA_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path / "data")


def _run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


# ---------- ingest / ask ----------


def test_ingest_then_ask_across_invocations(capsys, data_dir):
    """The index persists between processes - the whole point of --data-dir."""
    code, out, _ = _run(capsys, "--data-dir", data_dir, "ingest", str(CORPUS))
    assert code == EXIT_OK
    assert "new chunk(s)" in out

    code, out, _ = _run(capsys, "--data-dir", data_dir, "ask", "What is hybrid retrieval?")
    assert code == EXIT_OK
    assert out.startswith("# What is hybrid retrieval?")
    assert "## Sources" in out


def test_reingest_adds_nothing(capsys, data_dir):
    _run(capsys, "--data-dir", data_dir, "ingest", str(CORPUS))
    code, out, _ = _run(capsys, "--data-dir", data_dir, "ingest", str(CORPUS))
    assert code == EXIT_OK
    assert ": 0 new chunk(s)" in out


def test_ingest_checks_every_path_before_writing(capsys, data_dir, tmp_path):
    code, _, err = _run(
        capsys, "--data-dir", data_dir, "ingest", str(CORPUS), str(tmp_path / "missing.md")
    )
    assert code == EXIT_FAILURE
    assert "missing.md" in err
    # The valid path was not ingested either.
    code, _, err = _run(capsys, "--data-dir", data_dir, "ask", "anything")
    assert code == EXIT_FAILURE
    assert "corpus is empty" in err


def test_ask_on_empty_corpus_fails_with_a_hint(capsys, data_dir):
    code, out, err = _run(capsys, "--data-dir", data_dir, "ask", "anything")
    assert code == EXIT_FAILURE
    assert out == ""
    assert "yara ingest" in err


def test_ask_json_and_output_file(capsys, data_dir, tmp_path):
    _run(capsys, "--data-dir", data_dir, "ingest", str(CORPUS))
    target = tmp_path / "out" / "brief.json"
    code, out, err = _run(
        capsys, "--data-dir", data_dir, "ask", "What is BM25?", "--json", "-o", str(target)
    )
    assert code == EXIT_OK
    assert out == ""  # result went to the file, not stdout
    assert str(target) in err
    brief = json.loads(target.read_text(encoding="utf-8"))
    assert brief["question"] == "What is BM25?"
    assert brief["run_id"].startswith("run_")


# ---------- decide ----------


def test_decide_from_file(capsys, tmp_path):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"data": OPTIONS, "criteria": {"quality": 1.0}}))
    code, out, _ = _run(capsys, "decide", str(task))
    assert code == EXIT_OK
    result = json.loads(out)
    assert result["recommendations"][0]["option"]["name"] == "pricey"


def test_decide_from_stdin_bare_list_with_criterion_flags(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(OPTIONS)))
    code, out, _ = _run(capsys, "decide", "-c", "price=-1", "--top", "1")
    assert code == EXIT_OK
    recs = json.loads(out)["recommendations"]
    assert [r["option"]["name"] for r in recs] == ["cheap"]


def test_decide_does_not_create_a_data_dir(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(OPTIONS)))
    _run(capsys, "decide")
    assert not (tmp_path / ".yara").exists()


@pytest.mark.parametrize(
    "stdin, flags, message",
    [
        ("not json", [], "not valid JSON"),
        ('"a string"', [], "JSON object or a list"),
        (json.dumps(OPTIONS), ["-c", "price"], "FIELD=WEIGHT"),
        (json.dumps(OPTIONS), ["-c", "price=cheap"], "not a number"),
        (json.dumps(OPTIONS), ["--top", "0"], "--top"),
    ],
)
def test_decide_rejects_bad_input(capsys, monkeypatch, stdin, flags, message):
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code, out, err = _run(capsys, "decide", *flags)
    assert code == EXIT_FAILURE
    assert message in err


def test_decide_signals_an_empty_ranking(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("[]"))
    code, out, _ = _run(capsys, "decide")
    assert code == EXIT_FAILURE
    assert json.loads(out)["recommendations"] == []


# ---------- parser ----------


def test_subcommand_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2  # argparse usage error, distinct from EXIT_FAILURE


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("yara ")
