"""The container and CI contracts, checked without Docker or a runner.

CI builds the image and curls ``/health``, which is the real proof. These
tests cover the failure it cannot catch quickly: the *agreement* between
files that nothing imports from each other. The data directory is named in
the Dockerfile, mounted by Compose and read by ``Settings``; the supported
interpreters are listed in ``pyproject.toml`` and repeated in the CI matrix.
Each pair drifts silently, and the symptom shows up much later as an empty
corpus after a restart or a version the badge never actually tested.

Text matching rather than a YAML parser, deliberately: PyYAML is not a
dependency of this project and adding one so the tests can read a file that
GitHub parses anyway is a poor trade.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from yara.config import Settings

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"


def _dockerfile() -> str:
    """The Dockerfile with line continuations folded, so ENV blocks scan as one line."""
    return re.sub(r"\\\s*\n\s*", " ", DOCKERFILE.read_text(encoding="utf-8"))


def _directive(name: str) -> str:
    match = re.search(rf"^{name}\s+(.+)$", _dockerfile(), re.MULTILINE)
    assert match, f"Dockerfile has no {name} instruction"
    return match.group(1).strip()


@pytest.fixture(scope="module")
def data_dir() -> str:
    match = re.search(r"YARA_DATA_DIR=(\S+)", _dockerfile())
    assert match, "Dockerfile must set YARA_DATA_DIR so the image has a writable data path"
    return match.group(1)


# ---------- the image ----------


def test_image_runs_as_a_non_root_user():
    # The last USER wins; anything after it runs as that user.
    users = re.findall(r"^USER\s+(\S+)", _dockerfile(), re.MULTILINE)
    assert users, "Dockerfile never drops privileges"
    assert users[-1] not in {"root", "0"}


def test_image_serves_the_api_on_all_interfaces():
    cmd = _directive("CMD")
    assert "yara.api:app" in cmd
    # Bound to localhost the published port would accept connections and
    # then refuse them, which reads as a broken app rather than a typo.
    assert "0.0.0.0" in cmd


def test_healthcheck_uses_the_interpreter_not_curl():
    # The slim image ships no curl or wget; a probe that shells out to one
    # reports unhealthy forever while the service is fine.
    check = _directive("HEALTHCHECK")
    assert "/health" in check
    assert "curl" not in check and "wget" not in check


# ---------- the image and Compose agree ----------


def test_compose_mounts_a_volume_at_the_images_data_dir(data_dir):
    compose = COMPOSE.read_text(encoding="utf-8")
    assert f":{data_dir}" in compose, (
        f"the image writes to {data_dir}; Compose must mount a volume there or the "
        "corpus and run history vanish with the container"
    )
    assert re.search(r"^volumes:", compose, re.MULTILINE), "the mount must be a named volume"


def test_compose_publishes_the_exposed_port():
    port = _directive("EXPOSE").split()[0]
    assert f":{port}" in COMPOSE.read_text(encoding="utf-8")


def test_settings_takes_the_data_dir_from_the_environment(data_dir, monkeypatch):
    # The whole container contract in one assertion: the image configures the
    # app with an env var and nothing else.
    monkeypatch.setenv("YARA_DATA_DIR", data_dir)
    assert Settings().data_dir == Path(data_dir)


# ---------- CI ----------


def test_ci_matrix_covers_every_supported_interpreter():
    supported = set(re.findall(r"Python :: (3\.\d+)", PYPROJECT.read_text(encoding="utf-8")))
    assert supported, "pyproject should classify the interpreters it supports"
    workflow = WORKFLOW.read_text(encoding="utf-8")
    matrix = set(re.findall(r'"(3\.\d+)"', workflow))
    assert supported <= matrix, f"CI never tests {sorted(supported - matrix)}"


def test_ci_needs_no_api_keys():
    # The offline echo provider is what makes this true; a `secrets.` lookup
    # creeping in would mean forks and pull requests get a red, unfixable run.
    assert "secrets." not in WORKFLOW.read_text(encoding="utf-8")
