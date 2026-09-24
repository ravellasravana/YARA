# No `# syntax=` directive: nothing below needs a BuildKit-only instruction
# (no cache mounts, no heredocs), and pinning a frontend would make every
# build pull an extra image first. Plain Dockerfile syntax builds anywhere.

# Two stages so the runtime image never carries pip's cache, the build
# metadata, or a writable site-packages owned by the app user. The builder
# installs into a virtualenv; the runtime stage copies that one directory.
# Both stages share a base image, which is what makes the venv portable -
# swapping either tag alone would break the interpreter path baked into it.

ARG PYTHON_VERSION=3.12

# ---------- builder ----------

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# No build-essential: numpy, scikit-learn and pydantic-core all publish
# manylinux wheels for the supported interpreters, so nothing compiles here.
# If a future dependency has no wheel, the install fails loudly and this is
# the line to revisit rather than a mystery to debug at runtime.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src

# Metadata first, sources second: editing yara/ then only re-runs the last
# layer, and a dependency install of ~100 MB stays cached between builds.
COPY pyproject.toml README.md LICENSE ./
COPY yara ./yara

# The `api` extra, not `dev`: the image serves the API, it doesn't run tests.
RUN pip install --no-cache-dir ".[api]"

# ---------- runtime ----------

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    YARA_DATA_DIR=/data

# A fixed uid/gid, not just a name: bind-mounted host directories are matched
# by number, so pinning it keeps file ownership predictable across machines.
RUN groupadd --system --gid 1001 yara \
    && useradd --system --uid 1001 --gid yara --home-dir /home/yara --create-home yara

COPY --from=builder --chown=root:root /opt/venv /opt/venv

# Owned by the app user because this is the one path the process writes:
# the SQLite database and the vector index live here. Everything else can
# stay root-owned and read-only to the process.
RUN mkdir -p /data && chown yara:yara /data
VOLUME ["/data"]

USER yara
WORKDIR /data
EXPOSE 8000

# curl and wget are not in the slim image and adding them for a health probe
# would mean a package manager in the final layer; the interpreter is already
# here. Exit status is what Docker reads, so urlopen raising is the failure.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly from
# `docker stop` instead of a shell swallowing it. Host 0.0.0.0 is required
# for the port to be reachable from outside the container's namespace.
CMD ["uvicorn", "yara.api:app", "--host", "0.0.0.0", "--port", "8000"]
