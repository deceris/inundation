image := "deceris-inundation:dev"
in_container := env_var_or_default("IN_CONTAINER", "")
docker_prefix := if in_container == "1" { "" } else { "docker compose exec -T dev" }

# Build the dev image (system deps + venv + kp), frozen lock. Run whenever
# Dockerfile / pyproject.toml / uv.lock / vulkan/patches / build-kp-linux.sh change.
image-build:
    docker build -t {{image}} .

# Start the persistent dev container (docker-compose.yml). Run this once at
# the start of a dev session; all other recipes exec into it via
# docker_prefix instead of spinning up a throwaway container per command.
up:
    docker compose up -d

# Stop the persistent dev container.
down:
    docker compose down

# Regenerate uv.lock from pyproject.toml. Uses the official astral/uv image
# pinned to the newest published 0.11.x with python3.12-bookworm — same
# version as the Dockerfile's install-script pin. Both stay in sync. The
# dev image's own build depends on uv.lock already existing (COPY +
# --frozen), so resolving the lock can't run inside the dev image.
lock:
    docker run --rm -v "$(pwd):/workspace" -w /workspace \
        astral/uv:0.11.29-python3.12-trixie uv lock

lint:
    {{docker_prefix}} uv run ruff check .

format:
    {{docker_prefix}} uv run ruff format .

format-check:
    {{docker_prefix}} uv run ruff format --check .

format-file file:
    {{docker_prefix}} uv run ruff format {{file}}

pyright-lsp:
    {{docker_prefix}} uv run pyright-langserver --stdio

typecheck:
    {{docker_prefix}} uv run pyright

test:
    {{docker_prefix}} uv run pytest

# Lint markdown docs under docs/ + root README.md + AGENTS.md. Config at
# repo root: .pymarkdown (passed via --config; pymarkdown does not auto-
# discover). Archive under docs/archive/** is excluded via --exclude glob
# (frozen historical snapshot; original prose formatting preserved as-is).
mdlint:
    {{docker_prefix}} uv run pymarkdown --config .pymarkdown scan -r \
        -e "docs/archive/**" docs/ README.md AGENTS.md

# Auto-fix markdown issues where mechanical (MD007 ul-indent, MD012 blanks,
# MD013 line-length wraps, MD019/021 hash spacing, MD022/031/032 blank-line
# around headings/fences/lists, MD035 hr-style, MD046 code-block-style, MD047
# trailing-newline, MD048 code-fence-style). Non-fixable rules (MD040 fenced-
# code-language, MD041 first-line-h1, MD024 dup-heading, MD025 single-h1,
# etc.) still need manual touch-up. Always re-run `just mdlint` after.
mdlint-fix:
    {{docker_prefix}} uv run pymarkdown --config .pymarkdown fix -r \
        -e "docs/archive/**" docs/ README.md AGENTS.md

# Full CI gate.
check: lint format-check typecheck mdlint test


benchmark-lake *args:
    {{docker_prefix}} uv run python -m inundation.bench.lake_at_rest {{args}}


# Tier 1 radial dam-break benchmark.
benchmark-radial-dambreak *args:
    {{docker_prefix}} uv run python -m inundation.bench.radial_dambreak {{args}}

# EA Test 2 floodplain-depressions benchmark.
benchmark-depressions *args:
    {{docker_prefix}} uv run python -m inundation.bench.floodplain_depressions {{args}}

# EA Test 3 momentum-obstruction benchmark.
benchmark-obstruction *args:
    {{docker_prefix}} uv run python -m inundation.bench.momentum_obstruction {{args}}

# EA Test 4 flood-propagation benchmark.
benchmark-propagation *args:
    {{docker_prefix}} uv run python -m inundation.bench.flood_propagation {{args}}

# Steady flat-channel open-boundary benchmark (inflow + fixed depth).
benchmark-open-channel *args:
    {{docker_prefix}} uv run python -m inundation.bench.open_channel {{args}}

# Run the benchmark harnesses and fold logs, metrics, and animations into
# one self-contained HTML report at .tmp/bench-report/report.html.
benchmark-report *args:
    {{docker_prefix}} uv run python -m inundation.bench.report {{args}}
shell:
    {{docker_prefix}} bash

# Build the Apptainer SIF for production from the already-built dev image.
apptainer-build:
    apptainer build inundation.sif inundation.def

# Run a command inside the SIF, bind-mounting the repo at /workspace like the
# e.g. `just apptainer-run -- python -m inundation.bench.lake_at_rest`
apptainer-run *args:
    apptainer exec --nv --bind .:/workspace inundation.sif {{args}}