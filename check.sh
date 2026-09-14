#!/usr/bin/env bash
# Local + CI validation. Keep in lockstep with .github/workflows/ci.yml.
# Usage: ./check.sh [all|host|docker]  (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-all}"
case "${MODE}" in
  all | host | docker) ;;
  *)
    echo "usage: $0 [all|host|docker]" >&2
    exit 1
    ;;
esac

# Pin matches CI/local; bump here when upgrading ruff.
RUFF_VERSION="0.16.2"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing '$1' on PATH" >&2
    exit 1
  }
}

ruff_ver() {
  # Second field only — avoids suffix false-negatives and 0.16.2 vs 0.16.20 globs.
  "$1" --version | awk '{print $2}'
}

ensure_python() {
  # Prefer .venv when it has app deps (README / PEP 668 local flow).
  # Else use PATH python that can import them (CI setup-python). Never
  # prepend .venv/bin to PATH — that shadows a good interpreter with an
  # empty one after ensure_ruff installs only ruff.
  has_deps() {
    [[ -x "$1" ]] && "$1" -c "import ollama, pydantic, PIL, pillow_heif, piexif" >/dev/null 2>&1
  }

  if has_deps "${ROOT}/.venv/bin/python"; then
    PYTHON="${ROOT}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1 && has_deps "$(command -v python)"; then
    PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1 && has_deps "$(command -v python3)"; then
    PYTHON="$(command -v python3)"
  else
    echo "error: no Python with app deps (ollama, pydantic, pillow, pillow-heif, piexif)" >&2
    echo "hint: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi
}

ensure_ruff() {
  # Never prepend .venv/bin to PATH — that replaces `python` with an empty venv.
  if command -v ruff >/dev/null 2>&1 && [[ "$(ruff_ver ruff)" == "${RUFF_VERSION}" ]]; then
    RUFF="$(command -v ruff)"
    return
  fi
  if [[ -x "${ROOT}/.venv/bin/ruff" ]] &&
    [[ "$(ruff_ver "${ROOT}/.venv/bin/ruff")" == "${RUFF_VERSION}" ]]; then
    RUFF="${ROOT}/.venv/bin/ruff"
    return
  fi
  # PEP 668: never pip-install into a distro interpreter; use a local venv.
  echo "==> ${PYTHON} -m venv .venv && pip install ruff==${RUFF_VERSION}"
  "${PYTHON}" -m venv .venv
  "${ROOT}/.venv/bin/pip" install -q "ruff==${RUFF_VERSION}"
  RUFF="${ROOT}/.venv/bin/ruff"
}

ensure_container() {
  # Prefer a working docker (CI); fall back to a working podman (common on Fedora).
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    CONTAINER_ENGINE=docker
  elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
    CONTAINER_ENGINE=podman
  else
    echo "error: need a working docker or podman on PATH" >&2
    exit 1
  fi
}

run_host() {
  ensure_python
  need shellcheck
  ensure_ruff

  echo "==> ${PYTHON} image_cull.py --self-check"
  "${PYTHON}" image_cull.py --self-check

  echo "==> ${RUFF} check image_cull.py"
  "${RUFF}" check image_cull.py

  echo "==> shellcheck setup.sh check.sh"
  shellcheck setup.sh check.sh
}

run_docker() {
  ensure_container

  echo "==> ${CONTAINER_ENGINE} build -t image-cull:local ."
  "${CONTAINER_ENGINE}" build -t image-cull:local .

  echo "==> ${CONTAINER_ENGINE} run --rm image-cull:local --self-check"
  "${CONTAINER_ENGINE}" run --rm image-cull:local --self-check

  echo "==> ${CONTAINER_ENGINE} smoke test (volume mount writes)"
  if [ "$(id -u)" -eq 0 ]; then
    echo "error: running smoke test as root makes non-root permission validation a no-op" >&2
    exit 1
  fi
  SMOKE_DIR="$(mktemp -d)"
  trap 'rm -rf "${SMOKE_DIR}"' EXIT
  mkdir -p "${SMOKE_DIR}/photos" "${SMOKE_DIR}/rejects"
  touch "${SMOKE_DIR}/photos/smoke.jpg"
  cat << 'EOF' > "${SMOKE_DIR}/photos/cull-report.json"
{
  "meta": {"threshold": 7.0, "thresholds": {"ai": 7.0, "quality": null, "generation": null}},
  "results": [{"file": "smoke.jpg", "analysis": {"realism_score": 1.0, "is_realistic": false, "detected_artifacts": ["test"], "reasoning": "test"}}]
}
EOF
  USER_ARGS=("--user" "$(id -u):$(id -g)")
  if [ "${CONTAINER_ENGINE}" = "podman" ]; then
    USER_ARGS=("--userns=keep-id" "${USER_ARGS[@]}")
  fi
  "${CONTAINER_ENGINE}" run --rm \
    "${USER_ARGS[@]}" \
    -v "${SMOKE_DIR}/photos:/photos:z" \
    -v "${SMOKE_DIR}/rejects:/filtered:z" \
    image-cull:local --dir /photos --filter-dir /filtered --apply-report >/dev/null
  test -f "${SMOKE_DIR}/rejects/smoke.jpg"
  rm -rf "${SMOKE_DIR}"
  trap - EXIT
}

case "${MODE}" in
  host) run_host ;;
  docker) run_docker ;;
  all)
    run_host
    run_docker
    ;;
esac

echo "OK: all checks passed"
