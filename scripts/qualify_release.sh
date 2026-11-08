#!/usr/bin/env bash
# Release-candidate packaging qualification for mbsd.
#
#   scripts/qualify_release.sh            # qualify HEAD
#   PYTHONS="3.12" scripts/qualify_release.sh
#
# 1. Export HEAD from git (a clean checkout: no untracked or ignored files).
# 2. Build the sdist and wheel there, check them with twine.
# 3. For every supported Python, install the wheel into an isolated venv and
#    run the full test suite and corpus against the *installed* package.
# 4. Install the sdist the same way once.
# 5. Dependency floor (numpy 2.0, scipy 1.13 on the oldest Python) and
#    ceiling (newest releases on the newest Python).
# 6. A TestPyPI dry run of the upload (nothing is sent).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONS="${PYTHONS:-3.10 3.11 3.12 3.13}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mbsd-qualify.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
log() { printf '\n== %s\n' "$*"; }

log "clean export of $(git -C "$ROOT" rev-parse --short HEAD)"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$WORK"
mkdir -p "$WORK/src-tree" && mv "$WORK"/* "$WORK/src-tree/" 2>/dev/null || true
TREE="$WORK/src-tree"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$TREE/pyproject.toml")"

log "build sdist and wheel ($VERSION)"
(cd "$TREE" && uv build --out-dir "$WORK/dist" >/dev/null)
WHEEL="$(ls "$WORK"/dist/mbsd-"$VERSION"-py3-none-any.whl)"
SDIST="$(ls "$WORK"/dist/mbsd-"$VERSION".tar.gz)"
uvx --quiet twine check --strict "$WHEEL" "$SDIST"

# Tests run against the installed package: copy tests + corpus + schemas,
# but not src/, and clear the pytest pythonpath so `import mbsd` is the wheel.
SUITE="$WORK/suite"
mkdir -p "$SUITE"
cp -r "$TREE/tests" "$TREE/corpus" "$TREE/schemas" "$SUITE/"

run_suite() {  # $1 = venv python, $2 = label
  local py="$1" label="$2"
  "$py" - <<'PY'
import importlib.resources as resources, json, mbsd
files = resources.files("mbsd") / "schemas"
names = sorted(p.name for p in files.iterdir() if p.name.endswith(".json"))
assert "mbsd.planar.joint-loads-v3.schema.json" in names, names
assert "mbsd.api-surface-v1.json" in names, names
for name in names:
    json.loads((files / name).read_text(encoding="utf-8"))
assert json.loads((files / "mbsd.api-surface-v1.json").read_text()) == json.loads(json.dumps(mbsd.api_surface()))
assert "site-packages" in mbsd.__file__, mbsd.__file__
print(f"   mbsd {mbsd.__version__}: {len(names)} packaged schemas readable")
PY
  (cd "$SUITE" && "$py" -m pytest -q -p no:cacheprovider -o pythonpath= -o testpaths=tests tests 2>&1 | tail -1 | sed "s/^/   [$label] /")
}

venv_with() {  # $1 = python version, $2 = dir, rest = packages
  local version="$1" dir="$2"; shift 2
  uv venv --quiet --python "$version" "$dir"
  uv pip install --quiet --python "$dir/bin/python" "$@" pytest
}

for version in $PYTHONS; do
  log "wheel on Python $version"
  venv_with "$version" "$WORK/venv-$version" "$WHEEL"
  run_suite "$WORK/venv-$version/bin/python" "py$version wheel"
done

OLDEST="${PYTHONS%% *}"
NEWEST="${PYTHONS##* }"

log "sdist on Python $NEWEST"
venv_with "$NEWEST" "$WORK/venv-sdist" "$SDIST"
run_suite "$WORK/venv-sdist/bin/python" "py$NEWEST sdist"

log "dependency floor on Python $OLDEST (numpy 2.0, scipy 1.13)"
venv_with "$OLDEST" "$WORK/venv-floor" "$WHEEL" "numpy==2.0.*" "scipy==1.13.*"
"$WORK/venv-floor/bin/python" -c "import numpy, scipy; print('   numpy', numpy.__version__, 'scipy', scipy.__version__)"
run_suite "$WORK/venv-floor/bin/python" "floor"

log "dependency ceiling on Python $NEWEST (newest releases)"
venv_with "$NEWEST" "$WORK/venv-ceiling" "$WHEEL" --upgrade numpy scipy
"$WORK/venv-ceiling/bin/python" -c "import numpy, scipy; print('   numpy', numpy.__version__, 'scipy', scipy.__version__)"
run_suite "$WORK/venv-ceiling/bin/python" "ceiling"

log "TestPyPI dry run (nothing is uploaded)"
uv publish --dry-run --publish-url https://test.pypi.org/legacy/ --check-url https://test.pypi.org/simple/ \
  --token dry-run-placeholder "$WHEEL" "$SDIST" 2>&1 | sed 's/^/   /'

log "qualified mbsd $VERSION from $(git -C "$ROOT" rev-parse --short HEAD)"
