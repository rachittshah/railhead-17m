#!/usr/bin/env bash
# Build an OpenTTD engine variant into data/engines/<name>/ without touching Railhead's own build
# (~/railhead/.cache/openttd-src). Same recipe as ~/railhead/engine/build.sh (pinned tag + every Railhead
# patch, dedicated-only, OpenGFX), plus extra patches and compiler flags. Select a variant per process:
#
#   RAILHEAD_OPENTTD=$PWD/data/engines/<name>/build/openttd rh17m label ...
#
#   scripts/build_engine_variant.sh m4lto "-mcpu=apple-m4 -flto=thin" ""
#   scripts/build_engine_variant.sh m4lto-poll "-mcpu=apple-m4 -flto=thin" engine/0004-stdin-poll.patch
#   scripts/build_engine_variant.sh pgo-gen "-mcpu=apple-m4 -fprofile-generate=$PWD/data/engines/pgo-gen/prof" ""
#
# args: <name> <CXX flags> <space-separated extra patch files, may be empty>. LTO is requested by the flags
# themselves (-flto=thin), not CMAKE_INTERPROCEDURAL_OPTIMIZATION, so the same flags reach compile and link.
set -euo pipefail

NAME="${1:?usage: build_engine_variant.sh <name> <cxx flags> [extra patches]}"
FLAGS="${2:-}"
EXTRA="${3:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RH="${RAILHEAD_REPO:-$HOME/railhead}"
OUT="$ROOT/data/engines/$NAME"
SRC="$OUT/src"
BUILD="$OUT/build"
JOBS="${JOBS:-6}"

mkdir -p "$OUT"
if [ ! -d "$SRC/.git" ]; then
  # the pinned tree Railhead already cloned (shallow); a local clone keeps it offline and byte-identical
  git clone --quiet --no-local "file://$RH/.cache/openttd-src" "$SRC" 2>/dev/null \
    || git clone --quiet "$RH/.cache/openttd-src" "$SRC"
fi
git -C "$SRC" checkout --quiet -- .
for p in "$RH"/engine/patches/*.patch $EXTRA; do
  git -C "$SRC" apply --whitespace=nowarn "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
  echo "applied $(basename "$p")"
done

cmake -S "$SRC" -B "$BUILD" -G Ninja -DOPTION_DEDICATED=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="$FLAGS" -DCMAKE_EXE_LINKER_FLAGS="$FLAGS" >/dev/null
nice -n 10 ninja -C "$BUILD" -j "$JOBS" openttd

# a dedicated server still needs a base graphics set next to the binary: reuse Railhead's
for f in "$RH"/.cache/openttd-src/build/baseset/*[Oo]pen[Gg][Ff][Xx]*.tar; do
  ln -sf "$f" "$BUILD/baseset/$(basename "$f")"
done
echo "engine ready: $BUILD/openttd"
