#!/usr/bin/env bash
# Build a distributable tarball of lemonade-vision-server.
#
# Usage:       ./scripts/build_dist.sh
# Output:      dist/lemonade-vision-server-<version>.tar.gz
# Extraction:  tar -xzf <archive> --strip-components=1 -C <target>
#              The tarball has a single top-level directory that gets stripped,
#              matching lemond BackendManager's install flow.
#
# Archive contents:
#   src/lemonade_vision/   Python package source
#   pyproject.toml         Project metadata + dependencies
#   scripts/run.sh         Launcher invoked by lemond's VisionServer backend
#   README.md              (shipped for reference)
#   version.txt            Single-line version identifier

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="0.1.0"
STAGING_DIR="$(mktemp -d)"
PACKAGE_NAME="lemonade-vision-server-${VERSION}"
STAGE_ROOT="${STAGING_DIR}/${PACKAGE_NAME}"
DIST_DIR="${PROJECT_ROOT}/dist"

cleanup() { rm -rf "$STAGING_DIR"; }
trap cleanup EXIT

echo "==> Staging package: ${PACKAGE_NAME}"
mkdir -p "$STAGE_ROOT" "$DIST_DIR"

cp -r "$PROJECT_ROOT/src" "$STAGE_ROOT/src"
cp "$PROJECT_ROOT/pyproject.toml" "$STAGE_ROOT/pyproject.toml"
cp "$PROJECT_ROOT/README.md" "$STAGE_ROOT/README.md"
cp "$PROJECT_ROOT/scripts/run.sh" "$STAGE_ROOT/run.sh"
chmod +x "$STAGE_ROOT/run.sh"

echo "$VERSION" > "$STAGE_ROOT/version.txt"

echo "==> Creating tarball: ${DIST_DIR}/${PACKAGE_NAME}.tar.gz"
tar -czf "${DIST_DIR}/${PACKAGE_NAME}.tar.gz" -C "$STAGING_DIR" "$PACKAGE_NAME"

echo "==> Done: ${DIST_DIR}/${PACKAGE_NAME}.tar.gz"
