#!/usr/bin/env bash
# Extract only the Granger artifacts from a results tarball.
#   - outputs/granger_cache/<DATASET>.npz          (F-stats / p-values cache)
#   - outputs/dataset/<DATASET>/granger/*          (gcg matrix, fstats, pvalues, figures)
#
# Usage: bash scripts/extract_granger.sh <results.tar.gz> [dest_dir]
set -euo pipefail

T="${1:?usage: extract_granger.sh <results.tar.gz> [dest_dir]}"
DEST="${2:-granger_only}"

list=$(mktemp); trap 'rm -f "$list"' EXIT
tar tzf "$T" | grep -E 'granger_cache/[^/]+\.npz$|/granger/' > "$list" || true
[ -s "$list" ] || { echo "No Granger files found in $T" >&2; exit 1; }

mkdir -p "$DEST"
tar xzf "$T" -C "$DEST" -T "$list"
echo "Extracted $(grep -c . "$list") Granger files to $DEST/"
