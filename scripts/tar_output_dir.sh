#!/usr/bin/env bash
#
# Bundle ONE probe output directory into its own small archive for transfer.
#
# zip_results.sh sweeps in outputs/, checkpoints/, data/probe_inputs/ AND every
# outputs_*/ sensitivity dir, which is why the transfer archives run to ~4 GB.
# This packs a single output dir (the kind produced by
# `run_probe.py --output-dir <name>`) so the download is ~150 MB instead.
#
# Usage (run on the server; works from any cwd):
#   scripts/tar_output_dir.sh                      # defaults to outputs_fixed12
#   scripts/tar_output_dir.sh outputs_pmax12       # some other sweep dir
#   scripts/tar_output_dir.sh outputs_fixed12 --name my_archive
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DIR="outputs_fixed12"
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) DIR="${1%/}"; shift ;;
  esac
done

if [[ ! -d "$DIR" ]]; then
  echo "No such directory under ${ROOT}: ${DIR}" >&2
  echo "" >&2
  echo "Available probe output dirs:" >&2
  found=()
  [[ -d outputs ]] && found+=("outputs")
  shopt -s nullglob
  for d in outputs_*/; do found+=("${d%/}"); done
  shopt -u nullglob
  if [[ ${#found[@]} -eq 0 ]]; then
    echo "  (none)" >&2
  else
    for d in "${found[@]}"; do echo "  ${d%/}"; done >&2
  fi
  exit 1
fi

: "${NAME:=${DIR}_$(hostname -s 2>/dev/null || echo host)_$(date +%Y%m%d_%H%M%S)}"
ARCHIVE="${ROOT}/${NAME}.tar.gz"

rm -f "${ARCHIVE}"
tar --exclude='.DS_Store' --exclude='__pycache__' -czf "${ARCHIVE}" "${DIR}"

SIZE="$(du -h "${ARCHIVE}" | cut -f1)"
COUNT="$(find "${DIR}" -type f | wc -l | tr -d ' ')"

echo "Created: ${ARCHIVE}  (${SIZE}, ${COUNT} files from ${DIR}/)"
echo ""
echo "Pull it with:"
echo "  scp $(whoami)@$(hostname -s 2>/dev/null || echo SERVER):${ARCHIVE} ."
