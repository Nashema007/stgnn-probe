#!/usr/bin/env bash
#
# Bundle probe results / predictions / metrics into a single archive for transfer.
#
# Usage:
#   scripts/zip_results.sh                # results + predictions (light, no model weights)
#   scripts/zip_results.sh --with-weights # also include checkpoint .pt files (large)
#   scripts/zip_results.sh --name foo     # custom archive basename
#
# Automatically includes outputs/, data/probe_inputs/, checkpoints/, and any
# outputs_*/ sensitivity-run folder (e.g. outputs_pmax12 from the p_max=12 rerun).
#
set -euo pipefail

# Resolve repo root regardless of where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

WITH_WEIGHTS=0
NAME="stgnn_results_$(hostname -s 2>/dev/null || echo host)_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-weights) WITH_WEIGHTS=1; shift ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Candidate folders to include (skipped automatically if absent).
INCLUDE=(
  "outputs"                 # metrics, comparative dashboards, per-model results, plots
  "data/probe_inputs"       # predictions + ground_truth + adjacency the probes consumed
)

# Also sweep in any alternate probe-output dirs from sensitivity runs — e.g.
# outputs_pmax12 from the p_max=12 Granger rerun (run_probe.py --output-dir).
# Globbed so any outputs_* variant is picked up automatically; nullglob makes the
# loop a no-op when none exist.
shopt -s nullglob
for _alt in outputs_*/; do INCLUDE+=("${_alt%/}"); done
shopt -u nullglob

# Logical exclude patterns (translated per archiver below).
# Checkpoints hold predictions (*_preds.npy) plus adjacency; weights (*_best.pt) are optional.
EXCLUDE_PATTERNS=(
  ".DS_Store"
  "outputs/e2e_dummy"
  "__pycache__"
)

PRESENT=()
for d in "${INCLUDE[@]}"; do
  if [[ -e "$d" ]]; then PRESENT+=("$d"); else echo "skip (not found): $d"; fi
done

# Always grab prediction/adjacency arrays + metric json from checkpoints.
if [[ -d checkpoints ]]; then
  PRESENT+=("checkpoints")
  if [[ "${WITH_WEIGHTS}" -eq 0 ]]; then
    EXCLUDE_PATTERNS+=( "*.pt" )
    echo "note: excluding checkpoint *.pt weights (use --with-weights to include)"
  fi
fi

if [[ ${#PRESENT[@]} -eq 0 ]]; then
  echo "Nothing to archive — no result folders found under ${ROOT}" >&2
  exit 1
fi

echo "Archiving: ${PRESENT[*]}"

# Prefer zip; fall back to tar (.tar.gz) when zip isn't installed.
if command -v zip >/dev/null 2>&1; then
  ARCHIVE="${ROOT}/${NAME}.zip"
  ZIP_EXCLUDES=()
  for p in "${EXCLUDE_PATTERNS[@]}"; do
    case "$p" in
      *.pt)               ZIP_EXCLUDES+=( -x "*.pt" ) ;;
      outputs/e2e_dummy)  ZIP_EXCLUDES+=( -x "outputs/e2e_dummy/*" ) ;;
      *)                  ZIP_EXCLUDES+=( -x "*/${p}/*" -x "*/${p}" ) ;;
    esac
  done
  rm -f "${ARCHIVE}"
  zip -r -q "${ARCHIVE}" "${PRESENT[@]}" "${ZIP_EXCLUDES[@]}"
else
  echo "note: 'zip' not found — using tar (.tar.gz) instead"
  ARCHIVE="${ROOT}/${NAME}.tar.gz"
  TAR_EXCLUDES=()
  for p in "${EXCLUDE_PATTERNS[@]}"; do
    TAR_EXCLUDES+=( "--exclude=${p}" )
  done
  rm -f "${ARCHIVE}"
  tar "${TAR_EXCLUDES[@]}" -czf "${ARCHIVE}" "${PRESENT[@]}"
fi

SIZE="$(du -h "${ARCHIVE}" | cut -f1)"
echo ""
echo "Created: ${ARCHIVE}  (${SIZE})"
