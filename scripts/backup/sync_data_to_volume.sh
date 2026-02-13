#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync ./data to external volume using rsync.

Default destination:
  /Volumes/SPEED/BizData/Stock/Sage/data/

Usage:
  scripts/backup/sync_data_to_volume.sh [--strict] [--dst <path>]

Options:
  --dst <path>  Destination base directory (defaults to /Volumes/SPEED/BizData/Stock/Sage)
  --strict      Exit non-zero if destination is unavailable
EOF
}

strict=0
dst_base="/Volumes/SPEED/BizData/Stock/Sage"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dst)
      dst_base="${2:-}"
      shift 2
      ;;
    --strict)
      strict=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found; cannot sync data." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
src="${repo_root}/data"
dst="${dst_base}/data"

if [[ ! -d "$src" ]]; then
  echo "No data directory at: $src (nothing to sync)"
  exit 0
fi

if [[ ! -d "$dst_base" ]]; then
  msg="Destination not available: $dst_base (is the /Volumes/SPEED volume mounted?)"
  if [[ "$strict" -eq 1 ]]; then
    echo "$msg" >&2
    exit 1
  fi
  echo "$msg"
  exit 0
fi

log_dir="${repo_root}/logs/backup"
mkdir -p "$log_dir"
log_file="${log_dir}/data-sync.log"

if ! mkdir -p "$dst" 2>/dev/null; then
  msg="Cannot create destination directory: $dst (permission denied?)"
  if [[ "$strict" -eq 1 ]]; then
    echo "$msg" >&2
    exit 1
  fi
  echo "$msg"
  exit 0
fi

# If nothing needs to be transferred (common for delete-only events), exit quickly.
preview="$(
  rsync -ani \
    --exclude ".DS_Store" \
    "$src/" "$dst/" 2>/dev/null \
    | grep -E '^(>f|>d|cd|cS|cL|cD|c\.)' \
    || true
)"
if [[ -z "$preview" ]]; then
  # No updates to sync.
  exit 0
fi

{
  echo "== Sage data sync =="
  echo "time: $(date +\"%Y-%m-%dT%H:%M:%S%z\")"
  echo "src:  $src/"
  echo "dst:  $dst/"
  if [[ -t 1 ]]; then
    rsync -a \
      --human-readable \
      --stats \
      --exclude ".DS_Store" \
      --progress \
      "$src/" "$dst/"
  else
    rsync -a \
      --human-readable \
      --stats \
      --exclude ".DS_Store" \
      "$src/" "$dst/"
  fi
  echo
} | tee -a "$log_file"
