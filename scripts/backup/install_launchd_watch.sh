#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS (launchd) only." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
label="com.sage.data-sync"
plist_dir="${HOME}/Library/LaunchAgents"
plist_path="${plist_dir}/${label}.plist"

mkdir -p "$plist_dir"

cat >"$plist_path" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${label}</string>
    <key>ProgramArguments</key>
    <array>
      <string>${repo_root}/scripts/backup/sync_data_to_volume.sh</string>
    </array>
    <key>WatchPaths</key>
    <array>
      <string>${repo_root}/data</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${repo_root}/logs/backup/launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>${repo_root}/logs/backup/launchd.err.log</string>
  </dict>
</plist>
EOF

mkdir -p "${repo_root}/logs/backup"

launchctl unload "$plist_path" >/dev/null 2>&1 || true
launchctl load "$plist_path"

echo "Installed launchd watcher:"
echo "  $plist_path"
echo "It triggers sync on changes under:"
echo "  ${repo_root}/data"

