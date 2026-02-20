#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS (launchd) only." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
label="com.sage.pg-backup"
plist_dir="${HOME}/Library/LaunchAgents"
plist_path="${plist_dir}/${label}.plist"

backup_root="${SAGE_DB_BACKUP_ROOT:-/Volumes/SPEED/BizData/Stock/Sage/db}"
container="${SAGE_DB_CONTAINER:-sage-postgres}"
db_name="${SAGE_DB_NAME:-sage_db}"
db_user="${SAGE_DB_USER:-sage}"
db_password="${SAGE_DB_PASSWORD:-sage_dev_2026}"

mkdir -p "$plist_dir"
mkdir -p "${repo_root}/logs/backup"

cat >"$plist_path" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${label}</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>${repo_root}/scripts/backup/pg_backup_daily.sh</string>
      <string>incremental</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${repo_root}</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>SAGE_DB_BACKUP_ROOT</key>
      <string>${backup_root}</string>
      <key>SAGE_DB_CONTAINER</key>
      <string>${container}</string>
      <key>SAGE_DB_NAME</key>
      <string>${db_name}</string>
      <key>SAGE_DB_USER</key>
      <string>${db_user}</string>
      <key>SAGE_DB_PASSWORD</key>
      <string>${db_password}</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
      <key>Hour</key>
      <integer>2</integer>
      <key>Minute</key>
      <integer>30</integer>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${repo_root}/logs/backup/pg_backup.out.log</string>
    <key>StandardErrorPath</key>
    <string>${repo_root}/logs/backup/pg_backup.err.log</string>
  </dict>
</plist>
EOF

domain="gui/$(id -u)"
launchctl bootout "$domain" "$plist_path" >/dev/null 2>&1 || true
launchctl bootstrap "$domain" "$plist_path"
launchctl enable "$domain/$label" >/dev/null 2>&1 || true

echo "Installed launchd job:"
echo "  $plist_path"
echo "Daily incremental backup at 02:30"
