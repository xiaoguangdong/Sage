#!/usr/bin/env bash
set -euo pipefail

LABEL="com.sage.daily-tminus1"
SRC="/Users/dongxg/SourceCode/Sage/scripts/schedule/com.sage.daily-tminus1.plist"
DST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [ ! -f "$SRC" ]; then
  echo "missing plist: $SRC"
  exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents"

if launchctl list | grep -q "$LABEL"; then
  launchctl unload "$DST" || true
fi

cp "$SRC" "$DST"
launchctl load "$DST"
launchctl list | grep "$LABEL" || true
