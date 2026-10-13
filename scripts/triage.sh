#!/usr/bin/env bash
# triage.sh LOGFILE [N]: the N most frequent error signatures in a log.
# Volatile parts (timestamps, hex ids, numbers) are masked so that one
# failure repeated a thousand times counts as one signature, not a thousand.
set -euo pipefail

log="${1:?usage: triage.sh LOGFILE [N]}"
top="${2:-10}"
[[ -r "$log" ]] || { echo "triage: cannot read $log" >&2; exit 66; }

{ grep -E ' (ERROR|CRITICAL) ' "$log" || true; } \
  | sed -E -e 's/^[0-9T:. -]+//' \
           -e 's/[0-9a-f]{8,}/<id>/g' \
           -e 's/[0-9]+/<n>/g' \
  | sort | uniq -c | sort -rn \
  | awk -v n="$top" 'NR <= n'
