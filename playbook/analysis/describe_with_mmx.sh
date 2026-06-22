#!/usr/bin/env bash
# Walk the report dir and call `mmx vision describe` on each PNG. Writes
# the natural-language description next to each graph as <name>.txt.
# Useful for shippable summaries in chat / email.
set -u
report_dir="${1:?usage: describe_with_mmx.sh <report-dir>}"

if ! command -v mmx >/dev/null 2>&1; then
  echo "mmx CLI not on PATH; skipping descriptions" >&2
  exit 0
fi

for png in "$report_dir"/*.png; do
  [ -f "$png" ] || continue
  base="${png%.png}"
  out="$base.txt"
  [ -f "$out" ] && continue
  echo "describing $(basename "$png")..." >&2
  mmx vision describe \
    --image "$png" \
    --prompt "Describe this expert-routing analysis chart: axes, structure, key takeaways. Be concise." \
    --quiet --output json 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))' \
  > "$out" || echo "(failed for $png)" >&2
done
