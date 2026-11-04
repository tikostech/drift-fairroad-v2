#!/bin/sh
# Capture-only runs for building and evaluating the SL case base (no live gate).
cd "$(dirname "$0")/.."
for spec in clean_a: clean_b: clean_c: attack_sign_flip:sign_flip attack_scale:scale attack_noise:noise attack_inverted:inverted; do
  name=${spec%%:*}; attack=${spec#*:}
  echo "=== $name $(date +%T)"
  if [ -n "$attack" ]; then extra="--attack $attack"; else extra=""; fi
  PYTHONIOENCODING=utf-8 python -u run_fl_cae_system.py --num-rounds 10 --checkpoint-dir "sl_runs/$name" --sl-traces $extra > "sl_runs/$name.log" 2>&1
  echo "=== $name exit=$? $(date +%T) models=$(grep -c . "sl_runs/$name/sl_gate_log.jsonl" 2>/dev/null)"
done
