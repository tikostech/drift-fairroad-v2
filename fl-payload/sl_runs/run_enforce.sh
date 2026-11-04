#!/bin/sh
# Live-gate runs against sl_runs/case_base.npz, plus a stronger noise attack captured for offline eval.
cd "$(dirname "$0")/.."
run() {  # name, extra args...
  name=$1; shift
  echo "=== $name $(date +%T)"
  PYTHONIOENCODING=utf-8 python -u run_fl_cae_system.py --num-rounds 10 --checkpoint-dir "sl_runs/$name" "$@" > "sl_runs/$name.log" 2>&1
  echo "=== $name exit=$? $(date +%T) models=$(grep -c . "sl_runs/$name/sl_gate_log.jsonl" 2>/dev/null)"
}
# run enforce_clean     --sl-case-base sl_runs/case_base.npz
run enforce_scale     --sl-case-base sl_runs/case_base.npz --attack scale
run enforce_sign_flip --sl-case-base sl_runs/case_base.npz --attack sign_flip
run attack_noise_x2   --sl-traces --attack noise --attack-scale 2.0
