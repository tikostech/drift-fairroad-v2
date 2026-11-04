#!/bin/sh
# Live-gate validation of gate v6 (recommended): case base v2, rejection budget, checkpoints traced but not gated.
cd "$(dirname "$0")/.."
NEW=0
run() {  # name, extra args...
  name=$1; shift
  if [ -f "sl_runs/$name/server_round_10.pt" ] && [ -f "sl_runs/$name/sl_gate_log.jsonl" ]; then
    echo "=== $name skip (done)"; return; fi
  if [ -n "$MAX_RUNS" ] && [ "$NEW" -ge "$MAX_RUNS" ]; then echo "=== $name deferred"; return; fi
  if powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'run_fl_cae_system.py' }) { exit 0 } else { exit 1 }"; then
    echo "=== $name: another FL run is active; stopping"; exit 1; fi
  NEW=$((NEW + 1))
  rm -rf "sl_runs/$name"
  echo "=== $name $(date +%T)"
  PYTHONIOENCODING=utf-8 python -u run_fl_cae_system.py --num-rounds 10 --checkpoint-dir "sl_runs/$name" "$@" > "sl_runs/$name.log" 2>&1
  echo "=== $name exit=$? $(date +%T) models=$(grep -c . "sl_runs/$name/sl_gate_log.jsonl" 2>/dev/null)"
  sleep 5
}
for v in v6; do
  CB="sl_runs/case_base_v2.npz --sl-gate-scope clients"  # v6 = v5 with checkpoints traced, not gated
  for s in s1 s2 s3; do
    run live_${v}_clean_$s     --sl-case-base $CB
    run live_${v}_scale_$s     --sl-case-base $CB --attack scale
    run live_${v}_noise_x2_$s  --sl-case-base $CB --attack noise --attack-scale 2.0
    run live_${v}_sign_flip_$s --sl-case-base $CB --attack sign_flip
  done
done
