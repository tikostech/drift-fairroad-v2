#!/bin/sh
# Replicates (_s2, _s3) of every condition for the paper; the original runs are seed 1.
# enforce_noise_x2 is new, so it gets three runs.
cd "$(dirname "$0")/.."
CB="--sl-case-base sl_runs/case_base.npz"
# MAX_RUNS=N: stop after N new runs, so each invocation fits a background time limit; re-run to continue.
NEW=0
run() {  # name, extra args...
  name=$1; shift
  # Done = the round-10 checkpoint exists (a fully rejected round traces 3 models, not 4).
  if [ -f "sl_runs/$name/server_round_10.pt" ] && [ -f "sl_runs/$name/sl_gate_log.jsonl" ]; then
    echo "=== $name skip (done)"; return; fi
  if [ -n "$MAX_RUNS" ] && [ "$NEW" -ge "$MAX_RUNS" ]; then echo "=== $name deferred"; return; fi
  if powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'run_fl_cae_system.py' }) { exit 0 } else { exit 1 }"; then
    echo "=== $name: another FL run is active (two runs would share the server port); stopping"; exit 1; fi
  NEW=$((NEW + 1))
  rm -rf "sl_runs/$name"
  echo "=== $name $(date +%T)"
  PYTHONIOENCODING=utf-8 python -u run_fl_cae_system.py --num-rounds 10 --checkpoint-dir "sl_runs/$name" "$@" > "sl_runs/$name.log" 2>&1
  echo "=== $name exit=$? $(date +%T) models=$(grep -c . "sl_runs/$name/sl_gate_log.jsonl" 2>/dev/null)"
  sleep 5  # let the previous server's socket close before the next run binds the port
}
for s in s2 s3; do
  run clean_c_$s           --sl-traces
  run attack_sign_flip_$s  --sl-traces --attack sign_flip
  run attack_scale_$s      --sl-traces --attack scale
  run attack_noise_$s      --sl-traces --attack noise
  run attack_noise_x2_$s   --sl-traces --attack noise --attack-scale 2.0
  run attack_inverted_$s   --sl-traces --attack inverted
  run enforce_clean_$s     $CB
  run enforce_scale_$s     $CB --attack scale
  run enforce_sign_flip_$s $CB --attack sign_flip
done
for s in "" _s2 _s3; do
  run enforce_noise_x2$s   $CB --attack noise --attack-scale 2.0
done
