#!/bin/sh
# Valid client-dropout runs for the SL case base: one client absent, traced, not gated.
# When the gate excludes an attacker the global model becomes a 2-client average, so the case
# base needs valid 2-client trajectories too. Each client is dropped in turn, twice.
cd "$(dirname "$0")/.."
run() {  # name, extra args...
  name=$1; shift
  if [ -f "sl_runs/$name/server_round_10.pt" ] && [ -f "sl_runs/$name/sl_gate_log.jsonl" ]; then
    echo "=== $name skip (done)"; return; fi
  if powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'run_fl_cae_system.py' }) { exit 0 } else { exit 1 }"; then
    echo "=== $name: another FL run is active; stopping"; exit 1; fi
  rm -rf "sl_runs/$name"
  echo "=== $name $(date +%T)"
  PYTHONIOENCODING=utf-8 python -u run_fl_cae_system.py --num-rounds 10 --checkpoint-dir "sl_runs/$name" "$@" > "sl_runs/$name.log" 2>&1
  echo "=== $name exit=$? $(date +%T) models=$(grep -c . "sl_runs/$name/sl_gate_log.jsonl" 2>/dev/null)"
  sleep 5
}
for s in s1 s2; do
  for c in 1 2 3; do
    run base_dropout_c${c}_$s --sl-traces --absent-client client_$c
  done
done
