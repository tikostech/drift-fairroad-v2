#!/bin/sh
# Full SL-gate experiment on the GPU with the optimised pipeline, into sl_runs_gpu/.
#
#   sh sl_runs_gpu/run_gpu_experiment.sh base       # 14 valid runs for the case bases (8 clean, 6 client dropout)
#   sh sl_runs_gpu/run_gpu_experiment.sh capture    # 15 traced attack runs (5 attacks x 3), scored offline
#   sh sl_runs_gpu/run_gpu_experiment.sh casebase   # build case bases v1 (original), v2 (final), v3 (no coverage rule)
#   sh sl_runs_gpu/run_gpu_experiment.sh live       # gate versions v1-v6 live, 4 conditions x 3 runs each
#
# MAX_RUNS=N stops a phase after N new runs (background time limit); re-run to continue. Finished runs are
# skipped. Gate versions v1-v5 are reproduced with server policy flags:
#   v1  case base v1 (2 clean runs, max x1.25), every invalid update rejected, checkpoints gated from round 1
#   v2  case base v2 (14 runs, pooled calibration), same policy as v1
#   v3  v2 + rejection budget (1 per round) + round-1 checkpoint warm-up
#   v4  case base v3 (no coverage rule) + budget + warm-up + stall guard
#   v5  case base v2 + budget + warm-up + stall guard
#   v6  v5 with checkpoints traced but not gated (final)
cd "$(dirname "$0")/.."
PY=.venv-gpu/Scripts/python
D=sl_runs_gpu
NEW=0
run() {  # name, args...
  name=$1; shift
  if [ -f "$D/$name/server_round_10.pt" ] && [ -f "$D/$name/sl_gate_log.jsonl" ]; then return; fi
  if [ -n "$MAX_RUNS" ] && [ "$NEW" -ge "$MAX_RUNS" ]; then echo "=== $name deferred"; return; fi
  if powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'run_fl_cae_system.py' }) { exit 0 } else { exit 1 }"; then
    echo "=== $name: another FL run is active; stopping"; exit 1; fi
  NEW=$((NEW + 1))
  rm -rf "${D:?}/${name:?}"
  t0=$(date +%s)
  PYTHONIOENCODING=utf-8 $PY -u run_fl_cae_system.py --num-rounds 10 --checkpoint-dir "$D/$name" --no-final-eval "$@" > "$D/$name.log" 2>&1
  echo "=== $name exit=$? $(( $(date +%s) - t0 ))s models=$(grep -c . "$D/$name/sl_gate_log.jsonl" 2>/dev/null)"
  sleep 2
}
CLEAN="clean_a clean_b clean_c clean_c_s2 clean_c_s3 clean_d clean_d_s2 clean_d_s3"
DROPOUT="base_dropout_c1_s1 base_dropout_c2_s1 base_dropout_c3_s1 base_dropout_c1_s2 base_dropout_c2_s2 base_dropout_c3_s2"

case "$1" in
base)
  for n in $CLEAN; do run $n --sl-traces; done
  for n in $DROPOUT; do c=$(echo $n | sed 's/base_dropout_c\([0-9]\).*/\1/'); run $n --sl-traces --absent-client client_$c; done ;;
capture)
  for s in "" _s2 _s3; do
    run attack_sign_flip$s --sl-traces --attack sign_flip
    run attack_scale$s     --sl-traces --attack scale
    run attack_noise$s     --sl-traces --attack noise
    run attack_noise_x2$s  --sl-traces --attack noise --attack-scale 2.0
    run attack_inverted$s  --sl-traces --attack inverted
  done ;;
casebase)
  base_runs=""; for n in $CLEAN $DROPOUT; do base_runs="$base_runs $D/$n"; done
  $PY build_sl_case_base.py --runs $D/clean_a $D/clean_b --out $D/case_base_v1.npz --calibration max --q 1.0 --margin 1.25 --coverage-rule | tail -3
  $PY build_sl_case_base.py --runs $base_runs --out $D/case_base_v2.npz --target-fpr 0.005 --coverage-rule | grep -E "chosen|Case base:"
  $PY build_sl_case_base.py --runs $base_runs --out $D/case_base_v3.npz --target-fpr 0.005 | grep -E "chosen|Case base:" ;;
live)
  OLD="--sl-max-rejections 3 --sl-checkpoint-warmup 0 --sl-checkpoint-stall-guard -1"
  for v in v6 v5 v4 v3 v2 v1; do
    case $v in
      v1) G="--sl-case-base $D/case_base_v1.npz $OLD" ;;
      v2) G="--sl-case-base $D/case_base_v2.npz $OLD" ;;
      v3) G="--sl-case-base $D/case_base_v2.npz --sl-checkpoint-stall-guard -1" ;;
      v4) G="--sl-case-base $D/case_base_v3.npz" ;;
      v5) G="--sl-case-base $D/case_base_v2.npz" ;;
      v6) G="--sl-case-base $D/case_base_v2.npz --sl-gate-scope clients" ;;
    esac
    for s in s1 s2 s3; do
      run live_${v}_clean_$s     $G
      run live_${v}_scale_$s     $G --attack scale
      run live_${v}_noise_x2_$s  $G --attack noise --attack-scale 2.0
      run live_${v}_sign_flip_$s $G --attack sign_flip
    done
  done ;;
*) echo "usage: $0 base|capture|casebase|live"; exit 2 ;;
esac
