"""
Build the SL case base from valid models' traces.

Each run directory must contain sl_traces.npz from a server started with --sl-traces (and the
runner's run_meta.json). Only valid models are used: from an attacked run, the malicious
client's updates, the checkpoints that averaged them in, and models descended from an accepted
poisoned checkpoint are left out. Client-dropout runs (run_fl_cae_system.py --absent-client) are
valid and belong in the base: when the gate excludes a client the global model becomes an
average over fewer clients, and the base should have seen such trajectories.

    python build_sl_case_base.py --runs sl_runs/clean_* sl_runs/base_dropout_* --out sl_runs/case_base_v2.npz --target-fpr 0.005

--target-fpr picks the operating point from valid data only: each run is held out in turn and
scored against a case base built from the others, and the most sensitive (q, margin) whose
held-out false-positive rate meets the target is used. No attack data is involved.

The building and selection logic is IP and lives in drift-fairroad-service (case_base_builder.py); this
is the command line around it (the CBR service's build endpoint is the other way in).
"""
import argparse
import os

import numpy as np

import ip_core  # noqa: F401  (puts the IP core from drift-fairroad-service on sys.path)
from case_base_builder import build_case_base
from trace_store import load_run

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories with sl_traces.npz")
    ap.add_argument("--out", required=True, help="Output case base (.npz)")
    ap.add_argument("--k", type=int, default=5, help="Nearest valid cases averaged per probe image")
    ap.add_argument("--k-pca", type=int, default=0, help="PCA components of the comparison space (0 = none)")
    ap.add_argument("--window", type=int, default=1, help="Compare a round-r model with valid rounds r-window..r+window")
    ap.add_argument("--pooled-images", action="store_true", help="Compare against all probe images, not image by image")
    ap.add_argument("--calibration", choices=["max", "pooled"], default="pooled",
                    help="pooled: one quantile over all rounds of median-normalised held-out scores; "
                         "max: per-round quantile within the window (the original rule)")
    ap.add_argument("--coverage-rule", action="store_true",
                    help="Also reject on low feature coverage alone (off by default: it mostly added false positives)")
    ap.add_argument("--q", type=float, default=0.99, help="Quantile of held-out valid scores the threshold sits on")
    ap.add_argument("--margin", type=float, default=1.25, help="Multiplier on that quantile")
    ap.add_argument("--target-fpr", type=float, default=None,
                    help="Choose q and margin by leave-one-run-out FPR on the valid runs (needs >= 3 runs)")
    args = ap.parse_args()

    Zs, ids, rounds, images, groups, names, probe_hash = [], [], [], [], [], None, None
    for run_dir in args.runs:
        run = load_run(run_dir)
        if names is None:
            names, probe_hash = run["names"], run["probe_hash"]
        elif run["names"] != names or run["probe_hash"] != probe_hash:
            raise SystemExit(f"{run_dir}: feature names or probe set differ from {args.runs[0]}")
        tag = os.path.basename(os.path.normpath(run_dir))
        round_of = {m["model"]: m.get("compare_round", m["round"]) for m in run["models"]}
        valid = {m["model"] for m in run["models"] if m["label"] == "valid"}
        rows = np.isin(run["row_model"], list(valid))
        Zs.append(run["Z"][rows])
        ids += [f"{tag}/{m}" for m in run["row_model"][rows]]
        rounds += [round_of[m] for m in run["row_model"][rows]]
        images += list(run["row_image"][rows])
        groups += [tag] * int(rows.sum())
        absent = run["meta"].get("absent_clients") or []
        print(f"{run_dir}: {len(valid)} valid models of {len(run['models'])}"
              + (f" (absent: {', '.join(absent)})" if absent else ""))

    try:
        cb = build_case_base(
            np.concatenate(Zs), ids, rounds, images, groups, names, probe_hash,
            k=args.k, k_pca=args.k_pca, window=args.window, pooled_images=args.pooled_images,
            calibration=args.calibration, coverage_rule=args.coverage_rule, q=args.q, margin=args.margin,
            target_fpr=args.target_fpr, meta=dict(runs=[os.path.normpath(r) for r in args.runs]))
    except ValueError as e:
        raise SystemExit(str(e))
    cb.save(args.out)

    print(f"\nCase base: {cb.meta['n_models']} valid models, {cb.meta['n_cases']} cases, "
          f"{len(names)} features, calibrated over {cb.meta['n_groups']} held-out groups "
          f"({cb.calibration_method}, q={cb.q}, margin={cb.margin}, coverage rule {'on' if cb.coverage_rule else 'off'})")
    print(f"   {'round':>5} {'models':>6} {'held-out score max':>18} {'score thr':>9} "
          f"{'held-out cov min':>16} {'cov thr':>7}")
    for r, t in cb.thresholds.items():
        near = [c for c in cb.calibration if c["round"] == r]
        print(f"   {r:>5} {len(near):>6} {max(c['score'] for c in near):>18.3f} {t['score']:>9.3f} "
              f"{min(c['coverage'] for c in near):>16.3f} {t['coverage']:>7.3f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
