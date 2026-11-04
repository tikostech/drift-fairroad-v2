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
"""
import argparse
import os

import numpy as np

from sl_gate import SLCaseBase, load_run

Q_GRID = (0.95, 0.98, 0.99, 0.995, 1.0)
MARGIN_GRID = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)


def held_out_scores(Z, ids, rounds, images, groups, names, kw):
    """For each run: (case base without it, [(score, coverage, round) of its models])."""
    out = {}
    for g in np.unique(groups):
        tr = groups != g
        cb = SLCaseBase(**kw).fit(Z[tr], ids[tr], rounds[tr], images[tr], groups[tr], names)
        xs = []
        for m in np.unique(ids[~tr]):
            rows = ids == m
            s, c, *_ = cb._score(cb.z(Z[rows]), int(rounds[rows][0]))
            xs.append((s, c, cb._candidates(int(rounds[rows][0]))[1]))
        out[g] = (cb, xs)
        print(f"  held out {g}: {len(xs)} models", flush=True)
    return out


def held_out_fpr(held, calibration, q, margin, coverage_rule):
    flags = []
    for cb, xs in held.values():
        cb.set_thresholds(calibration, q, margin)
        flags += [s > cb.thresholds[r]["score"] or (coverage_rule and c < cb.thresholds[r]["coverage"])
                  for s, c, r in xs]
    return float(np.mean(flags)), int(np.sum(flags)), len(flags)


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

    Z = np.concatenate(Zs)
    ids, groups, rounds, images = np.array(ids), np.array(groups), np.array(rounds), np.array(images)
    # Hold out whole runs when there are several; otherwise each model is its own group.
    cal_groups = groups if len(set(groups)) > 1 else ids
    kw = dict(k=args.k, k_pca=args.k_pca, window=args.window, match_images=not args.pooled_images,
              calibration=args.calibration, coverage_rule=args.coverage_rule)
    q, margin = args.q, args.margin
    selection = None
    if args.target_fpr is not None:
        if len(set(groups)) < 3:
            raise SystemExit("--target-fpr needs at least 3 runs (each is held out in turn)")
        print(f"\nSelecting the operating point: leave-one-run-out over {len(set(groups))} valid runs")
        held = held_out_scores(Z, ids, rounds, images, groups, names, kw)
        full = SLCaseBase(**kw).fit(Z, ids, rounds, images, cal_groups, names)
        grid = []
        for qq in Q_GRID:
            for mm in MARGIN_GRID:
                fpr, k, n = held_out_fpr(held, args.calibration, qq, mm, args.coverage_rule)
                full.set_thresholds(args.calibration, qq, mm)
                level = float(np.mean([t["score"] for t in full.thresholds.values()]))
                grid.append((qq, mm, fpr, k, n, level))
        ok = [g for g in grid if g[2] <= args.target_fpr]
        if not ok:
            raise SystemExit(f"No (q, margin) reaches held-out FPR <= {args.target_fpr}; "
                             f"the lowest is {min(g[2] for g in grid):.4f}")
        q, margin, fpr, k, n, level = min(ok, key=lambda g: g[5])  # most sensitive that meets the target
        print(f"   {'q':>6} {'margin':>6} {'held-out FPR':>13} {'mean thr':>9}")
        for g in grid:
            mark = "  <- chosen" if (g[0], g[1]) == (q, margin) else ""
            print(f"   {g[0]:>6} {g[1]:>6} {g[3]:>5}/{g[4]:<4} {g[2]:>6.2%} {g[5]:>9.2f}{mark}")
        selection = dict(target_fpr=args.target_fpr, held_out_fpr=fpr, held_out_flagged=k, held_out_models=n,
                         q=q, margin=margin, grid=[dict(q=g[0], margin=g[1], fpr=g[2]) for g in grid])

    cb = SLCaseBase(q=q, margin=margin, **kw).fit(
        Z, ids, rounds, images, cal_groups, names,
        meta=dict(probe_hash=probe_hash, runs=[os.path.normpath(r) for r in args.runs], selection=selection))
    cb.save(args.out)

    print(f"\nCase base: {cb.meta['n_models']} valid models, {cb.meta['n_cases']} cases, "
          f"{len(names)} features, calibrated over {cb.meta['n_groups']} held-out groups "
          f"({args.calibration}, q={q}, margin={margin}, coverage rule {'on' if args.coverage_rule else 'off'})")
    print(f"   {'round':>5} {'models':>6} {'held-out score max':>18} {'score thr':>9} "
          f"{'held-out cov min':>16} {'cov thr':>7}")
    for r, t in cb.thresholds.items():
        near = [c for c in cb.calibration if c["round"] == r]
        print(f"   {r:>5} {len(near):>6} {max(c['score'] for c in near):>18.3f} {t['score']:>9.3f} "
              f"{min(c['coverage'] for c in near):>16.3f} {t['coverage']:>7.3f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
