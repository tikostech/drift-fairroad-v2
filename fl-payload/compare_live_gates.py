"""
Compare live-gate runs across case-base versions: false rejections, detection and final quality.

Run names: enforce_<attack>[_sN] (v1: sl_runs/case_base.npz) and live_<version>_<attack>_sN.
Decisions are the server's own (sl_gate_log.jsonl); quality is the round-10 checkpoint on the fixed
validation set used throughout (evaluate_cae.py).

    python compare_live_gates.py --out ../paper/data/live_gate_comparison.csv
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import config

RUNS = "sl_runs"
ATTACKS = ("clean", "scale", "noise_x2", "sign_flip")


def runs_by_version():
    out = defaultdict(lambda: defaultdict(list))
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        if not (os.path.isdir(d) and os.path.exists(os.path.join(d, "server_round_10.pt"))):
            continue
        m = re.match(r"^enforce_(clean|scale|noise_x2|sign_flip)(?:_s\d+)?$", name)
        if m:
            out["v1"][m.group(1)].append(name)
            continue
        m = re.match(r"^live_(\w+?)_(clean|scale|noise_x2|sign_flip)_s\d+$", name)
        if m:
            out[m.group(1)][m.group(2)].append(name)
    return out


def decisions(name):
    """Per-run counts from the server's gate log, with honest models split by lineage."""
    d = os.path.join(RUNS, name)
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    bad = meta.get("malicious_client") if meta.get("attack") else None
    log = [json.loads(l) for l in open(os.path.join(d, "sl_gate_log.jsonl"))]
    c = defaultdict(int)
    contaminated = False
    for l in sorted(log, key=lambda l: (l["round"], l["source"] == "global")):
        rejected = not l["accepted"]
        if l["source"] == "global":
            c["ckpt"] += 1
            c["ckpt_rejected"] += rejected
            if l["accepted"] and bad in (l.get("included") or []):
                contaminated = True
        elif l["source"] == bad:
            c["attacker"] += 1
            c["attacker_rejected"] += rejected
        else:
            key = "downstream" if contaminated else "lineage"
            c[f"honest_{key}"] += 1
            c[f"honest_{key}_rejected"] += rejected
    return c


def final_quality(names):
    sys.argv = sys.argv[:1]
    from evaluate_cae import CAEEvaluator
    ev = CAEEvaluator()
    dl = ev.get_fixed_dataloader(config.CLEAN_IMAGE_DIR, config.NOISY_IMAGE_DIR, config.NUM_CLIENTS, 0)
    out = {}
    for n in names:
        ev.model.load(os.path.join(RUNS, n, "server_round_10.pt"))
        out[n] = ev.evaluate_on_dataset(dl)["mse"]
    return out


def main():
    global RUNS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="CSV of per-version, per-attack results")
    ap.add_argument("--runs-dir", default=RUNS, help="Directory with the runs (e.g. sl_runs_gpu)")
    args = ap.parse_args()
    RUNS = args.runs_dir

    by = runs_by_version()
    quality = final_quality([n for v in by.values() for ns in v.values() for n in ns])
    rows = []
    for version in sorted(by, key=lambda v: (v != "v1", v)):
        for attack in ATTACKS:
            names = by[version].get(attack, [])
            if not names:
                continue
            tot = defaultdict(int)
            for n in names:
                for k, v in decisions(n).items():
                    tot[k] += v
            mse = [quality[n] for n in names]
            honest = tot["honest_lineage"] + tot["honest_downstream"]
            honest_rej = tot["honest_lineage_rejected"] + tot["honest_downstream_rejected"]
            rows.append(dict(version=version, attack=attack, runs=len(names),
                             honest_rejected=honest_rej, honest=honest,
                             lineage_rejected=tot["honest_lineage_rejected"], lineage=tot["honest_lineage"],
                             downstream_rejected=tot["honest_downstream_rejected"], downstream=tot["honest_downstream"],
                             attacker_rejected=tot["attacker_rejected"], attacker=tot["attacker"],
                             ckpt_rejected=tot["ckpt_rejected"], ckpt=tot["ckpt"],
                             mse_mean=float(np.mean(mse)), mse_std=float(np.std(mse, ddof=1)) if len(mse) > 1 else 0.0))

    print(f"{'version':<8} {'attack':<10} {'runs':>4} {'honest rejected':>16} {'(lineage / downstream)':>23} "
          f"{'attacker rejected':>18} {'ckpt rej.':>9} {'final MSE':>17}")
    for r in rows:
        print(f"{r['version']:<8} {r['attack']:<10} {r['runs']:>4} {r['honest_rejected']:>7}/{r['honest']:<8} "
              f"{r['lineage_rejected']:>9}/{r['lineage']:<4} {r['downstream_rejected']:>3}/{r['downstream']:<4} "
              f"{(str(r['attacker_rejected']) + '/' + str(r['attacker'])) if r['attacker'] else '-':>18} "
              f"{r['ckpt_rejected']:>4}/{r['ckpt']:<4} {r['mse_mean']:.4f} ± {r['mse_std']:.4f}")
    for version in sorted({r["version"] for r in rows}, key=lambda v: (v != "v1", v)):
        sel = [r for r in rows if r["version"] == version]
        hr, h = sum(r["honest_rejected"] for r in sel), sum(r["honest"] for r in sel)
        print(f"{version:<8} all honest client updates rejected: {hr}/{h} ({100 * hr / h:.1f}%)")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
