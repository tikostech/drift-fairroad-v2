"""
Score runs' traced models against an SL case base and report how well the gate separates
valid from spoofed models.

Labels come from each run's run_meta.json (see sl_gate.load_run). Every model is re-scored
offline, so runs captured with --sl-traces only (no live gate) can be evaluated too; where the
server did gate live, its recorded verdict is reported alongside.

    python eval_sl_gate.py --case-base sl_runs/case_base.npz --runs sl_runs/clean_c sl_runs/attack_*
"""
import argparse
import json
import os

import numpy as np

import ip_core  # noqa: F401  (puts the IP core from drift-fairroad-service on sys.path)
from sl_gate import SLCaseBase
from trace_store import load_run


def roc_auc(score, positive):
    """Probability a random positive outscores a random negative (ties count half)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def rates(rows):
    """TPR over spoofed models, FPR over valid ones; downstream models are counted on their own."""
    down = [r for r in rows if r["label"] == "downstream"]
    rows = [r for r in rows if r["label"] != "downstream"]
    y = np.array([r["spoofed"] for r in rows], bool)
    flag = np.array([r["verdict"] == "invalid" for r in rows], bool)
    tp, fn = int((flag & y).sum()), int((~flag & y).sum())
    fp, tn = int((flag & ~y).sum()), int((~flag & ~y).sum())
    return dict(n=len(rows), spoofed=int(y.sum()), tp=tp, fn=fn, fp=fp, tn=tn,
                tpr=tp / (tp + fn) if tp + fn else None, fpr=fp / (fp + tn) if fp + tn else None,
                auroc=roc_auc([r["score_ratio"] for r in rows], y),
                downstream=len(down), downstream_flagged=sum(r["verdict"] == "invalid" for r in down))


def fmt(v, spec=".3f"):
    return "  -  " if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="Write the full report as JSON")
    args = ap.parse_args()

    cb = SLCaseBase.load(args.case_base)
    report = {"case_base": args.case_base, "runs": {}}
    all_rows = []
    for run_dir in args.runs:
        run = load_run(run_dir)
        cb.check_compatible(run["names"], run["probe_hash"])
        tag = os.path.basename(os.path.normpath(run_dir))
        attack = run["meta"].get("attack") or "clean"
        rows = []
        for m in run["models"]:
            # compare_round: training progress the server gated at (= round unless rounds were rejected)
            d = cb.decide(run["Z"][run["row_model"] == m["model"]], m.get("compare_round", m["round"]))
            rows.append(dict(run=tag, attack=attack, model=m["model"], round=m["round"], kind=m["kind"],
                             label=m["label"], spoofed=m["spoofed"], verdict=d["verdict"], score_ratio=d["score_ratio"],
                             coverage=d["coverage"], live_verdict=m.get("verdict"),
                             live_accepted=m.get("accepted"), reasons=d["reasons"]))
        all_rows += rows
        report["runs"][tag] = dict(meta=run["meta"], models=rows, summary=rates(rows))

        print(f"\n== {tag}  (attack: {attack}"
              + (f" by {run['meta']['malicious_client']}" if attack != "clean" else "") + ")")
        print(f"   {'model':<16} {'truth':<10} {'verdict':<8} {'score/thr':>9} {'coverage':>9}  live")
        for r in rows:
            live = "" if r["live_verdict"] is None else f"{r['live_verdict']}{'' if r['live_accepted'] else ' (rejected)'}"
            mark = "" if r["label"] == "downstream" or (r["verdict"] == "invalid") == r["spoofed"] else "  <-- wrong"
            print(f"   {r['model']:<16} {r['label']:<10} {r['verdict']:<8} "
                  f"{r['score_ratio']:>9.2f} {r['coverage']:>9.3f}  {live}{mark}")

    print("\n== Summary")
    print(f"   {'slice':<34} {'n':>4} {'spoofed':>7} {'TPR':>6} {'FPR':>6} {'AUROC':>6}  downstream flagged")
    slices = {"all models": all_rows}
    for kind in ("client_update", "checkpoint"):
        slices[kind] = [r for r in all_rows if r["kind"] == kind]
    for attack in sorted({r["attack"] for r in all_rows}):
        for kind in ("client_update", "checkpoint"):
            slices[f"{attack} / {kind}"] = [r for r in all_rows if r["attack"] == attack and r["kind"] == kind]
    report["summary"] = {}
    for name, rows in slices.items():
        if not rows:
            continue
        s = rates(rows)
        report["summary"][name] = s
        print(f"   {name:<34} {s['n']:>4} {s['spoofed']:>7} {fmt(s['tpr'], '.2f'):>6} "
              f"{fmt(s['fpr'], '.2f'):>6} {fmt(s['auroc'], '.2f'):>6}  "
              + (f"{s['downstream_flagged']}/{s['downstream']}" if s['downstream'] else ""))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
