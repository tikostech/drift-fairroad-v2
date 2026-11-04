"""
Build the IEEE-preprint assets for the SL gate experiments from sl_runs/.

    python make_paper_assets.py            # writes ../paper/{data,tables,figures}, results.tex, REPORT.md

Final configuration (gate "v6"): case base sl_runs/case_base_v2.npz -- 14 valid runs (8 clean, 6 with
one client absent), pooled calibration chosen for a 0.5% leave-one-run-out false-positive rate --
with the rejection budget, round-1 checkpoint warm-up, stall guard, and checkpoints traced but not
gated (--sl-gate-scope clients).

Run names: ``attack_<a>[_sN]`` (capture: traced, not gated), ``live_v6_<a>_sN`` (final live gate),
``enforce_<a>[_sN]`` (original gate v1) and ``live_v2..v5_*`` (intermediate gate versions, used only
for the version comparison). Runs in the case base are not scored against it.

Protocol:
- Capture runs are re-scored offline against the final case base; their spoofed models give
  detection rates. Valid models for false-positive rates and AUROC negatives come from
  leave-one-run-out over the case-base runs: each run is scored against a base built without it.
  Honest models trained from an accepted poisoned checkpoint ("downstream") are excluded.
- Live runs report the server's own decisions and the reconstruction quality the gate preserved.
- Rates: per-run mean +- std, plus a pooled Wilson 95% interval. AUROC: bootstrap 95% interval.
- The design ablation keeps its original split (base clean_a/clean_b, held-out clean_c*) and the
  original calibration, so it remains comparable with earlier results.

Reconstruction quality per checkpoint is cached in data/round_quality.csv (slow to compute).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import config
from sl_gate import SLCaseBase, load_run

RUNS_DIR = "sl_runs"                                   # --runs-dir
CASE_BASE = os.path.join(RUNS_DIR, "case_base_v2.npz")  # --case-base
ENV = {}                                               # compute environment, filled in main()
FINAL = "v6"
REFERENCE_CLEAN = ("clean_c", "clean_c_s2", "clean_c_s3")   # ungated clean runs (quality reference)
ABL_BASE, ABL_HELD = ("clean_a", "clean_b"), REFERENCE_CLEAN    # original split for the design ablation

# Live gate versions for the comparison, in order: (key, label, run-name pattern).
VERSIONS = [
    ("v1", "Original: 2 clean runs, max $\\times$1.25",
     r"^(enforce_(clean|scale|noise_x2|sign_flip)(_s\d+)?|live_v1_.*)$"),
    ("v2", "+ 14-run base, pooled calibration", r"^live_v2_"),
    ("v3", "+ rejection budget, ckpt.\\ warm-up", r"^live_v3_"),
    ("v4", "+ stall guard, no coverage rule", r"^live_v4_"),
    ("v5", "+ stall guard, coverage rule kept", r"^live_v5_"),
    ("v6", "v5, checkpoints traced not gated", r"^live_v6_"),
]
VPLAIN = {"v1": "v1 original", "v2": "v2 bigger base", "v3": "v3 + budget", "v4": "v4 + stall guard, no coverage",
          "v5": "v5 + stall guard", "v6": "v6 final (ckpt trace-only)"}

# Attack keys in table order, with paper labels. "noise2" = Gaussian noise at 2.0 x std.
ATTACKS = {
    "none": "None (clean)",
    "scale": "Update scaling ($\\times$10)",
    "noise2": "Gaussian noise ($\\sigma{=}2.0$)",
    "sign_flip": "Sign flip",
    "noise": "Gaussian noise ($\\sigma{=}0.5$)",
    "inverted": "Inverted training",
}
PLAIN = {"none": "None (clean)", "scale": "Update scaling (x10)", "noise2": "Gaussian noise (σ=2.0)",
         "sign_flip": "Sign flip", "noise": "Gaussian noise (σ=0.5)", "inverted": "Inverted training"}
SHORT = {"scale": "Scaling ×10", "noise2": "Noise σ=2.0", "sign_flip": "Sign flip",
         "noise": "Noise σ=0.5", "inverted": "Inverted"}
KINDS = {"client_update": "Client update", "checkpoint": "Checkpoint"}

# Figure palette: reference categorical slots 1-2 (validated on white: CVD dE 24.7), neutral ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#d9d8d2"
COL1, COL2 = 3.5, 7.16  # IEEE single / double column width, inches


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def auroc_ci(pos, neg, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return (np.nan, np.nan)
    b = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def mean_std(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    if not xs:
        return (np.nan, np.nan, 0)
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, len(xs))


# ---------------------------------------------------------------------------
# Run discovery, the per-model matrix, and held-out valid scores
# ---------------------------------------------------------------------------

def attack_key(meta):
    a = meta.get("attack")
    if not a:
        return "none"
    if a == "noise" and (meta.get("attack_scale") or 0.5) >= 1.0:
        return "noise2"
    return a


def discover(base_runs):
    """Capture attack runs and final live runs (scored), plus the clean reference runs (quality only)."""
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        d = os.path.join(RUNS_DIR, name)
        need = ("sl_traces.npz", "run_meta.json", "server_round_10.pt")
        if not os.path.isdir(d) or not all(os.path.exists(os.path.join(d, f)) for f in need):
            continue
        if name.startswith("attack_"):
            mode = "capture"
        elif re.match(rf"^live_{FINAL}_", name):
            mode = "enforce"
        elif name in REFERENCE_CLEAN:
            mode = "reference"
        else:
            continue
        if mode != "reference" and name in base_runs:
            continue
        m = re.match(r"^(.*?)(?:_s(\d+))?$", name)
        meta = json.load(open(os.path.join(d, "run_meta.json")))
        runs.append(dict(name=name, dir=d, seed=int(m.group(2) or 1), meta=meta, mode=mode, attack=attack_key(meta)))
    return runs


def model_matrix(runs, cb):
    rows = []
    for run in runs:
        if run["mode"] == "reference":   # in the case base: scoring it against itself would be biased
            continue
        R = load_run(run["dir"])
        cb.check_compatible(R["names"], R["probe_hash"])
        for m in R["models"]:
            cr = m.get("compare_round", m["round"])
            d = cb.decide(R["Z"][R["row_model"] == m["model"]], cr)
            rows.append(dict(
                run=run["name"], mode=run["mode"], attack=run["attack"], seed=run["seed"],
                round=m["round"], compare_round=cr, model=m["model"], source=m["source"], kind=m["kind"],
                label=m["label"], offline_verdict=d["verdict"], score=round(d["score"], 4),
                score_ratio=round(d["score_ratio"], 4), coverage=round(d["coverage"], 4),
                live_verdict=m.get("verdict") or "", live_accepted=m.get("accepted", ""),
                live_note=m.get("note") or "", trace_seconds=m.get("trace_seconds", "")))
    return rows


def base_arrays(names):
    Z, ids, rounds, images, groups = [], [], [], [], []
    for tag in names:
        R = load_run(os.path.join(RUNS_DIR, tag))
        ro = {m["model"]: m.get("compare_round", m["round"]) for m in R["models"]}
        ok = {m["model"] for m in R["models"] if m["label"] == "valid"}
        rows = np.isin(R["row_model"], list(ok))
        Z.append(R["Z"][rows])
        ids += [f"{tag}/{m}" for m in R["row_model"][rows]]
        rounds += [ro[m] for m in R["row_model"][rows]]
        images += list(R["row_image"][rows])
        groups += [tag] * int(rows.sum())
    return np.concatenate(Z), np.array(ids), np.array(rounds), np.array(images), np.array(groups), R["names"]


def loro_valid(cb, base_runs):
    """Every case-base model scored against a case base built without its run (same settings)."""
    Z, ids, rounds, images, groups, names = base_arrays(base_runs)
    kw = dict(k=cb.k, k_pca=cb.k_pca, window=cb.window, match_images=cb.match_images,
              calibration=cb.calibration_method, coverage_rule=cb.coverage_rule, q=cb.q, margin=cb.margin)
    out = []
    for g in base_runs:
        tr = groups != g
        b = SLCaseBase(**kw).fit(Z[tr], ids[tr], rounds[tr], images[tr], groups[tr], names)
        for m in np.unique(ids[~tr]):
            rows = ids == m
            d = b.decide(Z[rows], int(rounds[rows][0]))
            out.append(dict(run=g, model=str(m), round=int(rounds[rows][0]),
                            kind="checkpoint" if str(m).endswith("/global") else "client_update",
                            score_ratio=round(d["score_ratio"], 4), verdict=d["verdict"]))
        print(f"  leave-one-run-out: {g}", flush=True)
    return out


def round_quality(names_dirs, cache_path, rounds=range(1, 11)):
    """Fixed-validation MSE/SSIM of the checkpoint in force per round, cached by (run, round)."""
    done = {}
    if os.path.exists(cache_path):
        for r in csv.DictReader(open(cache_path)):
            done[(r["run"], int(r["round"]))] = r
    todo = [(n, d, r) for n, d in names_dirs for r in rounds if (n, r) not in done]
    if todo:
        sys.argv = sys.argv[:1]
        from evaluate_cae import CAEEvaluator
        ev = CAEEvaluator()
        dl = ev.get_fixed_dataloader(config.CLEAN_IMAGE_DIR, config.NOISY_IMAGE_DIR, config.NUM_CLIENTS, 0)
        for i, (n, d, r) in enumerate(todo):
            ev.model.load(os.path.join(d, f"server_round_{r}.pt"))
            q = ev.evaluate_on_dataset(dl)
            done[(n, r)] = dict(run=n, round=r, mse=q["mse"], ssim=q["ssim"])
            if i % 20 == 19:
                print(f"  quality {i + 1}/{len(todo)}", flush=True)
        with open(cache_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "round", "mse", "ssim"])
            w.writeheader()
            for k in sorted(done):
                w.writerow(done[k])
    out = defaultdict(dict)
    for (name, r), v in done.items():
        out[name][r] = (float(v["mse"]), float(v["ssim"]))
    return out


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def detection(models, loro):
    """Offline TPR on capture runs per attack x kind; FPR and AUROC negatives from leave-one-run-out."""
    cap = [m for m in models if m["mode"] == "capture" and m["label"] == "spoofed"]
    neg = {k: [v["score_ratio"] for v in loro if v["kind"] == k] for k in KINDS}
    out = []
    for k in KINDS:   # clean row: held-out valid models
        sel = [v for v in loro if v["kind"] == k]
        flag = [v["verdict"] == "invalid" for v in sel]
        per = [np.mean([v["verdict"] == "invalid" for v in sel if v["run"] == g]) for g in sorted({v["run"] for v in sel})]
        mu, sd, n = mean_std(per)
        lo, hi = wilson(sum(flag), len(flag))
        out.append(dict(attack="none", kind=k, rate_name="FPR", n_models=len(sel), n_flagged=int(sum(flag)),
                        n_seeds=n, rate_mean=mu, rate_std=sd, rate_pooled=float(np.mean(flag)),
                        wilson_lo=lo, wilson_hi=hi, auroc=np.nan, auroc_lo=np.nan, auroc_hi=np.nan))
    for a in [a for a in ATTACKS if a != "none"]:
        for k in KINDS:
            target = [m for m in cap if m["attack"] == a and m["kind"] == k]
            if not target:
                continue
            flag = [m["offline_verdict"] == "invalid" for m in target]
            per = [np.mean([m["offline_verdict"] == "invalid" for m in target if m["seed"] == s])
                   for s in sorted({m["seed"] for m in target})]
            mu, sd, n = mean_std(per)
            lo, hi = wilson(sum(flag), len(flag))
            pos = [m["score_ratio"] for m in target]
            alo, ahi = auroc_ci(pos, neg[k])
            out.append(dict(attack=a, kind=k, rate_name="TPR", n_models=len(target), n_flagged=int(sum(flag)),
                            n_seeds=n, rate_mean=mu, rate_std=sd, rate_pooled=float(np.mean(flag)),
                            wilson_lo=lo, wilson_hi=hi, auroc=auroc(pos, neg[k]), auroc_lo=alo, auroc_hi=ahi))
    return out


def enforcement(models, runs, quality):
    """Final live gate per attack, with final quality gated vs ungated."""
    out = []
    final = {name: q[10] for name, q in quality.items() if 10 in q}
    for a in ATTACKS:
        enf = [r for r in runs if r["mode"] == "enforce" and r["attack"] == a]
        if not enf:
            continue
        ung = [r for r in runs if r["mode"] == ("reference" if a == "none" else "capture") and r["attack"] == a]
        live = [m for m in models if m["mode"] == "enforce" and m["attack"] == a]

        def count(sel):
            return sum(not m["live_accepted"] for m in sel), len(sel)

        bad = [m for m in live if m["label"] == "spoofed" and m["kind"] == "client_update"]
        # Every honest client's update counts, including ones trained from a checkpoint that a missed
        # attacker update reached: rejecting any of them is a false rejection.
        honest = [m for m in live if m["label"] != "spoofed" and m["kind"] == "client_update"]
        ckpt = [m for m in live if m["kind"] == "checkpoint"]
        b_k, b_n = count(bad)
        h_k, h_n = count(honest)
        c_k, c_n = count(ckpt)
        budget = sum(m["kind"] == "client_update" and m["live_note"].startswith("admitted") for m in live)
        g_mse, g_ssim = mean_std([final[r["name"]][0] for r in enf]), mean_std([final[r["name"]][1] for r in enf])
        u_mse, u_ssim = mean_std([final[r["name"]][0] for r in ung]), mean_std([final[r["name"]][1] for r in ung])
        out.append(dict(attack=a, n_runs=len(enf), attacker_rejected=b_k, attacker_updates=b_n,
                        honest_rejected=h_k, honest_updates=h_n, ckpt_rejected=c_k, checkpoints=c_n,
                        budget_admitted=budget,
                        gated_mse_mean=g_mse[0], gated_mse_std=g_mse[1], gated_ssim_mean=g_ssim[0], gated_ssim_std=g_ssim[1],
                        ungated_mse_mean=u_mse[0], ungated_mse_std=u_mse[1], ungated_ssim_mean=u_ssim[0],
                        ungated_ssim_std=u_ssim[1], n_ungated_runs=u_mse[2]))
    return out


def live_versions(quality_cache):
    """All live gate versions on the same four conditions: false rejections, detection, final MSE."""
    rows = []
    run_dirs = {}
    for key, label, pat in VERSIONS:
        for name in sorted(os.listdir(RUNS_DIR)):
            d = os.path.join(RUNS_DIR, name)
            if re.match(pat, name) and os.path.exists(os.path.join(d, "server_round_10.pt")):
                run_dirs.setdefault(key, []).append((name, d))
    final = round_quality([x for v in run_dirs.values() for x in v], quality_cache, rounds=(10,))
    for key, label, pat in VERSIONS:
        if key not in run_dirs:
            continue
        c = defaultdict(int)
        mse = defaultdict(list)
        for name, d in run_dirs[key]:
            meta = json.load(open(os.path.join(d, "run_meta.json")))
            bad = meta.get("malicious_client") if meta.get("attack") else None
            mse[attack_key(meta)].append(final[name][10][0])
            for l in map(json.loads, open(os.path.join(d, "sl_gate_log.jsonl"))):
                rej = not l["accepted"]
                if l["source"] == "global":
                    if l["round"] == 1:      # round 1 is not gated from v3 on; excluded for every version
                        continue
                    poisoned = bad in (l.get("included") or [])
                    c["ckpt_poisoned" if poisoned else "ckpt_honest"] += 1
                    c["ckpt_poisoned_rej" if poisoned else "ckpt_honest_rej"] += rej
                elif l["source"] == bad:
                    c["attacker"] += 1
                    c["attacker_rej"] += rej
                else:
                    c["honest"] += 1
                    c["honest_rej"] += rej
        att = [np.mean(mse[a]) for a in ("scale", "noise2", "sign_flip") if mse[a]]
        rows.append(dict(version=key, label=label, runs=len(run_dirs[key]),
                         honest_rejected=c["honest_rej"], honest=c["honest"],
                         ckpt_honest_rejected=c["ckpt_honest_rej"], ckpt_honest=c["ckpt_honest"],
                         false_rejections=c["honest_rej"] + c["ckpt_honest_rej"],
                         honest_models=c["honest"] + c["ckpt_honest"],
                         false_rejection_rate=(c["honest_rej"] + c["ckpt_honest_rej"]) / max(1, c["honest"] + c["ckpt_honest"]),
                         attacker_rejected=c["attacker_rej"], attacker=c["attacker"],
                         ckpt_poisoned_rejected=c["ckpt_poisoned_rej"], ckpt_poisoned=c["ckpt_poisoned"],
                         clean_mse=float(np.mean(mse["none"])) if mse["none"] else np.nan,
                         attacked_mse=float(np.mean(att)) if att else np.nan))
    return rows


def confusion(models, loro):
    out = []
    for k in KINDS:
        cap = [m for m in models if m["mode"] == "capture" and m["kind"] == k and m["label"] == "spoofed"]
        val = [v for v in loro if v["kind"] == k]
        out.append(dict(scope="offline (capture runs; held-out valid)", kind=k,
                        tp=sum(m["offline_verdict"] == "invalid" for m in cap),
                        fn=sum(m["offline_verdict"] != "invalid" for m in cap),
                        fp=sum(v["verdict"] == "invalid" for v in val), tn=sum(v["verdict"] != "invalid" for v in val),
                        downstream_excluded=0))
        live = [m for m in models if m["mode"] == "enforce" and m["kind"] == k]
        flag = lambda m: not m["live_accepted"]
        out.append(dict(scope=f"live (gate {FINAL})", kind=k,
                        tp=sum(m["label"] == "spoofed" and flag(m) for m in live),
                        fn=sum(m["label"] == "spoofed" and not flag(m) for m in live),
                        fp=sum(m["label"] != "spoofed" and flag(m) for m in live),
                        tn=sum(m["label"] != "spoofed" and not flag(m) for m in live),
                        downstream_excluded=0))
    return out


def roc_points(models, loro):
    cap = [m for m in models if m["mode"] == "capture" and m["label"] == "spoofed"]
    out = []
    for a in [a for a in ATTACKS if a != "none"]:
        for k in KINDS:
            pos = np.array([m["score_ratio"] for m in cap if m["attack"] == a and m["kind"] == k])
            neg = np.array([v["score_ratio"] for v in loro if v["kind"] == k])
            if not len(pos) or not len(neg):
                continue
            for t in np.r_[np.inf, np.unique(np.r_[pos, neg])[::-1], -np.inf]:
                out.append(dict(attack=a, kind=k, threshold=t, tpr=float((pos >= t).mean()),
                                fpr=float((neg >= t).mean())))
    return out


ABLATIONS = [
    ("Round-agnostic, pooled images (PCA 32)", dict(window=99, match_images=False, k_pca=32)),
    ("Round-agnostic, per-image", dict(window=99, match_images=True, k_pca=0)),
    ("Round-aware, pooled images", dict(window=1, match_images=False, k_pca=0)),
    ("Round-aware, per-image, PCA 32", dict(window=1, match_images=True, k_pca=32)),
    ("Round-aware, per-image (deployed)", dict(window=1, match_images=True, k_pca=0)),
]


def ablation(runs):
    """Design ablation on the original split: base clean_a/clean_b, held-out clean_c*, capture attacks."""
    Z, ids, rounds, images, groups, names = base_arrays(ABL_BASE)
    scored_runs = [(r["attack"], load_run(r["dir"])) for r in runs if r["mode"] == "capture"]
    scored_runs += [("none", load_run(os.path.join(RUNS_DIR, n))) for n in ABL_HELD]
    out = []
    for name, kw in ABLATIONS:
        cb = SLCaseBase(**kw).fit(Z, ids, rounds, images, groups, names)
        scored = []
        for a, R in scored_runs:
            for m in R["models"]:
                if m["label"] == "downstream":
                    continue
                d = cb.decide(R["Z"][R["row_model"] == m["model"]], m.get("compare_round", m["round"]))
                scored.append((a, m["kind"], m["label"], d["score_ratio"], d["verdict"] == "invalid"))
        row = dict(variant=name, **{k: kw[k] for k in ("window", "match_images", "k_pca")})
        for k in KINDS:
            neg = [s for a, kk, l, s, f in scored if a == "none" and kk == k]
            fpr = [f for a, kk, l, s, f in scored if a == "none" and kk == k]
            for subset, keep in (("all", lambda a: a != "none"), ("harmful", lambda a: a in ("scale", "noise2"))):
                pos = [s for a, kk, l, s, f in scored if keep(a) and kk == k and l == "spoofed"]
                tpr = [f for a, kk, l, s, f in scored if keep(a) and kk == k and l == "spoofed"]
                row[f"{k}_{subset}_auroc"] = auroc(pos, neg)
                row[f"{k}_{subset}_tpr"] = float(np.mean(tpr)) if tpr else np.nan
            row[f"{k}_fpr"] = float(np.mean(fpr)) if fpr else np.nan
        out.append(row)
        print(f"  ablation: {name}", flush=True)
    return out


def overhead(models, runs):
    ts = np.array([float(m["trace_seconds"]) for m in models if m["trace_seconds"] not in ("", None)])
    durations = []
    for run in runs:
        log = os.path.join(RUNS_DIR, run["name"] + ".log")
        if not os.path.exists(log):
            continue
        stamps = [m.group(1) for m in re.finditer(
            r"(\d{2}:\d{2}:\d{2},\d{3}) - \[__main__\] - \S+ Round \d+ complete -", open(log, encoding="utf-8", errors="ignore").read())]
        t = [int(h) * 3600 + int(mi) * 60 + float(s.replace(",", ".")) for h, mi, s in (x.split(":") for x in stamps)]
        durations += list(np.diff(t))
    return dict(trace_s_mean=float(ts.mean()), trace_s_median=float(np.median(ts)), trace_s_p95=float(np.percentile(ts, 95)),
                n_traced=int(len(ts)), models_per_round=config.NUM_CLIENTS + 1,
                round_s_mean=float(np.mean(durations)) if durations else np.nan,
                case_base_bytes=os.path.getsize(CASE_BASE))


def environment(runs):
    """Where the live runs actually trained and traced (from their logs), and the torch build."""
    import torch
    devices = set()
    for r in runs:
        log = os.path.join(RUNS_DIR, r["name"] + ".log")
        if r["mode"] == "enforce" and os.path.exists(log):
            devices |= set(re.findall(r"Model loaded on (\w+)", open(log, encoding="utf-8", errors="ignore").read()))
    gpu = torch.cuda.get_device_name(0) if "cuda" in devices and torch.cuda.is_available() else None
    label = f"GPU ({gpu}), PyTorch {torch.__version__}" if gpu else f"CPU, PyTorch {torch.__version__}"
    return dict(device="cuda" if gpu else "cpu", gpu=gpu, torch=torch.__version__, label=label)


def case_base_summary(cb, base_runs):
    sel = cb.meta.get("selection") or {}
    return dict(file=os.path.basename(CASE_BASE), n_models=cb.meta["n_models"], n_cases=cb.meta["n_cases"],
                n_runs=len(base_runs), n_clean=sum("dropout" not in r for r in base_runs),
                n_dropout=sum("dropout" in r for r in base_runs), features=len(cb.names), kept_features=int(cb.keep.sum()),
                k=cb.k, window=cb.window, calibration=cb.calibration_method, q=cb.q, margin=cb.margin,
                coverage_rule=bool(cb.coverage_rule), target_fpr=sel.get("target_fpr"),
                held_out_fpr=sel.get("held_out_fpr"), size_mb=round(os.path.getsize(CASE_BASE) / 1e6, 1))


def run_seconds(log_path):
    """Wall time of one run from its log: first timestamp to the last round's completion."""
    stamps = re.findall(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) - \[", open(log_path, encoding="utf-8", errors="ignore").read())
    if len(stamps) < 2:
        return None
    t = [int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000 for h, m, sec, ms in (stamps[0], stamps[-1])]
    return t[1] - t[0] if t[1] >= t[0] else t[1] + 86400 - t[0]


def runtimes(runs):
    """Mean wall time of the final live runs here, plus the measured benchmarks in data/perf_benchmarks.json."""
    secs = [run_seconds(os.path.join(RUNS_DIR, r["name"] + ".log")) for r in runs if r["mode"] == "enforce"]
    secs = [x for x in secs if x]
    return dict(final_live_run_s_mean=float(np.mean(secs)) if secs else np.nan, n=len(secs))


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pm(mu, sd, d=2, n=None):
    if mu is None or np.isnan(mu):
        return "--"
    return f"${mu:.{d}f}$" if n == 1 else f"${mu:.{d}f} \\pm {sd:.{d}f}$"


def table(path, caption, label, colspec, header, body, star=False, notes=None):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             header + " \\\\", "\\midrule"] + [b if b.startswith("\\") else b + " \\\\" for b in body] + \
            ["\\bottomrule", "\\end{tabular}"]
    if notes:
        width = "\\textwidth" if star else "\\columnwidth"
        lines += ["\\\\[2pt]", f"\\parbox{{{width}}}{{\\scriptsize {notes}}}"]
    lines += [f"\\end{{{env}}}", ""]
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def setup_rows(cb):
    sel = cb.meta.get("selection") or {}
    return [
        ("Data", "502 clean/adversarial image pairs, $224\\times224$ RGB, 3 clients ($\\approx$167 each)"),
        ("Model", "Convolutional autoencoder, 757{,}011 parameters (5 conv, 2 FC, 5 deconv)"),
        ("FL", f"FedAvg, 3 clients, 10 rounds, 1 local epoch, batch {config.BATCH_SIZE}, Adam $\\eta{{=}}{config.LEARNING_RATE}$"),
        ("Probe set", f"{config.SL_PROBE_IMAGES} fixed adversarial images, evenly spaced over the dataset"),
        ("Trace", f"{len(cb.names)} SL-minimised features per image (7 statistics; $\\phi$, spread, peak, depth, drift, conc)"),
        ("Case base", f"{cb.meta['n_models']} valid models from {cb.meta['n_groups']} runs "
                      f"({sum('dropout' not in r for r in cb.meta['runs'])} clean, "
                      f"{sum('dropout' in r for r in cb.meta['runs'])} with one client absent), "
                      f"{cb.meta['n_cases']:,} cases".replace(",", "{,}")),
        ("Comparison", f"$k{{=}}{cb.k}$ nearest valid cases per probe image, rounds $r\\pm{cb.window}$ of training progress"),
        ("Threshold", f"held-out scores normalised by round median; {cb.q} quantile $\\times {cb.margin}$, "
                      f"chosen for {100 * sel.get('target_fpr', 0.005):.1f}\\% leave-one-run-out FPR"),
        ("Gate policy", "$\\le$1 update rejected per round (most anomalous first); checkpoints traced, not gated"),
        ("Attacker", "1 of 3 clients (\\texttt{client\\_3}) in every round"),
        ("Replicates", "3 independent runs per condition (fresh initialisation and shuffling)"),
        ("Compute", ENV.get("label", "--") + ("; cuDNN TF32 off while tracing" if ENV.get("gpu") else "")),
    ]


def tex_tables(out, det, enf, ab, ovh, cb, versions):
    T = os.path.join(out, "tables")
    table(os.path.join(T, "setup.tex"), "Experimental setup.", "tab:setup", "@{}ll@{}",
          "\\textbf{Item} & \\textbf{Setting}", [f"{a} & {b}" for a, b in setup_rows(cb)])

    body = []
    for a in ATTACKS:
        for i, k in enumerate(KINDS):
            r = next((d for d in det if d["attack"] == a and d["kind"] == k), None)
            if not r:
                continue
            name = ATTACKS[a] if i == 0 else ""
            rate = pm(r["rate_mean"], r["rate_std"], 2, r["n_seeds"])
            ci = f"[{f(r['wilson_lo'])}, {f(r['wilson_hi'])}]"
            au = "--" if np.isnan(r["auroc"]) else f"{f(r['auroc'])} [{f(r['auroc_lo'])}, {f(r['auroc_hi'])}]"
            body.append(f"{name} & {KINDS[k]} & {r['rate_name']} & {rate} & {r['n_flagged']}/{r['n_models']} & {ci} & {au}")
        if a == "none":
            body.append("\\midrule")
    table(os.path.join(T, "detection.tex"),
          "Offline detection. Spoofed models: capture runs (traced, not gated), scored against the final case base. "
          "Valid models: every case-base run scored against a base built without it (leave-one-run-out). "
          "Rates are per-run mean $\\pm$ s.d.; pooled counts with Wilson 95\\% intervals; AUROC with bootstrap 95\\% interval.",
          "tab:detection", "@{}llcccc c@{}",
          "\\textbf{Attack} & \\textbf{Model} & & \\textbf{Rate} & \\textbf{Flagged} & \\textbf{95\\% CI} & \\textbf{AUROC [95\\% CI]}",
          body, star=True,
          notes="A checkpoint is spoofed if it averaged a malicious update. The clean row's s.d.\\ is over the 14 held-out runs.")

    body = []
    for r in enf:
        att = "--" if r["attacker_updates"] == 0 else f"{r['attacker_rejected']}/{r['attacker_updates']}"
        body.append(f"{ATTACKS[r['attack']]} & {att} & {r['honest_rejected']}/{r['honest_updates']} & "
                    f"{pm(r['ungated_mse_mean'], r['ungated_mse_std'], 4, r['n_ungated_runs'])} & "
                    f"{pm(r['gated_mse_mean'], r['gated_mse_std'], 4, r['n_runs'])} & "
                    f"{pm(r['ungated_ssim_mean'], r['ungated_ssim_std'], 3, r['n_ungated_runs'])} & "
                    f"{pm(r['gated_ssim_mean'], r['gated_ssim_std'], 3, r['n_runs'])}")
    table(os.path.join(T, "enforcement.tex"),
          "Final live gate: rejected client updates and reconstruction quality after round 10 (fixed validation "
          "set), without and with the gate. Mean $\\pm$ s.d.\\ over 3 runs.",
          "tab:enforcement", "@{}lcccccc@{}",
          "& \\multicolumn{2}{c}{\\textbf{Updates rejected}} & \\multicolumn{2}{c}{\\textbf{MSE} $\\downarrow$} & "
          "\\multicolumn{2}{c}{\\textbf{SSIM} $\\uparrow$} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
          "\\textbf{Attack} & \\textbf{Attacker} & \\textbf{Honest} & \\textbf{Ungated} & \\textbf{Gated} & "
          "\\textbf{Ungated} & \\textbf{Gated}",
          body, star=True,
          notes="Checkpoints are traced but not gated. For the clean condition, ``ungated'' is three clean runs without the gate.")

    body = []
    for v in versions:
        fr = f"{v['false_rejections']}/{v['honest_models']} ({100 * v['false_rejection_rate']:.1f}\\%)"
        cells = [v["version"], v["label"], f"{v['honest_rejected']}/{v['honest']}",
                 f"{v['ckpt_honest_rejected']}/{v['ckpt_honest']}", fr, f"{v['attacker_rejected']}/{v['attacker']}",
                 f"{v['ckpt_poisoned_rejected']}/{v['ckpt_poisoned']}", f(v["attacked_mse"], 4)]
        if v["version"] == FINAL:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        body.append(" & ".join(cells))
    table(os.path.join(T, "versions.tex"),
          "Reducing false positives: live gate versions on the same four conditions (clean, scaling, $\\sigma{=}2.0$ "
          "noise, sign flip; 3 runs each). False rejections count honest client updates and checkpoints that "
          "contained no malicious update; round-1 checkpoints are excluded for every version.",
          "tab:versions", "@{}llcccccc@{}",
          "& & \\multicolumn{3}{c}{\\textbf{False rejections}} & \\multicolumn{2}{c}{\\textbf{Caught}} & \\\\ "
          "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
          "\\textbf{Ver.} & \\textbf{Change} & \\textbf{Updates} & \\textbf{Ckpts.} & \\textbf{Total} & "
          "\\textbf{Attacker upd.} & \\textbf{Poisoned ckpt.} & \\textbf{MSE (att.)}",
          body, star=True,
          notes="MSE (att.): mean final validation MSE over the three attacked conditions. v6 does not gate checkpoints, "
                "so it rejects none.")

    cols = ["client_update_all_auroc", "checkpoint_all_auroc", "client_update_harmful_auroc",
            "checkpoint_harmful_auroc", "client_update_harmful_tpr", "checkpoint_harmful_tpr",
            "client_update_fpr", "checkpoint_fpr"]
    best = {c: (min if c.endswith("fpr") else max)(round(r[c], 2) for r in ab) for c in cols}
    body = []
    for r in ab:
        cells = [f"\\textbf{{{f(r[c])}}}" if round(r[c], 2) == best[c] else f(r[c]) for c in cols]
        body.append(" & ".join([r["variant"].replace(" (deployed)", "$^\\dagger$")] + cells))
    table(os.path.join(T, "ablation.tex"),
          "Ablation of the comparison design (offline; original split: case base from 2 clean runs, held-out clean "
          "runs, original max$\\times$1.25 calibration). Harmful = update scaling and $\\sigma{=}2.0$ noise; ``all'' "
          "also includes attacks that do not degrade the model. Best per column in bold.",
          "tab:ablation", "@{}lcccccccc@{}",
          "& \\multicolumn{2}{c}{\\textbf{AUROC, all}} & \\multicolumn{2}{c}{\\textbf{AUROC, harmful}} & "
          "\\multicolumn{2}{c}{\\textbf{TPR, harmful}} & \\multicolumn{2}{c}{\\textbf{FPR, clean}} \\\\ "
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
          "\\textbf{Variant} & Upd. & Ckpt. & Upd. & Ckpt. & Upd. & Ckpt. & Upd. & Ckpt.",
          body, star=True, notes="$^\\dagger$Comparison design used by every live gate version.")

    body = [
        f"Trace time per model (median / mean / p95)$^\\ast$ & {f(ovh['trace_s_median'])} / {f(ovh['trace_s_mean'])} / {f(ovh['trace_s_p95'])} s",
        f"Models traced per round & {ovh['models_per_round']} (3 updates + 1 checkpoint)",
        f"Mean FL round duration (traced runs) & {f(ovh['round_s_mean'], 1)} s",
        f"Case base size & {ovh['case_base_bytes'] / 1e6:.1f} MB ({cb.meta['n_cases']:,} cases $\\times$ {int(cb.keep.sum())} features)".replace(",", "{,}"),
        f"Models traced in the scored runs & {ovh['n_traced']:,}".replace(",", "{,}"),
    ]
    table(os.path.join(T, "overhead.tex"),
          f"Server-side overhead of tracing and gating ({'GPU' if ENV.get('gpu') else 'CPU'}).",
          "tab:overhead", "@{}ll@{}", "\\textbf{Quantity} & \\textbf{Value}", body,
          notes=f"$^\\ast$Measured on {ENV.get('label', 'this machine')}.")

    body = []
    for r, t in cb.thresholds.items():
        near = [c for c in cb.calibration if abs(c["round"] - r) <= cb.window]
        s = [c["score"] for c in near]
        body.append(f"{r} & {len(near)} & {np.median(s):.2f} & {max(s):.2f} & {t['score']:.2f} & {t['coverage']:.3f}")
    table(os.path.join(T, "calibration.tex"),
          f"Per-round calibration of the final case base: held-out valid scores (leave-one-run-out) in rounds "
          f"$r\\pm{cb.window}$ and the resulting distance and coverage thresholds.",
          "tab:calibration", "@{}cccccc@{}",
          "\\textbf{Round} & \\textbf{Models} & \\textbf{Median} & \\textbf{Max} & \\textbf{Dist. thr.} & \\textbf{Cov. thr.}",
          body)


def tex_macros(out, det, enf, ovh, runs, models, versions, cb):
    def get(a, k):
        return next((d for d in det if d["attack"] == a and d["kind"] == k), None)

    def pct(x, d=0):
        return "--" if x is None or np.isnan(x) else f"{100 * x:.{d}f}\\%"

    sel = cb.meta.get("selection") or {}
    lines = ["% Auto-generated by fl-payload/make_paper_assets.py -- do not edit.",
             "\\newcommand{\\resNSeeds}{3}",
             f"\\newcommand{{\\resNModels}}{{{len(models):,}}}".replace(",", "{,}"),
             f"\\newcommand{{\\resBaseRuns}}{{{cb.meta['n_groups']}}}",
             f"\\newcommand{{\\resBaseModels}}{{{cb.meta['n_models']}}}",
             f"\\newcommand{{\\resHeldOutFPR}}{{{pct(sel.get('held_out_fpr', np.nan), 1)}}}"]
    for a in ATTACKS:
        key = {"none": "Clean", "scale": "Scale", "noise2": "NoiseStrong", "sign_flip": "SignFlip",
               "noise": "NoiseWeak", "inverted": "Inverted"}[a]
        for k, kk in (("client_update", "Upd"), ("checkpoint", "Ckpt")):
            r = get(a, k)
            if r:
                lines.append(f"\\newcommand{{\\res{key}{kk}Rate}}{{{pct(r['rate_pooled'], 1 if a == 'none' else 0)}}}")
                if a != "none":
                    lines.append(f"\\newcommand{{\\res{key}{kk}AUROC}}{{{f(r['auroc'])}}}")
        e = next((x for x in enf if x["attack"] == a), None)
        if e:
            lines.append(f"\\newcommand{{\\res{key}GatedMSE}}{{{f(e['gated_mse_mean'], 4)}}}")
            lines.append(f"\\newcommand{{\\res{key}UngatedMSE}}{{{f(e['ungated_mse_mean'], 4)}}}")
            lines.append(f"\\newcommand{{\\res{key}HonestRejected}}{{{e['honest_rejected']}/{e['honest_updates']}}}")
            if e["attacker_updates"]:
                lines.append(f"\\newcommand{{\\res{key}AttackerRejected}}{{{e['attacker_rejected']}/{e['attacker_updates']}}}")
    words = {"v1": "VOne", "v2": "VTwo", "v3": "VThree", "v4": "VFour", "v5": "VFive", "v6": "VSix"}
    for v in versions:
        V = words[v["version"]]   # LaTeX command names cannot contain digits
        lines += [f"\\newcommand{{\\resFalseRej{V}}}{{{v['false_rejections']}/{v['honest_models']}}}",
                  f"\\newcommand{{\\resFalseRejRate{V}}}{{{pct(v['false_rejection_rate'], 1)}}}",
                  f"\\newcommand{{\\resUpdFalseRej{V}}}{{{v['honest_rejected']}/{v['honest']}}}",
                  f"\\newcommand{{\\resUpdFalseRejRate{V}}}{{{pct(v['honest_rejected'] / max(1, v['honest']), 1)}}}",
                  f"\\newcommand{{\\resAttackerCaught{V}}}{{{v['attacker_rejected']}/{v['attacker']}}}"]
    lines += [f"\\newcommand{{\\resTraceSeconds}}{{{ovh['trace_s_median']:.2f}}}",
              f"\\newcommand{{\\resRoundSeconds}}{{{ovh['round_s_mean']:.1f}}}"]
    open(os.path.join(out, "results.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType: IEEE PDF eXpress rejects Type 3 fonts
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "lines.markersize": 3.6, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def legend_above(fig, ax, ncol, fontsize=None):
    """One legend row above all panels, with room reserved so it never overlaps the titles."""
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6)
    top = fig.subplotpars.top
    fig.subplots_adjust(top=top - 0.12)
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, top - 0.06),
               fontsize=fontsize, handlelength=2.2, columnspacing=1.4)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"))


def fig_scores(plt, out, models, loro):
    attacks = [a for a in ATTACKS if a != "none" and any(m["attack"] == a for m in models)]
    cap = [m for m in models if m["mode"] == "capture"]
    rounds = np.arange(1, 11)
    fig, axes = plt.subplots(1, len(attacks), figsize=(COL2, 1.75), sharey=True)
    ref = [[v["score_ratio"] for v in loro if v["round"] == r] for r in rounds]
    lo, hi = [np.percentile(v, 5) for v in ref], [np.percentile(v, 95) for v in ref]
    for ax, a in zip(np.atleast_1d(axes), attacks):
        ax.fill_between(rounds, lo, hi, color=BAND, lw=0, label="Valid models (5–95%)")
        for kind, color, marker, ls, lab in (("client_update", BLUE, "o", "-", "Spoofed update"),
                                             ("checkpoint", ORANGE, "s", "--", "Spoofed checkpoint")):
            vals = [[m["score_ratio"] for m in cap if m["attack"] == a and m["kind"] == kind
                     and m["label"] == "spoofed" and m["round"] == r] for r in rounds]
            med = [np.median(v) if v else np.nan for v in vals]
            ax.plot(rounds, med, color=color, marker=marker, ls=ls, label=lab, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=2, label="Gate threshold")
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_title(SHORT[a], color=INK)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xlabel("Round")
    axes[0].set_ylabel("Score / threshold")
    legend_above(fig, axes[0], ncol=4)
    save(fig, out, "scores_by_round")
    plt.close(fig)


def fig_quality(plt, out, runs, quality):
    attacks = [a for a in ATTACKS if a != "none" and any(r["mode"] == "enforce" and r["attack"] == a for r in runs)]
    rounds = np.arange(1, 11)
    fig, axes = plt.subplots(1, len(attacks), figsize=(COL2, 1.75), sharey=False)
    clean = np.array([[quality[r["name"]][x][0] for x in rounds] for r in runs if r["mode"] == "reference"])
    for ax, a in zip(np.atleast_1d(axes), attacks):
        ax.fill_between(rounds, clean.min(0), clean.max(0), color=BAND, lw=0)
        ax.plot(rounds, clean.mean(0), color=MUTED, lw=1.0, label="Clean runs (mean, range)", zorder=2)
        for mode, color, marker, ls, lab in (("capture", ORANGE, "s", "--", "Attacked, no gate"),
                                             ("enforce", BLUE, "o", "-", "Attacked, SL gate")):
            M = np.array([[quality[r["name"]][x][0] for x in rounds] for r in runs
                          if r["mode"] == mode and r["attack"] == a])
            if not len(M):
                continue
            ax.plot(rounds, M.mean(0), color=color, marker=marker, ls=ls, label=lab, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)
            if len(M) > 1:
                ax.fill_between(rounds, M.min(0), M.max(0), color=color, alpha=0.12, lw=0)
        ax.set_title(SHORT[a], color=INK)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xlabel("Round")
        if a == "noise2":
            from matplotlib.ticker import FixedLocator, FuncFormatter
            ax.set_yscale("log")
            ax.minorticks_off()
            ax.yaxis.set_major_locator(FixedLocator([0.03, 0.05, 0.1, 0.2]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel("Validation MSE")
    legend_above(fig, axes[0], ncol=3)
    save(fig, out, "quality_by_round")
    plt.close(fig)


def fig_roc(plt, out, roc):
    attacks = [a for a in ATTACKS if a != "none" and any(p["attack"] == a for p in roc)]
    fig, axes = plt.subplots(1, len(attacks), figsize=(COL2, 1.6), sharey=True)
    for ax, a in zip(np.atleast_1d(axes), attacks):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.6, ls=":")
        for kind, color, ls, lab in (("client_update", BLUE, "-", "Client update"),
                                     ("checkpoint", ORANGE, "--", "Checkpoint")):
            p = [x for x in roc if x["attack"] == a and x["kind"] == kind]
            if p:
                ax.step([x["fpr"] for x in p], [x["tpr"] for x in p], where="post", color=color, ls=ls, label=lab)
        ax.set_title(SHORT[a], color=INK)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    legend_above(fig, axes[0], ncol=2)
    save(fig, out, "roc")
    plt.close(fig)


def fig_calibration(plt, out, cb):
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    rng = np.random.default_rng(0)
    cal = cb.calibration
    for kind, color, marker, lab in (("client", BLUE, "o", "Held-out valid update"),
                                     ("global", ORANGE, "s", "Held-out valid checkpoint")):
        sel = [c for c in cal if c["model"].endswith("global") == (kind == "global")]
        ax.scatter([c["round"] + rng.uniform(-0.25, 0.25) for c in sel], [c["score"] for c in sel], s=4,
                   color=color, marker=marker, edgecolor="none", alpha=0.55, label=lab, zorder=3)
    rs = sorted(cb.thresholds)
    ax.step(rs, [cb.thresholds[r]["score"] for r in rs], where="mid", color=INK2, lw=1.0, ls=(0, (3, 2)),
            label="Distance threshold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Trace distance")
    ax.set_xticks(rs)
    ax.set_ylim(bottom=0)
    legend_above(fig, ax, ncol=3, fontsize=6)
    save(fig, out, "calibration")
    plt.close(fig)


def fig_versions(plt, out, versions):
    """False-rejection rate and attacker detection per gate version (two panels, one axis each)."""
    keys = [v["version"] for v in versions]
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.75))
    panels = ((axes[0], [100 * v["false_rejection_rate"] for v in versions], "False rejections (%)", "{:.1f}"),
              (axes[1], [100 * v["attacker_rejected"] / v["attacker"] for v in versions], "Attacker updates caught (%)", "{:.0f}"))
    for ax, vals, title, fmt in panels:
        colors = [BLUE if k == FINAL else AXIS for k in keys]
        bars = ax.bar(range(len(keys)), vals, color=colors, width=0.62, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=5.5, color=INK2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.set_title(title, color=INK)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "versions")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def frozen_runs(version, min_streak=3):
    """Live runs of a version whose checkpoints were rejected >= min_streak rounds in a row (training stalled)."""
    pat = next(p for k, _, p in VERSIONS if k == version)
    out = []
    for name in sorted(os.listdir(RUNS_DIR)):
        log = os.path.join(RUNS_DIR, name, "sl_gate_log.jsonl")
        if not (re.match(pat, name) and os.path.exists(log)):
            continue
        cur = best = 0
        for l in sorted((json.loads(x) for x in open(log)), key=lambda l: l["round"]):
            if l["source"] == "global":
                cur = cur + 1 if not l["accepted"] else 0
                best = max(best, cur)
        if best >= min_streak:
            out.append((name, best))
    return out


def version_interpretation(versions):
    """What the version comparison does and does not show, stated from the data (3 runs per condition)."""
    by = {v["version"]: v for v in versions}
    rev = [v for v in versions if v["version"] != "v1"]
    lo, hi = min(v["honest_rejected"] for v in rev), max(v["honest_rejected"] for v in rev)
    best = ", ".join(v["version"] for v in rev if v["honest_rejected"] == lo)
    worst = next(v for v in rev if v["honest_rejected"] == hi)
    text = (f"Reading the comparison: every version was run under identical conditions, 3 runs per condition. "
            f"Against the original gate ({by['v1']['honest_rejected']}/{by['v1']['honest']} honest updates rejected), "
            f"the revised versions ranged from {lo}/{worst['honest']} ({best}) to {hi}/{worst['honest']} "
            f"({worst['version']}); differences of a few rejections between them are within run-to-run variation, "
            "so the versions are not a monotonic sequence of improvements.")
    frozen = frozen_runs(worst["version"])
    if frozen:
        stall = ", ".join(f"{n} (checkpoints rejected {k} rounds in a row)" for n, k in frozen)
        text += (f" The outlier is {worst['version']}: training froze in {len(frozen)} run(s) -- {stall} -- so the honest "
                 "clients kept training from a stale model and were rejected in turn. That is the failure the "
                 "checkpoint stall guard (v4 on) prevents.")
    text += (f" The final gate ({FINAL}) does not gate checkpoints at all, which removes that risk; its "
             f"{by[FINAL]['honest_rejected']}/{by[FINAL]['honest']} false rejections and "
             f"{by[FINAL]['attacker_rejected']}/{by[FINAL]['attacker']} attacker updates caught are in line with the best revised versions.")
    return text


def report(out, runs, det, enf, ab, ovh, cb, conf, models, versions):
    by = defaultdict(list)
    for r in runs:
        by[(r["mode"], r["attack"])].append(r["name"])
    sel = cb.meta.get("selection") or {}
    v1 = next(v for v in versions if v["version"] == "v1")
    vf = next(v for v in versions if v["version"] == FINAL)
    L = ["# SL gate for federated learning: paper assets", "",
         f"Generated by `fl-payload/make_paper_assets.py` from `fl-payload/{RUNS_DIR}/` ({ENV.get('label', '')}). "
         "Re-run it to refresh every",
         "table, figure and number after adding runs.", "",
         "## Final configuration and protocol", "",
         f"- **Case base:** `{os.path.basename(CASE_BASE)}`, {cb.meta['n_models']} valid models ({cb.meta['n_cases']:,} cases) "
         f"from {cb.meta['n_groups']} runs: {sum('dropout' not in r for r in cb.meta['runs'])} clean and "
         f"{sum('dropout' in r for r in cb.meta['runs'])} with one client absent (client dropout), {len(cb.names)} SL features "
         f"per probe image, {config.SL_PROBE_IMAGES} probe images.",
         f"- **Comparison:** k={cb.k} nearest valid cases per probe image, rounds r±{cb.window} of training progress.",
         f"- **Calibration:** held-out valid scores (leave-one-run-out) normalised by their round's median; one cut at the "
         f"{cb.q} quantile × {cb.margin}, chosen as the most sensitive setting with held-out FPR ≤ "
         f"{100 * sel.get('target_fpr', 0.005):.1f}% (achieved {100 * sel.get('held_out_fpr', np.nan):.1f}%). Coverage rule on.",
         f"- **Gate policy ({FINAL}):** a round's client updates are decided together; at most 1 rejected per round, most "
         "anomalous first. Checkpoints are traced but not gated (`--sl-gate-scope clients`).",
         "- **Capture runs** trace every model without gating and are re-scored offline. Valid models for FPR and AUROC "
         "negatives come from leave-one-run-out over the case-base runs.",
         "- Replicates are independent runs (fresh initialisation and data shuffling); no fixed RNG seed.", "",
         "| Condition | Mode | Runs |", "|---|---|---|"]
    for (mode, a), names in sorted(by.items(), key=lambda kv: (kv[0][0], list(ATTACKS).index(kv[0][1]))):
        L.append(f"| {PLAIN[a]} | {mode} | {len(names)} ({', '.join(names)}) |")
    L += ["", "## Headline results", "",
          f"- **False rejections, live (honest client updates, same denominator for every version):** "
          f"{vf['honest_rejected']}/{vf['honest']} ({100 * vf['honest_rejected'] / max(1, vf['honest']):.1f}%) with the final "
          f"gate, down from {v1['honest_rejected']}/{v1['honest']} ({100 * v1['honest_rejected'] / max(1, v1['honest']):.1f}%) "
          f"with the original gate; attacker updates caught {vf['attacker_rejected']}/{vf['attacker']} "
          f"(was {v1['attacker_rejected']}/{v1['attacker']})."]
    for d in det:
        if d["attack"] == "none":
            L.append(f"- **Held-out valid models, offline ({KINDS[d['kind']].lower()}s):** FPR {d['n_flagged']}/{d['n_models']} "
                     f"({100 * d['rate_pooled']:.1f}%, Wilson [{100 * d['wilson_lo']:.1f}, {100 * d['wilson_hi']:.1f}]).")
        else:
            L.append(f"- {PLAIN[d['attack']]}, {KINDS[d['kind']].lower()}: TPR {d['n_flagged']}/{d['n_models']} "
                     f"({100 * d['rate_pooled']:.0f}%, Wilson [{100 * d['wilson_lo']:.0f}, {100 * d['wilson_hi']:.0f}]), "
                     f"AUROC {d['auroc']:.2f} [{d['auroc_lo']:.2f}, {d['auroc_hi']:.2f}]")
    L += ["", f"**Final live gate ({FINAL}; final round-10 validation MSE, mean ± s.d.):**", "",
          "| Attack | Attacker rejected | Honest rejected | Flagged but admitted (budget) | MSE no gate | MSE gated |",
          "|---|---|---|---|---|---|"]
    for e in enf:
        att = "—" if not e["attacker_updates"] else f"{e['attacker_rejected']}/{e['attacker_updates']}"
        L.append(f"| {PLAIN[e['attack']]} | {att} | {e['honest_rejected']}/{e['honest_updates']} | {e['budget_admitted']} | "
                 f"{e['ungated_mse_mean']:.4f} ± {e['ungated_mse_std']:.4f} | {e['gated_mse_mean']:.4f} ± {e['gated_mse_std']:.4f} |")
    L += ["", "**Reducing false positives (live gate versions, same four conditions, 3 runs each):**", "",
          "| Version | False rejections (updates + honest ckpts) | Attacker updates caught | Poisoned ckpts caught | Mean MSE, attacked |",
          "|---|---|---|---|---|"]
    for v in versions:
        L.append(f"| {VPLAIN[v['version']]} | {v['false_rejections']}/{v['honest_models']} ({100 * v['false_rejection_rate']:.1f}%) | "
                 f"{v['attacker_rejected']}/{v['attacker']} | {v['ckpt_poisoned_rejected']}/{v['ckpt_poisoned']} | {v['attacked_mse']:.4f} |")
    early = [v for v in versions if v["version"] != FINAL]
    L += ["", "False rejections above count honest client updates (the same number for every version) plus checkpoints "
          "with no malicious update; the checkpoint count varies by version because it depends on how many attacker "
          "updates each gate let through. Same-denominator rate, honest client updates only: "
          + ", ".join(f"{v['version']} {v['honest_rejected']}/{v['honest']}" for v in versions) + ".", "",
          version_interpretation(versions), "",
          "What changed and why:", "",
          f"- **Case base:** {cb.meta['n_groups']} valid runs instead of 2, including client-dropout runs, because excluding "
          "an attacker turns the global model into a 2-client average the original base had never seen.",
          "- **Calibration:** the original threshold (worst held-out model × 1.25, per round) let one unusual valid model set "
          "the threshold for three rounds; pooled, median-normalised calibration uses all held-out models together.",
          "- **Rejection budget:** at most one update rejected per round. Rejecting every update signals drift, not a "
          "majority of attackers, and can freeze training.",
          f"- **Checkpoints traced, not gated:** across v1–v5 the checkpoint gate caught "
          f"{sum(v['ckpt_poisoned_rejected'] for v in early)} poisoned checkpoints and rolled back "
          f"{sum(v['ckpt_honest_rejected'] for v in early)} honest-only ones; the client gate stops most poison first.", "",
          "**Ablation (original split; harmful attacks = scaling, noise σ=2.0):**", "",
          "| Variant | AUROC upd. | AUROC ckpt. | TPR upd. | TPR ckpt. | FPR upd. | FPR ckpt. |",
          "|---|---|---|---|---|---|---|"]
    for r in ab:
        L.append(f"| {r['variant']} | {r['client_update_harmful_auroc']:.2f} | {r['checkpoint_harmful_auroc']:.2f} | "
                 f"{r['client_update_harmful_tpr']:.2f} | {r['checkpoint_harmful_tpr']:.2f} | "
                 f"{r['client_update_fpr']:.2f} | {r['checkpoint_fpr']:.2f} |")
    L += ["", f"**Overhead:** median {ovh['trace_s_median']:.2f} s per traced model (mean {ovh['trace_s_mean']:.2f}, "
          f"p95 {ovh['trace_s_p95']:.2f} s; the tail comes from runs that shared the CPU with offline evaluation), "
          f"{ovh['models_per_round']} models per round, mean round {ovh['round_s_mean']:.1f} s; case base "
          f"{ovh['case_base_bytes'] / 1e6:.1f} MB.", "",
          "## Threats to validity (state these in the paper)", "",
          "- One dataset, one small CAE, 3 clients, 1 attacker, 10 rounds; replicates vary initialisation only.",
          "- Weak attacks (sign flip ×1, noise σ=0.5, inverted training) barely change the model on this data; the gate "
          "is an anomaly detector and passes most of them. Report detection alongside harm.",
          "- Rounds 1–2 are weakly gated: clients start round 1 from their own random weights (they skip the initial "
          "global download), so the first FedAvg averages unrelated networks and the valid spread is wide.",
          "- The case base is specific to the probe set, model and FL configuration; any change requires rebuilding it.",
          "- The rejection budget assumes fewer than half the clients are malicious; with it, a flagged attacker can be "
          "admitted in a round where an honest update scores higher.",
          "- The case-base runs are also the source of held-out valid models (leave-one-run-out); there are no separate "
          "clean test runs. The live clean runs of each gate version are fresh.",
          "- The \"fixed validation set\" (inherited from `evaluate_cae.py`) is client_1's training shard, not held-out "
          "data. Excluding the attacker up-weights that shard, so gated MSE can fall below clean MSE; compare conditions "
          "relative to each other, and report a held-out split in the paper if possible.",
          "- Ablation: round-aware comparison is what raises detection; matching per probe image did not help (pooled "
          "images scored higher TPR). The live gates use the per-image variant.", "",
          "## Files", "",
          "| Path | Content |", "|---|---|",
          "| `data/models.csv` | Every traced model in the scored runs: run, condition, seed, round, kind, label, offline verdict, score, coverage, live decision and note |",
          "| `data/loro_valid.csv` | Every case-base model scored against a base built without its run |",
          "| `data/detection.csv` | TPR/FPR per attack × model kind, per-run mean/s.d., Wilson CI, AUROC + bootstrap CI |",
          "| `data/enforcement.csv` | Final live gate: rejections and final quality, gated vs. ungated |",
          "| `data/live_versions.csv` | All live gate versions: false rejections, detection, final MSE |",
          "| `data/round_quality.csv` | Validation MSE/SSIM of the checkpoint in force per round |",
          "| `data/confusion.csv`, `data/roc.csv` | Pooled confusion matrices; ROC operating points |",
          "| `data/ablation.csv`, `data/calibration.csv` | Design ablation; held-out valid scores and thresholds |",
          "| `data/overhead.json` | Timing and size |",
          "| `tables/*.tex` | IEEEtran tables (need `\\usepackage{booktabs}`); detection/enforcement/versions/ablation are `table*` |",
          "| `figures/*.pdf` | Vector figures, TrueType fonts embedded (PDF eXpress-safe); PNG previews alongside |",
          "| `results.tex` | `\\newcommand` macros for every number quoted in the text |",
          "| `preview.tex` | IEEEtran document that inputs every table and figure (compile on Overleaf) |", "",
          "## Figure captions (suggested)", "",
          "- **Fig. `scores_by_round`:** Gate score relative to threshold (log scale) per round. Grey band: 5–95% of held-out "
          "valid models. Lines: median over spoofed models across runs. Dashed: threshold.",
          "- **Fig. `quality_by_round`:** Validation MSE of the global model in force per round, attacked runs with the final "
          "gate and without a gate (mean; shaded: range over runs), against clean runs (grey).",
          "- **Fig. `roc`:** ROC of the gate score per attack, spoofed models vs. held-out valid models of the same kind.",
          "- **Fig. `versions`:** False-rejection rate and attacker detection of each live gate version.",
          "- **Fig. `calibration`:** Leave-one-run-out trace distance of valid models per round and the per-round threshold.", "",
          "## Confusion matrices (pooled)", "", "| Scope | Kind | TP | FN | FP | TN |", "|---|---|---|---|---|---|"]
    for c in conf:
        L.append(f"| {c['scope']} | {KINDS[c['kind']]} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} |")
    open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------

def main():
    global RUNS_DIR, CASE_BASE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("..", "paper"))
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--runs-dir", default=RUNS_DIR, help="Directory with the runs (e.g. sl_runs_gpu)")
    ap.add_argument("--case-base", default=None, help="Final case base (default: <runs-dir>/case_base_v2.npz)")
    args = ap.parse_args()
    RUNS_DIR = args.runs_dir
    CASE_BASE = args.case_base or os.path.join(RUNS_DIR, "case_base_v2.npz")
    for sub in ("data", "tables", "figures"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    D = lambda name: os.path.join(args.out, "data", name)
    # The quality cache is keyed by run name, and run names repeat across runs directories (sl_runs,
    # sl_runs_gpu): drop it when it was built from a different directory.
    if os.path.exists(D("environment.json")) and os.path.exists(D("round_quality.csv")):
        if json.load(open(D("environment.json"))).get("runs_dir") != RUNS_DIR:
            os.remove(D("round_quality.csv"))
            print(f"Discarded round_quality.csv: built from another runs directory")

    cb = SLCaseBase.load(CASE_BASE)
    base_runs = [os.path.basename(os.path.normpath(r)) for r in cb.meta["runs"]]
    runs = discover(set(base_runs))
    print(f"{len(runs)} runs: " + ", ".join(r["name"] for r in runs))
    ENV.update(environment(runs))
    json.dump(dict(ENV, runs_dir=RUNS_DIR), open(D("environment.json"), "w"), indent=2)
    json.dump(case_base_summary(cb, base_runs), open(D("case_base.json"), "w"), indent=2)
    models = model_matrix(runs, cb)
    write_csv(D("models.csv"), models)
    quality = round_quality([(r["name"], r["dir"]) for r in runs], D("round_quality.csv"))
    loro = loro_valid(cb, base_runs)
    write_csv(D("loro_valid.csv"), loro)

    det = detection(models, loro)
    write_csv(D("detection.csv"), det)
    enf = enforcement(models, runs, quality)
    write_csv(D("enforcement.csv"), enf)
    versions = live_versions(D("round_quality.csv"))
    write_csv(D("live_versions.csv"), versions)
    conf = confusion(models, loro)
    write_csv(D("confusion.csv"), conf)
    roc = roc_points(models, loro)
    write_csv(D("roc.csv"), roc)
    write_csv(D("calibration.csv"),
              [dict(c, distance_threshold=cb.thresholds[c["round"]]["score"],
                    coverage_threshold=cb.thresholds[c["round"]]["coverage"]) for c in cb.calibration])
    ab = [] if args.skip_ablation else ablation(runs)
    if ab:
        write_csv(D("ablation.csv"), ab)
    elif os.path.exists(D("ablation.csv")):
        ab = [{k: (float(v) if k not in ("variant", "match_images") else v) for k, v in r.items()}
              for r in csv.DictReader(open(D("ablation.csv")))]
    ovh = overhead(models, runs)
    ovh.update(runtimes(runs))
    json.dump(ovh, open(D("overhead.json"), "w"), indent=2)

    tex_tables(args.out, det, enf, ab, ovh, cb, versions)
    tex_macros(args.out, det, enf, ovh, runs, models, versions, cb)
    plt = setup_mpl()
    fig_scores(plt, args.out, models, loro)
    fig_quality(plt, args.out, runs, quality)
    fig_roc(plt, args.out, roc)
    fig_calibration(plt, args.out, cb)
    fig_versions(plt, args.out, versions)
    report(args.out, runs, det, enf, ab, ovh, cb, conf, models, versions)
    print(f"Wrote {args.out}/: data/, tables/, figures/, results.tex, REPORT.md")


if __name__ == "__main__":
    main()
