"""
Explain SL gate decisions in plain language, with a chart per decision.

Reads a run's sl_gate_log.jsonl and, for each selected model, writes a short explanation (verdict,
why, how far from the valid models, closest precedents, which trace features left the valid range)
and a PNG: distance and coverage against their thresholds, and each out-of-tolerance feature
against the range valid models occupy.

Models the server traced but did not gate (checkpoints under --sl-gate-scope clients) have no
logged decision; they are scored offline against the run's case base and marked as such.

    python explain_sl_decision.py sl_runs_demo/gated                      # every rejected/invalid model
    python explain_sl_decision.py sl_runs_demo/gated --model r5/client_3 r5/client_1
    python explain_sl_decision.py sl_runs_demo/gated --all

Output: <run>/explanations/explanations.md and one <round>_<source>.png per model.
"""
import argparse
import json
import os

import numpy as np

from sl_gate import SLCaseBase, load_run

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BAND = "#d9d8d4"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

ROLE = {"encoder": "Encoder", "decoder": "Decoder"}
LAYER = {"conv": "conv layers", "deconv": "deconv layers", "fc": "bottleneck (fully connected)"}
STAT = {
    "mean": "activation strength",
    "std": "activation variability",
    "max": "peak activation",
    "p90_over_mean": "high-activation share (p90 / mean)",
    "sparsity": "fraction of near-zero activations",
    "kurtosis": "tail heaviness (kurtosis)",
    "max_over_rms": "peak-to-RMS ratio",
}
# conv/deconv statistics are summarised across the layers of the family (sl_minimize)
ACROSS = {
    "phi": "average log level across layers",
    "spread": "variation across layers",
    "peak": "share of the strongest layer",
    "depth": "which layer dominates",
    "drift": "centre of mass across layers",
    "conc": "concentration in few layers",
}


def describe(feature):
    """'slmin.decoder.fc.mean' -> 'Decoder bottleneck (fully connected): activation strength'."""
    parts = feature.split(".")[1:]
    role, layer, stat = parts[0], parts[1], parts[2]
    text = f"{ROLE.get(role, role)} {LAYER.get(layer, layer)}: {STAT.get(stat, stat)}"
    if len(parts) > 3:
        text += f", {ACROSS.get(parts[3], parts[3])}"
    return text


def severity(ratio):
    if ratio < 0.8:
        return "comfortably inside what valid models produce"
    if ratio <= 1.0:
        return "close to the limit but still inside it"
    if ratio < 2:
        return "just over the limit"
    if ratio < 10:
        return "well beyond the limit"
    return "far beyond anything valid models produce"


def times(x):
    return f"{x:,.0f}x" if x >= 10 else f"{x:.2g}x"


def beyond(f):
    """How far a feature sits outside its valid range, in plain words."""
    lo, hi = f["bounds"]
    side = "above" if f["z"] > hi else "below"
    if hi - lo < 1e-6:                      # every valid model has the same value (e.g. which layer dominates)
        return f"every valid model has {lo:.2f}"
    k = (f["z"] - hi if side == "above" else lo - f["z"]) / (hi - lo)
    return f"{times(k)} the range width {side} it"


def action(m, offline):
    if offline:
        return "not gated live (traced only); scored offline against the run's case base"
    if m.get("included") is False or m.get("accepted") is False:
        what = "rejected (excluded from the average)" if m["source"] != "global" else "rejected (rolled back)"
    else:
        what = "accepted"
    return what + (f" -- {m['note']}" if m.get("note") else "")


def summary(m, d, offline, n_valid):
    """Everything an explanation shows, as data (the Markdown report and the demo page both render it).
    ``paragraph`` uses **bold** and `code` markup."""
    ok = d["verdict"] == "valid"
    r, rounds = d["round"], d["compared_rounds"]
    who = "the averaged checkpoint" if m["source"] == "global" else f"the update from {m['source']}"
    pool = f"{n_valid} valid models" if n_valid else "the valid models"
    when = (f"round {m['round']}" if r == m["round"]
            else f"round {m['round']}, compared at training progress {r} because earlier updates were rejected")
    para = (f"The gate compared {who} ({when}) with {pool} from rounds {rounds[0]}-{rounds[-1]} of the "
            f"case base. Probe image by probe image, its SL trace was matched with the closest valid traces: "
            f"its distance is **{d['score_ratio']:.2f}x the threshold**, {severity(d['score_ratio'])}. ")
    cov_ok = d["coverage"] >= d["coverage_threshold"]
    para += (f"{d['coverage']:.0%} of its trace features fall inside the range valid models occupy "
             f"(threshold {d['coverage_threshold']:.0%}"
             + ("" if d.get("coverage_rule", True) else ", reported as evidence only") + ")"
             + ("." if cov_ok else f", so {1 - d['coverage']:.0%} of its features are out of range where valid "
                                    f"models have at most {1 - d['coverage_threshold']:.0%}."))
    near = d.get("nearest_valid") or []
    if near:
        top = near[0]
        para += (f" Its closest precedent is `{top['model']}` (nearest on {top['probe_images']} of the "
                 f"probe images)" + ("." if d["score_ratio"] < 2 else ", and even that is far away."))
    features = [dict(feature=f["feature"], label=describe(f["feature"]), z=f["z"], lo=f["bounds"][0],
                     hi=f["bounds"][1], beyond=beyond(f)) for f in d.get("out_of_tolerance") or []]
    rejected = not offline and (m.get("included") is False or m.get("accepted") is False)
    return dict(model=m["model"], round=m["round"], source=m["source"], verdict=d["verdict"],
                rejected=rejected, offline=offline, action=action(m, offline), note=m.get("note"),
                reasons=d["reasons"] or ["no rule fired"], paragraph=para, score_ratio=d["score_ratio"],
                coverage=d["coverage"], coverage_threshold=d["coverage_threshold"],
                compared_rounds=rounds, nearest=d.get("nearest_valid") or [], features=features)


def explain(s):
    """Markdown for one summary()."""
    ok = s["verdict"] == "valid"
    lines = [f"### {s['model']} -- {'VALID' if ok else 'INVALID'}", "",
             f"- **Decision:** {s['action']}", f"- **Why:** {'; '.join(s['reasons'])}", "", s["paragraph"], ""]
    if s["features"]:
        lines.append("Features outside the valid range:" if not ok else "Features at the edge of the valid range:")
        lines += [f"- {f['label']} -- z = {f['z']:.2f} vs valid {f['lo']:.2f} to {f['hi']:.2f} ({f['beyond']}) "
                  f"`{f['feature']}`" for f in s["features"]]
    else:
        lines.append("Every trace feature is inside the valid range.")
    return "\n".join(lines) + "\n"


def chart(m, d, offline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oot = d.get("out_of_tolerance") or []
    n = max(len(oot), 1)
    fig = plt.figure(figsize=(9.5, 2.6 + 0.62 * n), facecolor=SURFACE)
    h = fig.get_figheight()
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.45 + 0.62 * n], hspace=0.75, wspace=0.28,
                          top=1 - 0.75 / h)
    ok = d["verdict"] == "valid"
    title = (f"{m['model']}: {'VALID' if ok else 'INVALID'} -- {action(m, offline)}")
    fig.suptitle(title, x=0.02, y=1 - 0.1 / h, ha="left", fontsize=12, color=INK, fontweight="bold")

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks([])

    # Distance: log scale, threshold at 1x
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    ratio = d["score_ratio"]
    hi = max(10, ratio * 3)
    ax.set_xscale("log")
    ax.set_xlim(0.1, hi)
    ticks = [t for t in (0.1, 1, 10, 100, 1000, 10000) if t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}x" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axvspan(0.1, 1.0, color=BAND, alpha=0.6, lw=0)
    ax.axvline(1.0, color=INK2, lw=1.2, ls="--")
    c = GOOD if ratio <= 1 else CRITICAL
    ax.plot([ratio], [0], "o", ms=10, color=c, mec=SURFACE, mew=2, zorder=3)
    ax.annotate(f"{ratio:.2f}x", (ratio, 0), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(1.0, -0.62, "threshold", ha="center", fontsize=8, color=INK2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_title("Distance to nearest valid models (x threshold)", loc="left", fontsize=10, color=INK)

    # Coverage: linear 0-100 %
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    cov, thr = d["coverage"], d["coverage_threshold"]
    ax.set_xlim(0, 1)
    ax.axvspan(thr, 1, color=BAND, alpha=0.6, lw=0)
    ax.axvline(thr, color=INK2, lw=1.2, ls="--")
    c = GOOD if cov >= thr else CRITICAL
    ax.plot([cov], [0], "o", ms=10, color=c, mec=SURFACE, mew=2, zorder=3)
    ax.annotate(f"{cov:.0%}", (cov, 0), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(thr, -0.62, "threshold", ha="center", fontsize=8, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(-0.8, 0.8)
    ax.set_title("Features inside the valid range", loc="left", fontsize=10, color=INK)

    # Out-of-tolerance features vs valid range (symlog: keeps the range readable next to large z)
    ax = fig.add_subplot(gs[1, :])
    style(ax)
    ax.set_title(("Trace features outside the valid range" if not ok else
                  "Trace features nearest the edge of the valid range") + " (standardised value z)"
                 if oot else "Trace features", loc="left", fontsize=10, color=INK)
    if not oot:
        ax.text(0.5, 0.5, "Every trace feature is inside the valid range", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=INK2)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xscale("symlog", linthresh=2)
        ys = np.arange(len(oot))[::-1]
        zmax = max(max(abs(f["z"]), abs(f["bounds"][0]), abs(f["bounds"][1])) for f in oot)
        lim = max(4, zmax * 2)
        left = -lim if min(min(f["z"], f["bounds"][0]) for f in oot) < -2 else -4
        ax.set_xlim(left, lim)
        ticks = [t for t in (-1000, -100, -10, -1, 0, 1, 10, 100, 1000) if left <= t <= lim]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}".replace("-", "−") for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for y, f in zip(ys, oot):
            lo, hi_ = f["bounds"]
            if hi_ - lo < 1e-6:   # all valid models share one value: draw it as a tick
                ax.plot([lo, lo], [y - 0.25, y + 0.25], color=INK2, lw=2.5, solid_capstyle="round", zorder=2)
            else:
                ax.barh(y, hi_ - lo, left=lo, height=0.42, color=BAND, edgecolor="none", zorder=1)
            ax.plot([f["z"]], [y], "o", ms=9, color=CRITICAL if not ok else INK2, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(f"z = {f['z']:.2f}", (f["z"], y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{describe(f['feature'])}\n{f['feature']}" for f in oot], fontsize=8.5, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(oot) - 0.3)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        ax.legend(handles=[Patch(color=BAND, label="valid range (middle 95% of valid models, same rounds)"),
                           Line2D([], [], color=INK2, lw=2.5, label="all valid models equal"),
                           Line2D([], [], marker="o", ls="", color=CRITICAL if not ok else INK2,
                                  label="this model (mean over probe images)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.45 / n - 0.12), ncol=3, frameon=False,
                  fontsize=8.5, labelcolor=INK2)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def valid_pool(cb, d):
    """Number of valid case-base models in the rounds a decision compared against."""
    return len({c["model"] for c in cb.calibration if c["round"] in d["compared_rounds"]}) if cb else None


def load_decisions(run_dir, case_base=None):
    """``(meta, case_base_path, case_base, log, [(log entry, decision, offline)])`` for a run.

    Models with no logged decision (traced, not gated) are scored offline against ``case_base``
    (default: the one the run used); without a case base they are left out.
    """
    log = [json.loads(line) for line in open(os.path.join(run_dir, "sl_gate_log.jsonl"))]
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    cb_path = case_base or meta.get("sl_case_base")
    cb = SLCaseBase.load(cb_path) if cb_path and os.path.exists(cb_path) else None
    run, items = None, []
    for m in log:
        d, offline = m.get("decision"), False
        if d is None:
            if cb is None:
                continue
            run = run or load_run(run_dir)
            d = cb.decide(run["Z"][run["row_model"] == m["model"]], m.get("compare_round", m["round"]))
            offline = True
        items.append((m, d, offline))
    return meta, cb_path, cb, log, items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Run directory with sl_gate_log.jsonl")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--model", nargs="+", help="Model ids to explain, e.g. r5/client_3 r5/global")
    sel.add_argument("--all", action="store_true", help="Explain every logged model")
    ap.add_argument("--case-base", default=None,
                    help="Case base for models without a logged decision (default: the run's own)")
    ap.add_argument("--out", default=None, help="Output directory (default: <run>/explanations)")
    args = ap.parse_args()

    meta, cb_path, cb, log, items = load_decisions(args.run, args.case_base)
    if args.model:
        missing = [x for x in args.model if x not in {m["model"] for m in log}]
        if missing:
            raise SystemExit(f"Not in the log: {', '.join(missing)}")
        items = [it for it in items if it[0]["model"] in args.model]
        items.sort(key=lambda it: args.model.index(it[0]["model"]))
    elif not args.all:
        items = [it for it in items if it[1]["verdict"] == "invalid" or it[0].get("accepted") is False]
        if not items:
            raise SystemExit("No rejected or invalid models in this run; pass --model or --all.")

    out = args.out or os.path.join(args.run, "explanations")
    os.makedirs(out, exist_ok=True)
    attack = meta.get("attack")
    md = [f"# SL gate decisions: {os.path.normpath(args.run)}", "",
          f"- Case base: `{cb_path}`" + (f" ({cb.meta['n_models']} valid models)" if cb else ""),
          f"- Gate: {meta.get('sl_gate_mode') or 'trace only'}, scope {meta.get('sl_gate_scope') or '-'}",
          f"- Attack: {attack + ' by ' + meta['malicious_client'] if attack else 'none'}",
          f"- Explained: {len(items)} of {len(log)} logged models", ""]
    for m, d, offline in items:
        md.append(explain(summary(m, d, offline, valid_pool(cb, d))))
        png = os.path.join(out, m["model"].replace("/", "_") + ".png")
        chart(m, d, offline, png)
        md.append(f"![{m['model']}]({os.path.basename(png)})\n")
        print(f"{m['model']:<16} {d['verdict']:<8} {d['score_ratio']:>8.2f}x  -> {png}")
    path = os.path.join(out, "explanations.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
