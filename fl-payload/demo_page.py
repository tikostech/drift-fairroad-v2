"""
The SL gate demo page: one self-contained HTML file (no server, no internet) that shows a gated run's
decisions and explains each one. demo_sl_gate.py builds it after training; it can also be rebuilt alone:

    python demo_page.py demo_runs/gated --ungated demo_runs/ungated --quality --open

Carries no IP. Every explanation is rendered from the decision the CBR service returned, which the
server logs in sl_gate_log.jsonl: verdict, distance and coverage against their thresholds, reasons,
closest valid precedents and out-of-range trace features. Models the gate didn't decide live (the
checkpoints with --sl-gate-scope clients, and the ungated baseline's checkpoints) are scored by the CBR
service from the run's trace archive, with the same case base. Reconstruction quality (--quality) uses
the FL project's own evaluator and needs torch and the image data.
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.append(os.path.dirname(HERE))  # ip_client sits at the repo root

import config  # noqa: E402
from ip_client import CBRServiceClient  # noqa: E402
from trace_store import load_run  # noqa: E402

DEFAULT_CBR_URL = config.CBR_SERVICE_URL  # http://127.0.0.1:8602

ATTACKS = {"scale": "update scaling", "sign_flip": "sign flip", "noise": "Gaussian noise",
           "inverted": "inverted training"}


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
        return "not gated live (traced only); scored by the CBR service against the run's case base"
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


def _evaluator():
    sys.argv = sys.argv[:1]
    from evaluate_cae import CAEEvaluator
    ev = CAEEvaluator()
    clean, noisy = (p if os.path.isabs(p) else os.path.join(HERE, p)
                    for p in (config.CLEAN_IMAGE_DIR, config.NOISY_IMAGE_DIR))
    dl = ev.get_fixed_dataloader(clean, noisy, config.NUM_CLIENTS, 0)
    return ev, dl


def _checkpoints(run_dir):
    rounds = sorted(int(f[len("server_round_"):-3]) for f in os.listdir(run_dir)
                    if f.startswith("server_round_") and f.endswith(".pt"))
    return [(r, os.path.join(run_dir, f"server_round_{r}.pt")) for r in rounds]


def round_quality(run_dirs, ev=None):
    """{run_dir: [{round, mse, ssim}]} on the fixed validation set, cached in <run>/quality.json."""
    out, todo = {}, []
    for d in run_dirs:
        cache = os.path.join(d, "quality.json")
        if os.path.exists(cache) and os.path.getmtime(cache) >= max(os.path.getmtime(p) for _, p in _checkpoints(d)):
            out[d] = json.load(open(cache))
        else:
            todo.append(d)
    if todo:
        ev, dl = ev or _evaluator()
        for d in todo:
            rows = []
            for r, path in _checkpoints(d):
                ev.model.load(path)
                q = ev.evaluate_on_dataset(dl)
                rows.append(dict(round=r, mse=float(q["mse"]), ssim=float(q["ssim"])))
            json.dump(rows, open(os.path.join(d, "quality.json"), "w"), indent=1)
            out[d] = rows
    return out


def reconstructions(run_dirs, n=3, ev=None):
    """Final-round reconstructions of the first ``n`` fixed-validation images, as PNG data URIs:
    [{noisy, clean, <run_dir>: output, ...}]."""
    import torch
    from matplotlib import image as mpimg

    def uri(t):
        buf = io.BytesIO()
        mpimg.imsave(buf, t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy(), format="png")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    ev, dl = ev or _evaluator()
    noisy, clean = next(iter(dl))
    noisy, clean = noisy[:n], clean[:n]
    rows = [dict(noisy=uri(noisy[i]), clean=uri(clean[i])) for i in range(len(noisy))]
    for d in run_dirs:
        ev.model.load(_checkpoints(d)[-1][1])
        ev.model.model.eval()
        with torch.no_grad():
            out = ev.model.model(noisy.to(ev.device))
        for i, row in enumerate(rows):
            row[d] = uri(out[i])
    return rows


def load_decisions(run_dir, cbr=None, case_base=None):
    """``(meta, case_base_name, case_base_info, log, [(log entry, decision, offline)])`` for a run.

    Logged decisions come from the CBR service. Models with none (traced, not gated) are scored by
    ``cbr`` against ``case_base`` (default: the one the run used) from the trace archive; without a
    service they are left out.
    """
    log = [json.loads(line) for line in open(os.path.join(run_dir, "sl_gate_log.jsonl"), encoding="utf-8")]
    meta = json.load(open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8"))
    name = case_base or meta.get("sl_case_base") or next(
        (m["decision"]["case_base"] for m in log if (m.get("decision") or {}).get("case_base")), None)
    if name and (name.endswith(".npz") or "/" in name or "\\" in name):
        name = os.path.splitext(os.path.basename(name.replace("\\", "/")))[0]  # older runs recorded a file path
    info = cbr.describe(name) if cbr and name else None
    run, items = None, []
    for m in log:
        d, offline = m.get("decision"), False
        if d is None:
            if info is None:
                continue
            run = run or load_run(run_dir)
            d = cbr.decide(name, run["Z"][run["row_model"] == m["model"]], m.get("compare_round", m["round"]))
            offline = True
        items.append((m, d, offline))
    return meta, name, info, log, items


def page_data(gated_dir, ungated_dir=None, quality=None, recon=None, cbr=None):
    meta, name, info, log, items = load_decisions(gated_dir, cbr)
    models = []
    for m, d, offline in items:
        s = summary(m, d, offline, None)
        s.update(kind="checkpoint" if m["source"] == "global" else "client_update", run="gated")
        models.append(s)
    # The ungated run's checkpoints, scored against the same case base: what the gate would have said
    # about checkpoints that averaged the attacker in.
    baseline = []
    if ungated_dir and info:
        _, _, _, _, uitems = load_decisions(ungated_dir, cbr, name)
        for m, d, _ in uitems:
            if m["source"] != "global":
                continue
            s = summary(m, d, True, None)
            s.update(model=f"{m['model']} (no gate)", kind="checkpoint", run="ungated", rejected=False,
                     action="no gate in this run; scored by the CBR service against the same case base "
                            "(what the gate would have said)")
            baseline.append(s)
    attack = meta.get("attack")
    label = ATTACKS.get(attack, attack) if attack else None
    if label and meta.get("attack_scale"):
        label += f" (x{meta['attack_scale']:g})"
    info = info or {}
    data = dict(
        run=os.path.normpath(gated_dir), ungated=os.path.normpath(ungated_dir) if ungated_dir else None,
        attack=label, malicious=meta.get("malicious_client") if attack else None,
        gate_mode=meta.get("sl_gate_mode"), gate_scope=meta.get("sl_gate_scope"),
        case_base=name, n_valid=info.get("n_models"),
        n_base_runs=len((info.get("meta") or {}).get("runs", [])) or None,
        target_fpr=(info.get("selection") or {}).get("target_fpr"),
        rounds=max(m["round"] for m in models), models=models, baseline=baseline,
        generated=time.strftime("%Y-%m-%d %H:%M"))
    if quality:
        data["quality"] = {"with": quality.get(gated_dir), "without": quality.get(ungated_dir)}
    if recon:
        data["recon"] = [dict(noisy=r["noisy"], clean=r["clean"], with_gate=r.get(gated_dir),
                              without_gate=r.get(ungated_dir)) for r in recon]
    return data


def build_page(data, out_html):
    payload = json.dumps(data).replace("</", "<\\/")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__DATA__", payload))
    return out_html


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gated", help="Gated run directory (sl_gate_log.jsonl, run_meta.json, sl_traces.npz)")
    ap.add_argument("--ungated", default=None, help="Same attack without the gate, for the comparison")
    ap.add_argument("--quality", action="store_true", help="Measure reconstruction error per round (needs torch)")
    ap.add_argument("--cbr", default=os.environ.get("CBR_SERVICE_URL") or DEFAULT_CBR_URL,
                    help=f"CBR service URL (default: $CBR_SERVICE_URL or {DEFAULT_CBR_URL})")
    ap.add_argument("--api-key", default=None,
                    help="IP services API key (default: $IP_SERVICE_API_KEY, else the services' default key)")
    ap.add_argument("--out", default=None, help="Output HTML (default: <gated>/../sl_gate_demo.html)")
    ap.add_argument("--open", action="store_true", help="Open the page in the browser")
    args = ap.parse_args()
    quality = recon = None
    if args.quality:
        dirs = [d for d in (args.gated, args.ungated) if d]
        ev = _evaluator()
        quality = round_quality(dirs, ev)
        recon = reconstructions(dirs, ev=ev)
    cbr = CBRServiceClient(args.cbr, api_key=args.api_key)
    out = args.out or os.path.join(os.path.dirname(os.path.normpath(args.gated)), "sl_gate_demo.html")
    build_page(page_data(args.gated, args.ungated, quality, recon, cbr), out)
    print(f"Saved: {out}")
    if args.open:
        webbrowser.open("file:///" + os.path.abspath(out).replace(os.sep, "/"))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SL Gate Demo</title>
<style>
:root {
  color-scheme: light;
  --bg: #f4f3f0; --surface: #fcfcfb; --raised: #ffffff; --line: #e3e2de; --grid: #ecebe7;
  --ink: #0b0b0b; --ink2: #52514e; --ink3: #7a7975; --band: #dddcd7; --zone: #eeede9;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #6250d6;
  --good: #0ca30c; --critical: #d03b3b; --good-bg: #e7f5e7; --critical-bg: #fbeaea; --muted-bg: #efeeea;
  --focus: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --raised: #222220; --line: #33332f; --grid: #2a2a27;
    --ink: #f0efec; --ink2: #c3c2b7; --ink3: #8f8e86; --band: #45453f; --zone: #242422;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #9085e9;
    --good-bg: #16301a; --critical-bg: #3a1c1c; --muted-bg: #2a2a27; --focus: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --raised: #222220; --line: #33332f; --grid: #2a2a27;
  --ink: #f0efec; --ink2: #c3c2b7; --ink3: #8f8e86; --band: #45453f; --zone: #242422;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #9085e9;
  --good-bg: #16301a; --critical-bg: #3a1c1c; --muted-bg: #2a2a27; --focus: #3987e5;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 0 0 2px; }
.sub { color: var(--ink2); margin: 0; }
.hint { color: var(--ink3); font-size: 13px; margin: 0 0 12px; }
code, .mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 0.86em; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px; min-width: 0; }
header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 20px; flex-wrap: wrap; }
.theme { border: 1px solid var(--line); background: var(--surface); color: var(--ink2); border-radius: 8px;
  padding: 6px 10px; font: inherit; font-size: 13px; cursor: pointer; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile .k { color: var(--ink2); font-size: 13px; }
.tile .v { font-size: 30px; font-weight: 650; letter-spacing: -0.02em; line-height: 1.15; margin-top: 4px; }
.tile .d { color: var(--ink2); font-size: 13px; margin-top: 2px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.explore { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 16px; margin-bottom: 16px; }
@media (max-width: 900px) { .grid2, .explore { grid-template-columns: 1fr; } }
svg { display: block; overflow: visible; }
svg text { fill: var(--ink2); font-size: 11.5px; font-family: inherit; }
svg text.halo { paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink2); font-size: 13px; margin: 8px 0 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.badge { display: inline-flex; align-items: center; gap: 5px; border-radius: 999px; padding: 2px 10px;
  font-size: 12.5px; font-weight: 600; white-space: nowrap; }
.badge.bad { background: var(--critical-bg); color: var(--critical); }
.badge.ok { background: var(--good-bg); color: var(--good); }
.badge.mut { background: var(--muted-bg); color: var(--ink2); }
:root[data-theme="dark"] .badge.bad, :root[data-theme="dark"] .badge.ok { filter: brightness(1.25); }
.ex-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.ex-head h2 { font-size: 20px; }
.facts { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; margin: 0 0 10px; font-size: 14px; }
.facts dt { color: var(--ink2); }
.facts dd { margin: 0; }
.para { margin: 0 0 14px; }
.para code { background: var(--muted-bg); padding: 1px 5px; border-radius: 4px; }
.gauges { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
@media (max-width: 520px) { .gauges { grid-template-columns: 1fr; } }
.gauge .t { font-size: 13px; color: var(--ink2); }
.gauge .n { font-size: 22px; font-weight: 650; }
.feat { margin: 0 0 10px; }
.feat .lab { font-size: 13.5px; }
.feat .raw { color: var(--ink3); font-size: 11.5px; }
.feat .note { color: var(--ink2); font-size: 12.5px; }
.near { margin: 0; padding-left: 18px; font-size: 13.5px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 12px; }
.chip { border: 1px solid var(--line); background: var(--raised); color: var(--ink); border-radius: 999px;
  padding: 3px 11px; font: inherit; font-size: 13px; cursor: pointer; }
.chip[aria-pressed="true"] { background: var(--ink); color: var(--surface); border-color: var(--ink); }
.chip:focus-visible, .theme:focus-visible, tr:focus-visible, .navb:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.navb { border: 1px solid var(--line); background: var(--raised); color: var(--ink); border-radius: 8px;
  padding: 4px 10px; font: inherit; font-size: 13px; cursor: pointer; }
.tablewrap { overflow-x: auto; max-height: 460px; overflow-y: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th { text-align: left; color: var(--ink2); font-weight: 600; position: sticky; top: 0; background: var(--surface);
  border-bottom: 1px solid var(--line); padding: 6px 8px; }
td { padding: 6px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--zone); }
tbody tr[aria-selected="true"] { background: var(--muted-bg); box-shadow: inset 3px 0 0 var(--ink); }
.recon { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.recon figure { margin: 0; }
.recon img { width: 100%; border-radius: 8px; display: block; image-rendering: auto; background: var(--zone); }
.recon figcaption { font-size: 12.5px; color: var(--ink2); margin-top: 4px; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.steps .n { font-weight: 650; color: var(--ink2); font-size: 13px; }
.steps p { margin: 4px 0 0; font-size: 14px; }
.tip { position: fixed; pointer-events: none; background: var(--raised); color: var(--ink); border: 1px solid var(--line);
  border-radius: 8px; padding: 6px 9px; font-size: 12.5px; box-shadow: 0 4px 16px rgba(0,0,0,.12); z-index: 10;
  display: none; max-width: 260px; }
.empty { color: var(--ink2); font-size: 14px; padding: 12px 0; }
section { margin-bottom: 16px; }
footer { color: var(--ink3); font-size: 12.5px; margin-top: 24px; }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1 id="title">SL gate demo</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <button class="theme" id="theme" type="button" aria-label="Toggle light or dark theme">Theme</button>
  </header>

  <div class="tiles" id="tiles"></div>

  <div class="grid2">
    <div class="card" id="qualityCard">
      <h2>Reconstruction error per round</h2>
      <p class="hint">Mean squared error on a fixed validation set (lower is better). Same attack, with and without the gate.</p>
      <div id="quality"></div>
    </div>
    <div class="card">
      <h2>Gate decisions per round</h2>
      <p class="hint">Distance of each model's SL trace to the nearest valid models, as a multiple of the threshold (log scale). Click a mark to explain it.</p>
      <div class="legend" id="decLegend"></div>
      <div id="decisions"></div>
    </div>
  </div>

  <section class="card" id="ckCard">
    <h2>Checkpoints with and without the gate</h2>
    <p class="hint" id="ckHint"></p>
    <div class="legend" id="ckLegend"></div>
    <div id="checkpoints"></div>
  </section>

  <div class="explore">
    <div class="card" id="explain" aria-live="polite"></div>
    <div class="card">
      <h2>All decisions</h2>
      <p class="hint">Click a row, or use the left and right arrow keys to step through.</p>
      <div class="chips" id="filters" role="group" aria-label="Filter decisions"></div>
      <div class="tablewrap"><table>
        <thead><tr><th>Round</th><th>Model</th><th>Decision</th><th class="num">Distance</th><th class="num">Coverage</th></tr></thead>
        <tbody id="rows"></tbody>
      </table></div>
    </div>
  </div>

  <section class="card" id="reconCard">
    <h2>What the final model produces</h2>
    <p class="hint">Round-10 reconstructions of validation images.</p>
    <div class="recon" id="recon"></div>
  </section>

  <section class="card">
    <h2>How the gate decides</h2>
    <div class="steps" id="steps"></div>
  </section>
  <footer id="foot"></footer>
</main>
<div class="tip" id="tip" role="tooltip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("data").textContent);
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>");
  const fmtX = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)) + "×";
  const pct = (x) => Math.round(x * 100) + "%";
  const sources = [...new Set(D.models.map((m) => m.source))].sort((a, b) => (a === "global") - (b === "global") || a.localeCompare(b));
  const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
  const color = (src) => (src === "global" ? "var(--s4)" : COLORS[sources.filter((s) => s !== "global").indexOf(src) % 3]);
  const nameOf = (src) => (src === "global" ? "Checkpoint" : src.replace("client_", "Client "));
  const state = (m) => (m.offline ? "trace" : m.rejected ? "rejected" : "accepted");
  const BASE = D.baseline || [];
  const ALL = D.models.concat(BASE);
  const label = (m) => nameOf(m.source) + (m.run === "ungated" ? " (no gate)" : BASE.length && m.kind === "checkpoint" ? " (with gate)" : "");
  function badge(m) {
    if (m.run === "ungated") return m.verdict === "invalid" ? '<span class="badge bad">✕ Would be rejected</span>' : '<span class="badge ok">✓ Would pass</span>';
    if (m.offline) return '<span class="badge mut">○ Traced only</span>';
    if (m.rejected) return '<span class="badge bad">✕ Rejected</span>';
    if (m.verdict === "invalid") return '<span class="badge bad">⚠ Flagged, admitted</span>';
    return '<span class="badge ok">✓ Accepted</span>';
  }
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const tip = $("tip");
  function showTip(html, ev) {
    tip.innerHTML = html; tip.style.display = "block";
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tip.style.display = "none"; };

  // ---------- header + tiles ----------
  $("title").textContent = D.attack ? `SL gate vs. ${D.attack} attack by ${D.malicious.replace("client_", "client ")}` : "SL gate on a clean run";
  $("subtitle").textContent = `${D.rounds} federated rounds · gate ${D.gate_mode || "off"}` +
    (D.gate_scope === "clients" ? " on client updates (checkpoints traced only)" : D.gate_scope ? ` (scope: ${D.gate_scope})` : "") +
    (D.n_valid ? ` · case base of ${D.n_valid} valid models` : "");
  (function tiles() {
    const t = [];
    const upd = D.models.filter((m) => m.kind === "client_update");
    if (D.malicious) {
      const bad = upd.filter((m) => m.source === D.malicious);
      t.push(["Attacker updates rejected", `${bad.filter((m) => m.rejected).length} / ${bad.length}`, `every update from ${D.malicious.replace("client_", "client ")}`]);
    } else {
      t.push(["Updates rejected", `${upd.filter((m) => m.rejected).length} / ${upd.length}`, "no attacker in this run"]);
    }
    const good = upd.filter((m) => m.source !== D.malicious);
    t.push(["Honest updates accepted", `${good.filter((m) => !m.rejected).length} / ${good.length}`, "false alarms cost training data"]);
    const q = D.quality;
    if (q && q.with && q.with.length) {
      const w = q.with[q.with.length - 1].mse;
      if (q.without && q.without.length) {
        const wo = q.without[q.without.length - 1].mse;
        t.push(["Final error with the gate", w.toFixed(3), `vs ${wo.toFixed(3)} without (${Math.round((1 - w / wo) * 100)}% lower)`]);
      } else t.push(["Final error with the gate", w.toFixed(3), "mean squared error, round " + q.with[q.with.length - 1].round]);
    }
    const ck = D.models.filter((m) => m.kind === "checkpoint");
    const flagged = (l) => l.filter((m) => m.verdict === "invalid").length;
    if (BASE.length) t.push(["Poisoned checkpoints flagged", `${flagged(BASE)} / ${BASE.length}`, `in the run without the gate; with it: ${flagged(ck)} / ${ck.length}`]);
    else if (ck.length) t.push(["Checkpoints that look valid", `${ck.length - flagged(ck)} / ${ck.length}`, ck.some((m) => m.offline) ? "scored offline, not gated" : "gated live"]);
    $("tiles").innerHTML = t.map(([k, v, d]) => `<div class="card tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
  })();

  // ---------- selection ----------
  let filter = D.malicious ? "rejected" : "all";
  const filtered = () => filter === "baseline" ? BASE : D.models.filter((m) => filter === "all" ? true : filter === "rejected" ? m.rejected || m.verdict === "invalid" : m.source === filter);
  let sel = (D.models.find((m) => m.rejected && m.round === Math.min(5, D.rounds)) || D.models.find((m) => m.rejected) || D.models[0]).model;
  function select(id) { sel = id; renderDecisions(); renderCheckpoints(); renderExplain(); renderTable(); }

  // ---------- checkpoints with vs without the gate ----------
  function renderCheckpoints() {
    const box = $("checkpoints");
    const runs = [["gated", "With gate", "var(--ink)", "", D.models.filter((m) => m.kind === "checkpoint")],
                  ["ungated", "Without gate", "var(--ink2)", "6 5", BASE]].filter((r) => r[4].length);
    if (!BASE.length) { $("ckCard").style.display = "none"; return; }
    $("ckHint").textContent = "Each round's averaged checkpoint, scored against the same case base (log scale). Without the gate the attacker's update is averaged in; the gate would have flagged every point above the threshold. Click a point to explain it.";
    $("ckLegend").innerHTML = runs.map(([, l, c, dash]) => `<span><svg width="26" height="10" aria-hidden="true"><line x1="0" x2="26" y1="5" y2="5" stroke="${c}" stroke-width="2" stroke-dasharray="${dash}"/></svg>${l}</span>`).join("") +
      `<span><i class="sw" style="background:var(--critical)"></i>above the threshold (would be rejected)</span>`;
    const W = Math.max(box.clientWidth, 280), H = 240, m = { l: 48, r: 104, t: 12, b: 30 };
    const R = D.rounds, L = Math.log10;
    const ratios = runs.flatMap((r) => r[4].map((d) => d.score_ratio));
    const lo = Math.min(0.1, Math.min(...ratios) / 1.5), hi = Math.max(10, Math.max(...ratios) * 1.6);
    const x = (r) => m.l + (r - 1) / Math.max(R - 1, 1) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (L(v) - L(lo)) / (L(hi) - L(lo))) * (H - m.t - m.b);
    box.innerHTML = "";
    const svg = el("svg", { width: W, height: H, role: "img", "aria-label": "Checkpoint distance per round with and without the gate" }, box);
    el("rect", { x: m.l, y: y(1), width: W - m.l - m.r, height: H - m.b - y(1), fill: "var(--zone)" }, svg);
    for (let e = Math.ceil(L(lo)); e <= Math.floor(L(hi)); e++) {
      const v = Math.pow(10, e);
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--grid)" }, svg);
      el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = (v < 1 ? v : v.toFixed(0)) + "×";
    }
    el("line", { x1: m.l, x2: W - m.r, y1: y(1), y2: y(1), stroke: "var(--ink2)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, svg);
    const tl = el("text", { x: m.l + 4, y: y(1) - 6, class: "halo" }, svg); tl.textContent = "threshold"; tl.style.fill = "var(--ink)";
    const vz = el("text", { x: m.l + 4, y: H - m.b - 6, class: "halo" }, svg); vz.textContent = "valid zone";
    for (let r = 1; r <= R; r++) el("text", { x: x(r), y: H - 10, "text-anchor": "middle" }, svg).textContent = r;
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--ink3)" }, svg);
    const ends = [];
    runs.forEach(([, l, c, dash, pts]) => {
      pts = pts.slice().sort((a, b) => a.round - b.round);
      el("path", { d: pts.map((p, i) => (i ? "L" : "M") + x(p.round) + " " + y(p.score_ratio)).join(""), fill: "none", stroke: c, "stroke-width": 2, "stroke-dasharray": dash, "stroke-linejoin": "round" }, svg);
      pts.forEach((p) => {
        const cx = x(p.round), cy = y(p.score_ratio), bad = p.verdict === "invalid";
        const g = el("g", { tabindex: 0, role: "button", "aria-label": `${p.model}: ${fmtX(p.score_ratio)} threshold, ${bad ? "would be rejected" : "valid"}` }, svg);
        g.style.cursor = "pointer";
        if (p.model === sel) el("circle", { cx, cy, r: 11, fill: "none", stroke: "var(--ink)", "stroke-width": 2 }, g);
        el("circle", { cx, cy, r: 5.5, fill: bad ? "var(--critical)" : c, stroke: "var(--surface)", "stroke-width": 2 }, g);
        el("circle", { cx, cy, r: 12, fill: "transparent" }, g);
        const go = () => { select(p.model); $("explain").scrollIntoView({ behavior: "smooth", block: "nearest" }); };
        g.addEventListener("click", go);
        g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go(); } });
        g.addEventListener("mousemove", (ev) => showTip(`<strong>${esc(p.model)}</strong><br>${fmtX(p.score_ratio)} threshold · ${pct(p.coverage)} coverage<br>${badge(p)}`, ev));
        g.addEventListener("mouseleave", hideTip);
      });
      const last = pts[pts.length - 1];
      ends.push({ x: x(last.round), y: y(last.score_ratio), label: l });
    });
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
    ends.forEach((e) => { const t = el("text", { x: e.x + 12, y: e.y + 4 }, svg); t.textContent = e.label; t.style.fill = "var(--ink)"; });
  }

  // ---------- quality chart ----------
  function renderQuality() {
    const q = D.quality;
    const box = $("quality");
    if (!q || !q.with) { $("qualityCard").querySelector(".hint").textContent = "Run with --quality to measure it."; box.innerHTML = ""; return; }
    const series = [["with", "With gate", "var(--ink)", ""], ["without", "Without gate", "var(--ink2)", "6 5"]].filter((s) => q[s[0]]);
    const W = Math.max(box.clientWidth, 280), H = 250, m = { l: 48, r: 96, t: 10, b: 30 };
    const all = series.flatMap((s) => q[s[0]].map((p) => p.mse));
    const R = Math.max(...series.flatMap((s) => q[s[0]].map((p) => p.round)));
    const lo = 0, hi = Math.max(...all) * 1.08;
    const x = (r) => m.l + (r - 1) / Math.max(R - 1, 1) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    box.innerHTML = "";
    const svg = el("svg", { width: W, height: H, role: "img", "aria-label": "Reconstruction error per round with and without the gate" }, box);
    const ticks = niceTicks(lo, hi, 4);
    ticks.forEach((v) => {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--grid)" }, svg);
      el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(3);
    });
    for (let r = 1; r <= R; r++) el("text", { x: x(r), y: H - 10, "text-anchor": "middle" }, svg).textContent = r;
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--ink3)" }, svg);
    const ends = [];
    series.forEach(([k, label, c, dash]) => {
      const pts = q[k];
      el("path", { d: pts.map((p, i) => (i ? "L" : "M") + x(p.round) + " " + y(p.mse)).join(""), fill: "none", stroke: c, "stroke-width": 2, "stroke-dasharray": dash, "stroke-linejoin": "round" }, svg);
      pts.forEach((p) => el("circle", { cx: x(p.round), cy: y(p.mse), r: 3.5, fill: c, stroke: "var(--surface)", "stroke-width": 2 }, svg));
      const last = pts[pts.length - 1];
      ends.push({ y: y(last.mse), label: `${label} ${last.mse.toFixed(3)}`, x: x(last.round) });
    });
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
    ends.forEach((e) => { const t = el("text", { x: e.x + 8, y: e.y + 4 }, svg); t.textContent = e.label; t.style.fill = "var(--ink)"; });
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--ink3)", "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (ev) => {
      const b = svg.getBoundingClientRect();
      const r = Math.min(R, Math.max(1, Math.round(1 + (ev.clientX - b.left - m.l) / (W - m.l - m.r) * (R - 1))));
      cross.setAttribute("x1", x(r)); cross.setAttribute("x2", x(r)); cross.setAttribute("visibility", "visible");
      showTip(`<strong>Round ${r}</strong><br>` + series.map(([k, label]) => { const p = q[k].find((p) => p.round === r); return p ? `${label}: ${p.mse.toFixed(4)}` : ""; }).join("<br>"), ev);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  }
  function niceTicks(lo, hi, n) {
    const step0 = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((k) => k * p).find((s) => s >= step0);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // ---------- decisions chart ----------
  $("decLegend").innerHTML = sources.map((s) => `<span><i class="sw" style="background:${color(s)}"></i>${nameOf(s)}${s === D.malicious ? " (attacker)" : ""}</span>`).join("") +
    `<span>● accepted</span><span>✕ rejected</span>` + (D.models.some((m) => m.offline) ? "<span>○ traced only</span>" : "");
  function renderDecisions() {
    const box = $("decisions");
    const W = Math.max(box.clientWidth, 280), H = 270, m = { l: 48, r: 14, t: 12, b: 30 };
    const R = D.rounds;
    const ratios = D.models.map((d) => d.score_ratio);
    const lo = Math.min(0.1, Math.min(...ratios) / 1.5), hi = Math.max(10, Math.max(...ratios) * 1.6);
    const L = Math.log10;
    const y = (v) => m.t + (1 - (L(v) - L(lo)) / (L(hi) - L(lo))) * (H - m.t - m.b);
    const bw = (W - m.l - m.r) / R;
    const x = (r, src) => m.l + (r - 0.5) * bw + (sources.indexOf(src) - (sources.length - 1) / 2) * Math.min(12, bw / (sources.length + 1));
    box.innerHTML = "";
    const svg = el("svg", { width: W, height: H, role: "img", "aria-label": "Gate distance per model and round" }, box);
    el("rect", { x: m.l, y: y(1), width: W - m.l - m.r, height: H - m.b - y(1), fill: "var(--zone)" }, svg);
    for (let e = Math.ceil(L(lo)); e <= Math.floor(L(hi)); e++) {
      const v = Math.pow(10, e);
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--grid)" }, svg);
      el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = (v < 1 ? v : v.toFixed(0)) + "×";
    }
    el("line", { x1: m.l, x2: W - m.r, y1: y(1), y2: y(1), stroke: "var(--ink2)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, svg);
    const tl = el("text", { x: m.l + 4, y: y(1) - 6, class: "halo" }, svg); tl.textContent = "threshold"; tl.style.fill = "var(--ink)";
    const vz = el("text", { x: m.l + 4, y: H - m.b - 6, class: "halo" }, svg); vz.textContent = "valid zone";
    for (let r = 1; r <= R; r++) el("text", { x: m.l + (r - 0.5) * bw, y: H - 10, "text-anchor": "middle" }, svg).textContent = r;
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--ink3)" }, svg);
    D.models.forEach((d) => {
      const cx = x(d.round, d.source), cy = y(d.score_ratio), c = color(d.source), s = state(d);
      const g = el("g", { tabindex: 0, role: "button", "aria-label": `${d.model}: ${fmtX(d.score_ratio)} threshold, ${s}` }, svg);
      g.style.cursor = "pointer";
      if (d.model === sel) el("circle", { cx, cy, r: 11, fill: "none", stroke: "var(--ink)", "stroke-width": 2 }, g);
      if (s === "rejected") {
        el("path", { d: `M${cx - 5.5} ${cy - 5.5}L${cx + 5.5} ${cy + 5.5}M${cx + 5.5} ${cy - 5.5}L${cx - 5.5} ${cy + 5.5}`, stroke: "var(--surface)", "stroke-width": 6, "stroke-linecap": "round" }, g);
        el("path", { d: `M${cx - 5.5} ${cy - 5.5}L${cx + 5.5} ${cy + 5.5}M${cx + 5.5} ${cy - 5.5}L${cx - 5.5} ${cy + 5.5}`, stroke: c, "stroke-width": 3, "stroke-linecap": "round" }, g);
      } else if (s === "trace") {
        el("circle", { cx, cy, r: 5, fill: "var(--surface)", stroke: c, "stroke-width": 2.5 }, g);
      } else {
        el("circle", { cx, cy, r: 5.5, fill: c, stroke: "var(--surface)", "stroke-width": 2 }, g);
      }
      el("circle", { cx, cy, r: 12, fill: "transparent" }, g);
      g.addEventListener("click", () => select(d.model));
      g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(d.model); } });
      g.addEventListener("mousemove", (ev) => showTip(`<strong>${esc(d.model)}</strong><br>${fmtX(d.score_ratio)} threshold · ${pct(d.coverage)} coverage<br>${badge(d)}`, ev));
      g.addEventListener("mouseleave", hideTip);
    });
  }

  // ---------- explanation ----------
  function symlog(v) { const L = 2; return Math.abs(v) <= L ? v / L : Math.sign(v) * (1 + Math.log10(Math.abs(v) / L)); }
  function gauge(title, value, label, thr, scale, good) {
    const W = 260, H = 44, m = { l: 6, r: 6 };
    const f = (v) => m.l + scale(v) * (W - m.l - m.r);
    const ok = good(value);
    const parts = [`<rect x="${m.l}" y="17" width="${W - m.l - m.r}" height="4" rx="2" fill="var(--grid)"/>`];
    const x0 = good(0) ? f(Math.max(scale.min, 0)) : f(thr), x1 = good(0) ? f(thr) : W - m.r;
    parts.push(`<rect x="${Math.min(x0, x1)}" y="13" width="${Math.abs(x1 - x0)}" height="12" rx="4" fill="var(--band)"/>`);
    parts.push(`<line x1="${f(thr)}" x2="${f(thr)}" y1="9" y2="29" stroke="var(--ink2)" stroke-width="1.5" stroke-dasharray="3 2"/>`);
    parts.push(`<text x="${f(thr)}" y="42" text-anchor="middle">threshold</text>`);
    parts.push(`<circle cx="${f(value)}" cy="19" r="7" fill="${ok ? "var(--good)" : "var(--critical)"}" stroke="var(--surface)" stroke-width="2.5"/>`);
    return `<div class="gauge"><div class="t">${title}</div><div class="n">${label} <span class="badge ${ok ? "ok" : "bad"}">${ok ? "✓ inside" : "✕ outside"}</span></div>` +
      `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none" aria-hidden="true">${parts.join("")}</svg></div>`;
  }
  function renderExplain() {
    const d = ALL.find((m) => m.model === sel);
    const box = $("explain");
    if (!d) { box.innerHTML = '<p class="empty">Select a decision.</p>'; return; }
    const lmin = Math.log10(0.1), lmax = Math.log10(Math.max(10, d.score_ratio * 2));
    const ls = (v) => (Math.log10(Math.max(v, 0.1)) - lmin) / (lmax - lmin);
    ls.min = 0.1;
    const cs = (v) => v; cs.min = 0;
    let html = `<div class="ex-head"><h2 class="mono">${esc(d.model)}</h2>${badge(d)}</div>` +
      `<dl class="facts"><dt>Decision</dt><dd>${esc(d.action)}</dd><dt>Why</dt><dd>${d.reasons.map(esc).join("; ")}</dd></dl>` +
      `<p class="para">${md(d.paragraph)}</p>` +
      `<div class="gauges">${gauge("Distance to nearest valid models", d.score_ratio, fmtX(d.score_ratio), 1, ls, (v) => v <= 1)}` +
      gauge("Features inside the valid range", d.coverage, pct(d.coverage), d.coverage_threshold, cs, (v) => v >= d.coverage_threshold) + "</div>";
    if (d.features.length) {
      const ok = d.verdict === "valid";
      const vals = d.features.flatMap((f) => [f.z, f.lo, f.hi]);
      const a = Math.min(symlog(Math.min(...vals)) - 0.25, symlog(-2)), b = Math.max(symlog(Math.max(...vals)) + 0.35, symlog(4));
      const W = 400, H = 26, pad = 6;
      const X = (v) => pad + (symlog(v) - a) / (b - a) * (W - 2 * pad - 60);
      const TICKS = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000].filter((t) => symlog(t) >= a && symlog(t) <= b);
      const grid = TICKS.map((t) => `<line x1="${X(t)}" x2="${X(t)}" y1="2" y2="24" stroke="var(--grid)"/>`).join("");
      html += `<h2 style="font-size:15px;margin:4px 0 2px">${ok ? "Trace features nearest the edge of the valid range" : "Trace features outside the valid range"}</h2>` +
        `<div class="legend"><span><i class="sw" style="background:var(--band);border-radius:3px;width:18px"></i>valid range (middle 95% of valid models, same rounds)</span><span><i class="sw" style="background:${ok ? "var(--ink2)" : "var(--critical)"}"></i>this model</span></div>`;
      d.features.forEach((f) => {
        const zero = Math.abs(f.hi - f.lo) < 1e-6;
        const band = zero ? `<line x1="${X(f.lo)}" x2="${X(f.lo)}" y1="5" y2="21" stroke="var(--ink2)" stroke-width="3" stroke-linecap="round"/>`
          : `<rect x="${X(f.lo)}" y="7" width="${Math.max(X(f.hi) - X(f.lo), 2)}" height="12" rx="3" fill="var(--band)"/>`;
        html += `<div class="feat"><div class="lab">${esc(f.label)}</div><div class="raw mono">${esc(f.feature)}</div>` +
          `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" aria-label="${esc(f.label)}: z ${f.z.toFixed(2)}, valid ${f.lo.toFixed(2)} to ${f.hi.toFixed(2)}">` +
          `${grid}${band}` +
          `<circle cx="${X(f.z)}" cy="13" r="6" fill="${ok ? "var(--ink2)" : "var(--critical)"}" stroke="var(--surface)" stroke-width="2"/>` +
          `<text x="${X(f.z) + 10}" y="17" style="fill:var(--ink)">z ${f.z.toFixed(2)}</text></svg>` +
          `<div class="note">valid ${f.lo.toFixed(2)} to ${f.hi.toFixed(2)} · ${esc(f.beyond)}</div></div>`;
      });
      html += `<svg viewBox="0 0 ${W} 18" width="100%" height="18" aria-hidden="true">` +
        TICKS.map((t) => `<text x="${X(t)}" y="12" text-anchor="middle">${t < 0 ? "−" + (-t) : t}</text>`).join("") +
        `</svg><p class="hint" style="margin:0 0 4px">standardised value z (log scale beyond ±2)</p>`;
    } else {
      html += '<p class="empty">Every trace feature is inside the valid range.</p>';
    }
    if (d.nearest.length) {
      html += `<h2 style="font-size:15px;margin:12px 0 4px">Closest valid precedents</h2><ul class="near">` +
        d.nearest.map((n) => `<li><span class="mono">${esc(n.model)}</span> — nearest on ${n.probe_images} probe images</li>`).join("") + "</ul>";
    }
    const same = ALL.filter((m) => m.round === d.round && m.model !== d.model);
    html += `<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px">` +
      `<button class="navb" id="prev" type="button">← Previous</button><button class="navb" id="next" type="button">Next →</button>` +
      (same.length ? `<span class="hint" style="margin:0 0 0 6px">Same round:</span>` + same.map((m) => `<button class="chip" type="button" data-m="${esc(m.model)}">${esc(label(m))}</button>`).join("") : "") + "</div>";
    box.innerHTML = html;
    box.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => select(b.dataset.m)));
    $("prev").addEventListener("click", () => step(-1));
    $("next").addEventListener("click", () => step(1));
  }
  function step(k) {
    const list = filtered().some((m) => m.model === sel) ? filtered() : ALL;
    const i = list.findIndex((m) => m.model === sel);
    select(list[(i + k + list.length) % list.length].model);
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.target.closest && ev.target.closest("input, textarea")) return;
    if (ev.key === "ArrowRight") step(1);
    if (ev.key === "ArrowLeft") step(-1);
  });

  // ---------- table ----------
  const FILTERS = [["all", "All"], ["rejected", "Flagged"]].concat(sources.map((s) => [s, nameOf(s)]))
    .concat(BASE.length ? [["baseline", "Checkpoints, no gate"]] : []);
  function renderFilters() {
    $("filters").innerHTML = FILTERS.map(([k, l]) => `<button class="chip" type="button" data-f="${esc(k)}" aria-pressed="${k === filter}">${esc(l)}</button>`).join("");
    $("filters").querySelectorAll("[data-f]").forEach((b) => b.addEventListener("click", () => { filter = b.dataset.f; renderFilters(); renderTable(); }));
  }
  function renderTable() {
    const rows = filtered();
    $("rows").innerHTML = rows.length ? rows.map((m) => `<tr tabindex="0" data-m="${esc(m.model)}" aria-selected="${m.model === sel}">` +
      `<td>${m.round}</td><td><i class="sw" style="background:${color(m.source)};margin-right:6px"></i>${esc(label(m))}</td><td>${badge(m)}</td>` +
      `<td class="num">${fmtX(m.score_ratio)}</td><td class="num">${pct(m.coverage)}</td></tr>`).join("")
      : '<tr><td colspan="5" class="empty">No decisions match this filter.</td></tr>';
    $("rows").querySelectorAll("tr[data-m]").forEach((tr) => {
      tr.addEventListener("click", () => select(tr.dataset.m));
      tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter") select(tr.dataset.m); });
    });
  }

  // ---------- reconstructions, steps, footer ----------
  (function () {
    if (!D.recon) { $("reconCard").style.display = "none"; return; }
    const cols = [["noisy", "Input (noisy)"], ["without_gate", "Without gate"], ["with_gate", "With gate"], ["clean", "Target (clean)"]];
    $("recon").innerHTML = D.recon.map((r) => cols.map(([k, l]) => r[k] ? `<figure><img src="${r[k]}" alt="${l}"><figcaption>${l}</figcaption></figure>` : "<div></div>").join("")).join("");
  })();
  $("steps").innerHTML = [
    ["1 · Trace", "The server runs every client update on the same probe images and records its SL trace: 98 statistics of how each layer family activates."],
    ["2 · Retrieve", `It finds the closest traces among ${D.n_valid ? D.n_valid + " " : ""}known-valid models at the same stage of training${D.n_base_runs ? `, collected from ${D.n_base_runs} clean training runs` : ""}.`],
    ["3 · Decide and explain", `If the update is further from every valid precedent than the threshold allows${D.target_fpr ? ` (set for a ${(D.target_fpr * 100).toFixed(1)}% false-alarm rate)` : ""}, it is rejected. The trace features that moved are the explanation.`],
  ].map(([n, p]) => `<div><div class="n">${n}</div><p>${p}</p></div>`).join("");
  $("foot").innerHTML = `Run <span class="mono">${esc(D.run)}</span>${D.ungated ? ` · baseline <span class="mono">${esc(D.ungated)}</span>` : ""} · case base <span class="mono">${esc(D.case_base || "-")}</span> · generated ${esc(D.generated)}`;

  // ---------- theme ----------
  $("theme").addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (!document.documentElement.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    try { localStorage.setItem("slgate-theme", dark ? "light" : "dark"); } catch (e) {}
  });
  try { const t = localStorage.getItem("slgate-theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (e) {}

  function renderAll() { renderQuality(); renderDecisions(); renderCheckpoints(); renderExplain(); renderFilters(); renderTable(); }
  let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { renderQuality(); renderDecisions(); renderCheckpoints(); }, 120); });
  renderAll();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
