"""
Full integration: the FL system trained end to end with its SL gate served by drift-fairroad-service.

Needs the SL and CBR services running (locally or in Docker). Then:

  1. checks the services are ready and the case base fits the SL service's probe set and features
  2. runs run_fl_cae_system.py with --sl-service-url / --cbr-service-url (optionally with an attacker)
  3. reads the run's sl_gate_log.jsonl and sl_traces.npz and checks
       - every model was traced and every in-scope model decided through the services, with no errors
       - honest client updates rejected <= --max-false-rejections
       - with an attack, the attacker's updates rejected in >= --min-attacker-rejection of rounds
       - the trace archive reads back with labels (trace_store.load_run) and holds every model

    python -m integration_tests.check_integration --api-key KEY --rounds 5 --attack scale

The runner's plot files under fl-payload/plots are restored afterwards, so the check leaves the
working tree as it found it.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL_PAYLOAD = os.path.join(REPO_ROOT, "fl-payload")
sys.path[:0] = [FL_PAYLOAD, REPO_ROOT]

from ip_client import CBRServiceClient, SLServiceClient  # noqa: E402
from trace_store import load_run  # noqa: E402

# What the measured gate catches reliably (see drift-fairroad-service paper/REPORT.md): scale and noise attacks; sign-flip and
# inverted training barely harm the model and mostly pass, so no minimum is asserted for them.
DEFAULT_MIN_ATTACKER = {"scale": 0.8, "noise": 0.5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sl", default=os.environ.get("SL_SERVICE_URL") or "http://127.0.0.1:8601")
    ap.add_argument("--cbr", default=os.environ.get("CBR_SERVICE_URL") or "http://127.0.0.1:8602")
    ap.add_argument("--api-key", default=os.environ.get("IP_SERVICE_API_KEY"))
    ap.add_argument("--case-base", default="case_base_v2", help="Case base name on the CBR service")
    ap.add_argument("--scope", choices=["both", "checkpoint", "clients"], default="clients")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--attack", choices=["none", "sign_flip", "scale", "noise", "inverted"], default="scale")
    ap.add_argument("--malicious-client", default="client_3")
    ap.add_argument("--max-false-rejections", type=int, default=0)
    ap.add_argument("--min-attacker-rejection", type=float, default=None,
                    help="Share of rounds the attacker must be rejected in (default: 0.8 scale, 0.5 noise, else 0)")
    ap.add_argument("--out", default=None, help="Run directory (default: a new temp dir)")
    args = ap.parse_args()
    attack = None if args.attack == "none" else args.attack
    min_attacker = (args.min_attacker_rejection if args.min_attacker_rejection is not None
                    else DEFAULT_MIN_ATTACKER.get(attack, 0.0))
    out = os.path.abspath(args.out or tempfile.mkdtemp(prefix="fl_integration_"))
    os.makedirs(out, exist_ok=True)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)

    print("1. Services")
    sl, cbr = SLServiceClient(args.sl, api_key=args.api_key), CBRServiceClient(args.cbr, api_key=args.api_key)
    sl.wait_ready(60)
    cbr.wait_ready(60)
    info = sl.info()
    desc = cbr.describe(args.case_base)
    check("SL and CBR services ready", True, f"{info['service']} ({info['probe']['n_images']} probe images, "
          f"{info['features']['n']} features); {desc['name']} ({desc['n_models']} valid models)")
    check("case base fits the SL service's probe set and features",
          cbr.check(args.case_base, info["features"]["names"], info["probe"]["probe_hash"])["compatible"],
          f"probe hash {info['probe']['probe_hash']}")

    print(f"2. Training {args.rounds} round(s), attack={attack or 'none'}, gate scope={args.scope} -> {out}")
    plots = os.path.join(FL_PAYLOAD, "plots")
    saved = {f: open(os.path.join(plots, f), "rb").read() for f in os.listdir(plots)} if os.path.isdir(plots) else {}
    cmd = [sys.executable, "run_fl_cae_system.py", "--num-rounds", str(args.rounds), "--checkpoint-dir", out,
           "--sl-case-base", args.case_base, "--sl-gate-scope", args.scope, "--no-final-eval",
           "--sl-service-url", args.sl, "--cbr-service-url", args.cbr]
    if attack:
        cmd += ["--attack", attack, "--malicious-client", args.malicious_client]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if args.api_key:
        env["IP_SERVICE_API_KEY"] = args.api_key
    t0 = time.time()
    try:
        with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=FL_PAYLOAD, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=1800).returncode
    finally:
        for f, data in saved.items():  # the runner rewrites tracked plot files
            with open(os.path.join(plots, f), "wb") as fh:
                fh.write(data)
        for f in set(os.listdir(plots)) - set(saved) if os.path.isdir(plots) else ():
            os.remove(os.path.join(plots, f))
    check("training run completed", rc == 0, f"exit {rc}, {time.time() - t0:.0f} s (log: {out}/run.log)")

    print("3. Gate decisions and trace archive")
    log_path = os.path.join(out, "sl_gate_log.jsonl")
    if not os.path.exists(log_path):
        check("gate log written", False, log_path)
        sys.exit(1)
    recs = [json.loads(line) for line in open(log_path, encoding="utf-8")]
    updates = [r for r in recs if r["source"] != "global"]
    ckpts = [r for r in recs if r["source"] == "global"]
    errors = [r for r in recs if r.get("error")]
    in_scope = [r for r in recs if args.scope == "both" or (args.scope == "clients") == (r["source"] != "global")]
    rounds = sorted({r["round"] for r in recs})

    print(f"   {'round':>5}  " + "  ".join(f"{c:>18}" for c in sorted({r['source'] for r in updates})) + "  checkpoint")
    for rnd in rounds:
        cells = []
        for c in sorted({r["source"] for r in updates}):
            r = next((x for x in updates if x["round"] == rnd and x["source"] == c), None)
            cells.append("-" if r is None else
                         f"{(r['verdict'] or 'ungated'):>7} {r['score_ratio'] or 0:6.2f}x{'' if r['accepted'] else ' REJ'}")
        g = next((x for x in ckpts if x["round"] == rnd), None)
        cells.append("-" if g is None else ("kept" if g["accepted"] else "rolled back"))
        print(f"   {rnd:>5}  " + "  ".join(f"{c:>18}" for c in cells))

    check("every round traced (client updates + checkpoint)", len(rounds) == args.rounds and len(ckpts) == args.rounds,
          f"{len(updates)} client updates, {len(ckpts)} checkpoints over {len(rounds)} rounds")
    check("no service errors", not errors, f"{len(errors)} records with an error")
    check(f"every in-scope model decided by the CBR service ({args.scope})",
          all(r["gated"] and r["verdict"] for r in in_scope), f"{len(in_scope)} models")
    honest = [r for r in updates if r["source"] != (args.malicious_client if attack else None)]
    false_rej = [r for r in honest if not r["accepted"]]
    check("honest client updates rejected", len(false_rej) <= args.max_false_rejections,
          f"{len(false_rej)}/{len(honest)} (allowed {args.max_false_rejections})")
    if attack:
        bad = [r for r in updates if r["source"] == args.malicious_client]
        caught = [r for r in bad if not r["accepted"]]
        rate = len(caught) / max(len(bad), 1)
        check("attacker's updates rejected", rate >= min_attacker,
              f"{len(caught)}/{len(bad)} rounds ({rate:.0%}; required {min_attacker:.0%})")
    run = load_run(out)
    traced = {r["model"] for r in recs if not r.get("error")}
    check("trace archive reads back with labels", {m["model"] for m in run["models"]} == traced,
          f"{len(run['models'])} models, {run['Z'].shape[0]} trace rows, probe hash {run['probe_hash']}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
