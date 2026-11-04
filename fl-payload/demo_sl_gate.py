"""
Live demo of the SL gate. The same poisoning attack is trained twice, without and with the gate; the
console shows each round and every gate decision as it happens, then the results and the gate's
explanations open in the browser (sl_gate_demo.html, one self-contained file).

    .venv-gpu/Scripts/python demo_sl_gate.py               # scaling attack by client_3, ~4 min on the GPU
    .venv-gpu/Scripts/python demo_sl_gate.py --pause       # wait for Enter between steps (when presenting)
    .venv-gpu/Scripts/python demo_sl_gate.py --reuse       # no training: rebuild the page from the last runs
    .venv-gpu/Scripts/python demo_sl_gate.py --attack sign_flip

Talk track: DEMO.md.
"""
import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 4
GATE = re.compile(r"SL gate (r\d+)/(\S+): (VALID|INVALID) \(score ([\d.]+)x threshold, coverage ([\d.]+)\)(.*)")
ROUND = re.compile(r"Round (\d+) complete - Avg Loss: ([\d.]+)")
ATTACKER = re.compile(r"MALICIOUS CLIENT: (.*)")


def say(text=""):
    print(text, flush=True)


def banner(i, title, *lines):
    say("\n" + "=" * 78)
    say(f" STEP {i}/{STEPS}  {title}")
    for line in lines:
        say(f"   {line}")
    say("=" * 78)


def pause(args, what="continue"):
    if args.pause:
        input(f"\n   [Enter] to {what} ... ")


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_fl(run_dir, extra, rounds):
    """Train one run, streaming a condensed view; the full output goes to <run_dir>.log."""
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    cmd = [sys.executable, "-u", "run_fl_cae_system.py", "--num-rounds", str(rounds),
           "--checkpoint-dir", run_dir, "--no-final-eval"] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(run_dir + ".log", "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            log.write(line)
            if m := ATTACKER.search(line):
                say(f"   ! malicious client: {m.group(1)}")
            elif m := GATE.search(line):
                r, who, verdict, ratio, cov, rest = m.groups()
                action = "REJECTED" if "REJECTED" in rest else ("admitted" if "admit" in rest else "accepted")
                mark = "x" if action == "REJECTED" else ("!" if verdict == "INVALID" else " ")
                say(f"     {mark} {r:<4} {who:<9} {verdict:<8} {float(ratio):>8.2f}x threshold   "
                    f"coverage {float(cov):.0%}   -> {action}")
            elif (m := ROUND.search(line)) and m.group(1) != "0":
                say(f"   Round {m.group(1):>2} complete   avg training loss {float(m.group(2)):.4f}")
            elif "Port" in line and "in use" in line or line.startswith("✗"):
                say("   " + line.rstrip())
        p.wait()
    say(f"   ... finished in {time.time() - t0:.0f} s (log: {os.path.relpath(run_dir + '.log', HERE)})")
    if p.returncode != 0 or not os.path.exists(os.path.join(run_dir, f"server_round_{rounds}.pt")):
        raise SystemExit(f"Training failed: see {run_dir}.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", choices=["scale", "sign_flip", "noise", "inverted"], default="scale")
    ap.add_argument("--attack-scale", type=float, default=None, help="Attack strength (default: the client's)")
    ap.add_argument("--rounds", type=int, default=10, help="Federated rounds (the case base covers 1-10)")
    ap.add_argument("--case-base", default="sl_runs_gpu/case_base_v2.npz", help="Case base of valid models (gate v6)")
    ap.add_argument("--out", default="sl_runs_demo", help="Where the runs and the page go")
    ap.add_argument("--reuse", action="store_true", help="Skip training; rebuild the page from existing runs")
    ap.add_argument("--pause", action="store_true", help="Wait for Enter between steps")
    ap.add_argument("--no-open", action="store_true", help="Do not open the browser")
    args = ap.parse_args()
    os.chdir(HERE)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    import config
    import torch
    from sl_demo_page import _evaluator, build_page, page_data, reconstructions, round_quality

    ungated, gated = os.path.join(args.out, "ungated"), os.path.join(args.out, "gated")
    attack = ["--attack", args.attack] + (["--attack-scale", str(args.attack_scale)] if args.attack_scale else [])
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slower: torch has no CUDA here)"
    say(f"\nSL gate demo -- attack: {args.attack} by client_3, {args.rounds} rounds, device: {device}")
    if not os.path.exists(args.case_base):
        raise SystemExit(f"Case base not found: {args.case_base}")
    if not args.reuse and port_in_use(config.SERVER_PORT):
        raise SystemExit(f"Port {config.SERVER_PORT} is in use: another FL run is active. Wait for it to finish.")
    os.makedirs(args.out, exist_ok=True)

    if args.reuse:
        for d in (ungated, gated):
            if not os.path.exists(os.path.join(d, "sl_gate_log.jsonl")):
                raise SystemExit(f"--reuse: {d} has no finished run")
        say("Reusing the last runs (no training).")
    else:
        banner(1, "Federated training WITHOUT the gate",
               "3 clients train a denoising autoencoder; client_3 is poisoned and sends manipulated updates.",
               "The server averages every update it receives.")
        pause(args, "start training")
        run_fl(ungated, ["--sl-traces"] + attack, args.rounds)

        banner(2, "The same attack WITH the SL gate",
               "Before averaging, the server traces each update on 32 probe images and compares the trace",
               "with valid models from the case base. Updates too far from every valid precedent are rejected.",
               "Legend:  x = rejected   ! = flagged but admitted   (blank) = accepted")
        pause(args, "start training")
        run_fl(gated, ["--sl-case-base", args.case_base, "--sl-gate-scope", "clients"] + attack, args.rounds)

    banner(3, "Measuring the models", "Reconstruction error (MSE) of every round's model on a fixed validation set.")
    pause(args)
    ev = _evaluator()
    quality = round_quality([ungated, gated], ev)
    recon = reconstructions([ungated, gated], ev=ev)
    say(f"\n   {'round':>5}  {'without gate':>12}  {'with gate':>9}")
    for a, b in zip(quality[ungated], quality[gated]):
        say(f"   {a['round']:>5}  {a['mse']:>12.4f}  {b['mse']:>9.4f}")

    banner(4, "Explaining the gate's decisions", "Opening the demo page.")
    data = page_data(gated, ungated, quality, recon)
    page = build_page(data, os.path.join(args.out, "sl_gate_demo.html"))
    ups = [m for m in data["models"] if m["kind"] == "client_update"]
    bad = [m for m in ups if m["source"] == data["malicious"]]
    good = [m for m in ups if m["source"] != data["malicious"]]
    w, wo = quality[gated][-1]["mse"], quality[ungated][-1]["mse"]
    say(f"\n   Attacker updates rejected:  {sum(m['rejected'] for m in bad)}/{len(bad)}")
    say(f"   Honest updates accepted:    {sum(not m['rejected'] for m in good)}/{len(good)}")
    say(f"   Final MSE:                  {w:.4f} with the gate vs {wo:.4f} without ({(1 - w / wo):.0%} lower)")
    say(f"   Page:                       {os.path.abspath(page)}")
    if not args.no_open:
        webbrowser.open("file:///" + os.path.abspath(page).replace("\\", "/"))


if __name__ == "__main__":
    main()
