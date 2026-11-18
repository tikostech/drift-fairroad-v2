"""
This repo carries no IP: the probes, gates and CBR live in drift-fairroad-service.

  1. no IP definition (top-level class/def of the probes, gate, CBR, SL minimisation) in any source file
  2. no compiled bytecode of an IP module (a tracked .pyc is the IP in another form)
  3. no IP package directories (sc/, ip_services/)
  4. ip_client/ is the vendored copy recorded in ip_client/VENDORED.txt, and matches the service repo's
     (when ../drift-fairroad-service is checked out)

    python -m integration_tests.check_no_ip
"""
import hashlib
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_REPO = os.environ.get("IP_SERVICE_REPO") or os.path.join(os.path.dirname(REPO_ROOT), "drift-fairroad-service")

# Kept in step with drift-fairroad-service/tests/check_ip_core.py (which checks they're all defined there).
# Indented methods of the same name (e.g. ip_client's HTTP SLServiceClient.minimize) are wrappers and
# aren't matched.
IP_DEFS = ("SLCaseBase", "ModelTracer", "CAESLProbe", "SLMinProbe", "SLMinBatchProbe", "ActivationProbe",
           "BatchActivationProbe", "SynapsesLoggerV49", "SequentialCollection", "SequentialCollectionCase",
           "SequentialCollectionAssociation", "minimize", "batch_layer_stats", "_layer_stats",
           "_batch_layer_stats", "build_case_base", "held_out_scores", "held_out_fpr", "load_probe_images",
           "parse_names", "probe_name")
DEF_RE = re.compile(rf"^(class|def) ({'|'.join(IP_DEFS)})\b", re.M)
IP_MODULES = re.compile(r"^(sl_gate|sl_probe_cae|case_base_builder|sl_minimize|activation_probe|SynapsesLogger\w*|"
                        r"SequentialCollection\w*|ModelSpec)\.", re.I)
SKIP_DIRS = {".git", ".venv", ".venv-gpu", "data", "paper", ".idea", ".claude", "node_modules"}


def walk():
    for d, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            yield os.path.join(d, f)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ok = True
    rel = lambda p: os.path.relpath(p, REPO_ROOT)  # noqa: E731

    print("1. IP definitions in source files")
    hits = [f"{rel(p)}: {m.group(0)}" for p in walk() if p.endswith((".py", ".pyx", ".ipynb"))
            for m in DEF_RE.finditer(open(p, encoding="utf-8", errors="replace").read())]
    print("   none" if not hits else "\n".join(f"   FOUND {h}" for h in hits))
    ok &= not hits

    print("2. compiled bytecode of IP modules")
    pyc = [rel(p) for p in walk() if p.endswith(".pyc") and IP_MODULES.match(os.path.basename(p))]
    print("   none" if not pyc else "\n".join(f"   FOUND {p}" for p in pyc))
    ok &= not pyc

    print("3. IP package directories")
    dirs = [d for d in ("sc", "ip_services") if os.path.isdir(os.path.join(REPO_ROOT, d))]
    print("   none" if not dirs else "\n".join(f"   FOUND {d}/" for d in dirs))
    ok &= not dirs

    print("4. vendored ip_client")
    mine = os.path.join(REPO_ROOT, "ip_client", "__init__.py")
    recorded = re.search(r"^([0-9a-f]{64})\s+__init__\.py$",
                         open(os.path.join(REPO_ROOT, "ip_client", "VENDORED.txt"), encoding="utf-8").read(), re.M)
    same = bool(recorded) and sha(mine) == recorded.group(1)
    print(f"   {'ok ' if same else 'CHANGED'} matches VENDORED.txt")
    ok &= same
    theirs = os.path.join(SERVICE_REPO, "ip_client", "__init__.py")
    if os.path.isfile(theirs):
        same = sha(mine) == sha(theirs)
        print(f"   {'ok ' if same else 'DRIFTED'} matches {theirs}")
        ok &= same
    else:
        print(f"   (service repo not found at {SERVICE_REPO}; skipped)")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
