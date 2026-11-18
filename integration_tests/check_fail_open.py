"""
The FL server's behaviour when an IP service call fails mid-run (service down, or the case base
removed): the model is admitted ungated, the error is logged and recorded in sl_gate_log.jsonl,
and training continues (fail-open).

Needs both IP services running (drift-fairroad-service); run from this repo's root:

    python -m integration_tests.check_fail_open --api-key KEY
"""
import argparse
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL_PAYLOAD = os.path.join(REPO_ROOT, "fl-payload")
sys.path[:0] = [FL_PAYLOAD, REPO_ROOT]
os.chdir(FL_PAYLOAD)  # config paths are relative to fl-payload/

from fl_server import FLServer  # noqa: E402

DEAD = "http://127.0.0.1:9"  # discard port: nothing listens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", default="http://localhost:8601")
    ap.add_argument("--cbr", default="http://localhost:8602")
    ap.add_argument("--api-key", default=os.environ.get("IP_SERVICE_API_KEY"))
    ap.add_argument("--case-base", default="case_base_v2")
    args = ap.parse_args()

    out = tempfile.mkdtemp(prefix="fail_open_")
    srv = FLServer("127.0.0.1", 0, 3, 2, "fedavg", checkpoint_dir=out, sl_case_base=args.case_base,
                   sl_gate_scope="clients", sl_service_url=args.sl, cbr_service_url=args.cbr, ip_api_key=args.api_key)
    weights = srv.model.get_weights()
    ok = True

    t = srv._trace_and_decide(weights, 1, "client_1")
    print(f"services up:   gated={t['record']['gated']} verdict={t['record']['verdict']}")
    ok &= t["record"]["gated"] and t["decision"] is not None

    srv.case_base.name = "removed_mid_run"  # the CBR service answers 404
    t = srv._trace_and_decide(weights, 1, "client_1")
    srv._commit(t, True)
    print(f"case base gone: gated={t['record']['gated']} traced={t['Z'] is not None} error={t['record'].get('error', '')[:60]!r}")
    ok &= (not t["record"]["gated"]) and t["Z"] is not None and not t["invalid"] and "404" in t["record"]["error"]
    srv.case_base.name = args.case_base

    srv.case_base.client.base, srv.case_base.client.retries = DEAD, 1
    t = srv._trace_and_decide(weights, 1, "client_2")
    srv._commit(t, True)
    print(f"CBR down:      gated={t['record']['gated']} traced={t['Z'] is not None} error={t['record'].get('error', '')[:60]!r}")
    ok &= (not t["record"]["gated"]) and t["Z"] is not None and not t["invalid"] and "error" in t["record"]

    srv.tracer.client.base, srv.tracer.client.retries = DEAD, 1
    t = srv._trace_and_decide(weights, 1, "client_3")
    srv._commit(t, True)
    print(f"SL down:       gated={t['record']['gated']} traced={t['Z'] is not None} error={t['record'].get('error', '')[:60]!r}")
    ok &= (not t["record"]["gated"]) and t["Z"] is None and not t["invalid"]

    log = [json.loads(line) for line in open(os.path.join(out, "sl_gate_log.jsonl"))]
    print(f"gate log: {len(log)} records, {sum('error' in r for r in log)} with the outage recorded")
    ok &= len(log) == 3 and all("error" in r and r["accepted"] for r in log)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
