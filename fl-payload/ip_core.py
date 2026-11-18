"""
Finds the project's core IP -- the SL minimisation (``sc/``), the CAE probe (``sl_probe_cae``), the
CBR gate (``sl_gate``) and ``case_base_builder`` -- for fl-payload's in-process tools:
``import ip_core`` before ``from sl_gate import ...``.

The IP is not in this repo. It lives in the drift-fairroad-service repo (ip_services/core), found via
$DRIFT_FAIRROAD_IP_CORE, or else the sibling checkout ../drift-fairroad-service. Deployments without it
(the FL server image) use the web services instead (fl_server.py --sl-service-url / --cbr-service-url);
importing this raises ImportError there.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [os.environ.get("DRIFT_FAIRROAD_IP_CORE"),
               os.path.join(os.path.dirname(_REPO), "drift-fairroad-service", "ip_services", "core")]
CORE_DIR = next((p for p in _CANDIDATES if p and os.path.isfile(os.path.join(p, "sl_gate.py"))), None)
if CORE_DIR is None:
    raise ImportError("IP core not found: check out drift-fairroad-service next to this repo, or set "
                      "DRIFT_FAIRROAD_IP_CORE to its ip_services/core -- or use the IP web services "
                      "(--sl-service-url / --cbr-service-url)")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
