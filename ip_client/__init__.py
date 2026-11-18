"""
HTTP clients for the SL and CBR web services (see ip_services/).

This package holds no IP: it serialises weights and trace matrices and calls the services. It is
what the main project imports instead of ``sc``, ``sl_probe_cae`` and ``sl_gate``.

Low level::

    sl = SLServiceClient("http://localhost:8601", api_key=...)
    Z, names, probe_hash = sl.trace(state_dict)
    cbr = CBRServiceClient("http://localhost:8602", api_key=...)
    decision = cbr.decide("case_base_v2", Z, round=4)

Drop-in replacements for the in-process objects the FL server uses::

    RemoteModelTracer(sl)            ~ sl_gate.ModelTracer   (.trace(state_dict), .names, .probe_hash)
    RemoteCaseBase(cbr, "name")      ~ sl_gate.SLCaseBase    (.decide(Z, round), .check_compatible, .meta)

The key is ``api_key=``, else ``$IP_SERVICE_API_KEY``, else ``DEFAULT_API_KEY`` -- the services' own
default, so an unconfigured local setup works. Pass ``api_key=""`` to send no key.

Transient failures (connection errors, 502/503/504) are retried with backoff and then raised as
``ServiceUnavailable``; anything the service rejects is raised as ``ServiceError`` with its status
and error code.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid

import numpy as np
import requests

__all__ = ["ServiceError", "ServiceUnavailable", "SLServiceClient", "CBRServiceClient",
           "RemoteModelTracer", "RemoteCaseBase", "DEFAULT_API_KEY", "DEFAULT_ADMIN_KEY"]

# The services' keys when IP_SERVICE_API_KEY / IP_SERVICE_ADMIN_KEY are unset (ip_services/common.py).
DEFAULT_API_KEY = "ipsev001"
DEFAULT_ADMIN_KEY = "ipsev001ADMIN"

RETRY_STATUS = (502, 503, 504)


class ServiceError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, detail: dict | None = None, url: str = ""):
        super().__init__(f"{url}: HTTP {status} {code}: {message}")
        self.status, self.code, self.message, self.detail = status, code, message, detail or {}


class ServiceUnavailable(ServiceError):
    """The service could not be reached, or kept answering 5xx, after all retries."""


class _Http:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 60.0, retries: int = 3,
                 backoff: float = 0.5):
        self.base = base_url.rstrip("/")
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.session = requests.Session()
        key = api_key if api_key is not None else (os.environ.get("IP_SERVICE_API_KEY") or DEFAULT_API_KEY)
        if key:
            self.session.headers["X-API-Key"] = key

    def request(self, method: str, path: str, **kw):
        url = f"{self.base}{path}"
        headers = dict(kw.pop("headers", {}) or {}, **{"X-Request-ID": uuid.uuid4().hex[:12]})
        last = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                r = self.session.request(method, url, timeout=self.timeout, headers=headers, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = ServiceUnavailable(0, "unreachable", f"{type(e).__name__}: {e}", url=url)
                continue
            if r.status_code in RETRY_STATUS:
                last = ServiceUnavailable(r.status_code, "unavailable", r.text[:200], url=url)
                continue
            if r.status_code >= 400:
                try:
                    err = r.json()["error"]
                except Exception:
                    err = {"code": "http_error", "message": r.text[:200]}
                detail = {k: v for k, v in err.items() if k not in ("code", "message")}
                cls = ServiceUnavailable if r.status_code >= 500 else ServiceError
                raise cls(r.status_code, err.get("code", "http_error"), err.get("message", ""), detail, url)
            return r.json()
        raise last

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)

    def healthy(self) -> bool:
        try:
            return self.session.get(f"{self.base}/readyz", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while not self.healthy():
            if time.time() > deadline:
                raise ServiceUnavailable(0, "not_ready", f"not ready after {timeout:.0f} s", url=self.base)
            time.sleep(0.5)


def _rows(Z) -> list:
    """Float64 rows for JSON. Python's float repr round-trips exactly, so the service sees the same bits."""
    return np.asarray(Z, dtype=np.float64).tolist()


# ---------------------------------------------------------------------------
# SL minimise / probe
# ---------------------------------------------------------------------------

class SLServiceClient(_Http):
    def info(self) -> dict:
        return self.get("/v1/info")

    def trace(self, state_dict) -> tuple[np.ndarray, list[str], str]:
        """SL traces of one model over the service's probe set: ``(Z, names, probe_hash)``."""
        import torch  # only needed to serialise weights
        buf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, buf)
        out = self.post("/v1/trace", data=buf.getvalue(), headers={"Content-Type": "application/octet-stream"})
        return np.asarray(out["Z"], dtype=np.float64), out["names"], out["probe_hash"]

    def minimize(self, X, names, scope: str = "role", bands: int = 1, stats=None) -> tuple[np.ndarray, list[str]]:
        body = dict(X=_rows(X), names=list(names), scope=scope, bands=bands)
        if stats is not None:
            body["stats"] = list(stats)
        out = self.post("/v1/minimize", json=body)
        return np.asarray(out["Z"], dtype=np.float64), out["names"]


class RemoteModelTracer:
    """``sl_gate.ModelTracer`` over HTTP: ``trace(state_dict) -> Z``, with ``names`` and ``probe_hash``."""

    def __init__(self, client: SLServiceClient):
        self.client = client
        info = client.info()
        self.probe_files = list(info["probe"]["files"])
        self.probe_hash = info["probe"]["probe_hash"]
        self.names: list[str] = list(info["features"]["names"])
        self.info = info

    def trace(self, state_dict) -> np.ndarray:
        Z, names, probe_hash = self.client.trace(state_dict)
        if probe_hash != self.probe_hash:
            raise ServiceError(409, "probe_changed", "the SL service's probe set changed mid-run",
                               url=self.client.base)
        self.names = names
        return Z


# ---------------------------------------------------------------------------
# CBR gate
# ---------------------------------------------------------------------------

class CBRServiceClient(_Http):
    def list(self) -> list[dict]:
        return self.get("/v1/case-bases")["case_bases"]

    def describe(self, name: str) -> dict:
        return self.get(f"/v1/case-bases/{name}")

    def check(self, name: str, names, probe_hash: str) -> dict:
        return self.post(f"/v1/case-bases/{name}/compatibility", json=dict(names=list(names), probe_hash=probe_hash))

    def decide(self, name: str, Z, round: int, names=None, probe_hash: str | None = None) -> dict:
        body = dict(Z=_rows(Z), round=int(round))
        if names is not None:
            body.update(names=list(names), probe_hash=probe_hash)
        return self.post(f"/v1/case-bases/{name}/decide", json=body)

    def upload(self, name: str, path: str) -> dict:
        with open(path, "rb") as f:
            data = f.read()  # bytes, not the file: a retry must resend the whole body
        return self.request("PUT", f"/v1/case-bases/{name}", data=data,
                            headers={"Content-Type": "application/octet-stream"})

    def delete(self, name: str) -> dict:
        return self.request("DELETE", f"/v1/case-bases/{name}")

    def build(self, name: str, Z, model_ids, rounds, images, groups, names, probe_hash: str, **params) -> dict:
        """Fit and store a case base from valid models' traces; sent as .npz (trace sets get large)."""
        buf = io.BytesIO()
        np.savez(buf, Z=np.asarray(Z, dtype=np.float64), model_ids=np.asarray(model_ids).astype(str),
                 rounds=np.asarray(rounds, int), images=np.asarray(images, int), groups=np.asarray(groups).astype(str),
                 names=np.asarray(list(names)), probe_hash=probe_hash, params=json.dumps(params))
        return self.post(f"/v1/case-bases/{name}/build", data=buf.getvalue(),
                         headers={"Content-Type": "application/x-npz"})


class RemoteCaseBase:
    """``sl_gate.SLCaseBase`` over HTTP, for the calls the FL server makes."""

    def __init__(self, client: CBRServiceClient, name: str):
        self.client, self.name = client, name
        d = client.describe(name)
        self.meta = dict(d.get("meta") or {}, probe_hash=d.get("probe_hash"), n_models=d.get("n_models"))
        self.names = d.get("feature_names", [])
        self.description = d

    def check_compatible(self, names, probe_hash: str) -> None:
        try:
            self.client.check(self.name, names, probe_hash)
        except ServiceError as e:
            if e.status == 409:
                raise ValueError(f"SL case base '{self.name}': {e.message}") from e
            raise

    def decide(self, Z, round_num: int) -> dict:
        return self.client.decide(self.name, Z, round_num)
