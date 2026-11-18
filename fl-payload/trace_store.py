"""
Per-run archive of every model the FL server traces (``sl_traces.npz``): writing it (``TraceStore``)
and reading it back with ground-truth labels (``load_run``).

Carries no IP -- the SL probe and the case base live in drift-fairroad-service -- so the server needs it
whether it traces in-process or through the SL web service.
"""
import json
import os
import threading

import numpy as np


class TraceStore:
    """Per-run archive of every traced model: rows of traces plus one record per model.

    Rewritten after each model so a crashed run still leaves its traces behind.
    """

    def __init__(self, path: str, probe_files, probe_hash: str):
        self.path = path
        self.probe_files, self.probe_hash = list(probe_files), probe_hash
        self.Z: list[np.ndarray] = []
        self.models: list[dict] = []
        self.names: list[str] = []
        self._lock = threading.Lock()

    def add(self, Z: np.ndarray, names, record: dict) -> None:
        with self._lock:
            self.names = list(names)
            self.Z.append(Z)
            self.models.append(record)
            np.savez(
                self.path,
                Z=np.concatenate(self.Z),
                row_model=np.concatenate([[m["model"]] * len(z) for m, z in zip(self.models, self.Z)]),
                row_image=np.concatenate([np.arange(len(z)) for z in self.Z]),
                names=np.array(self.names),
                models=json.dumps(self.models),
                probe_files=np.array(self.probe_files),
                probe_hash=self.probe_hash,
            )


def load_run(run_dir: str):
    """A run directory's traces, with each model record labelled.

    Ground truth comes from run_meta.json (written by run_fl_cae_system.py). ``label`` is:
      spoofed     the malicious client's update, or a checkpoint that averaged one in
      downstream  honest, but descended from a poisoned global model that was accepted (every
                  later model trains from it); neither valid nor spoofed, reported separately
      valid       everything else
    ``spoofed`` is True only for the first.
    """
    d = np.load(os.path.join(run_dir, "sl_traces.npz"), allow_pickle=False)
    models = json.loads(str(d["models"]))
    meta_path = os.path.join(run_dir, "run_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    bad = meta.get("malicious_client") if meta.get("attack") else None
    contaminated = False  # is the global model in force descended from a poisoned one?
    for m in sorted(models, key=lambda m: (m["round"], m["source"] == "global")):
        is_global = m["source"] == "global"
        spoofed = bool(bad) and (m["source"] == bad or (is_global and bad in (m.get("included") or [])))
        m["label"] = "spoofed" if spoofed else ("downstream" if contaminated else "valid")
        m["spoofed"] = spoofed
        m["kind"] = "checkpoint" if is_global else "client_update"
        if is_global and m.get("accepted", True) and spoofed:
            contaminated = True
    row_model = d["row_model"].astype(str)
    if "row_image" in d.files:
        row_image = d["row_image"]
    else:  # archives written before row_image was recorded: each model's rows are in probe order
        n_probe = len(d["probe_files"])
        row_image = np.tile(np.arange(n_probe), len(row_model) // n_probe)
    return dict(Z=d["Z"], row_model=row_model, row_image=row_image,
                names=[str(n) for n in d["names"]], models=models, probe_hash=str(d["probe_hash"]),
                meta=meta)
