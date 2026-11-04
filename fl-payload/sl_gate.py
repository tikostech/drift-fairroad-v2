"""
SL-trace case base and gate for FL checkpoints and client updates.

Every model the server sees -- a client's submitted weights or the averaged checkpoint -- is
loaded into a scratch CAE and run over a fixed probe set of images. ``CAESLProbe`` turns each
probe image's activations into one SL-minimised trace. A case is one probe image's trace from
one model, tagged with the model, its FL round and the probe image index.

The case base is one-class: it is built from valid models only. That follows the one-class
parts of tikos.profiler.cyber.training (its HybridCBR is supervised and needs labelled threats,
which FL does not have):

- Space: z-score fitted on the case base, constant columns dropped, optional PCA to ``k_pca``
  (hybrid_cbr.Space).
- Tolerance: per-feature 2.5/97.5 percentile bounds of the valid cases; coverage is the share
  of features inside them (cbr_reasoning.DomainTolerance). Features whose bounds collapse to a
  point are left out of coverage -- any deviation from a constant would count as a miss.
- Novelty: mean distance to the ``k`` nearest valid cases, with the cut placed on held-out valid
  data (the save_novelty_thresholds / tune_fpr novelty-gate pattern).

Two things are specific to FL checkpoints:

- Comparison is round-aware. A model from round r is compared only with valid cases from rounds
  r-window..r+window, and its tolerance bounds and thresholds come from those rounds. Valid
  models move a long way over training (the round-1 average ignores its input), so against all
  rounds a poisoned late model can sit next to a valid early one.
- Neighbours are matched per probe image: image i's trace is compared only with image i's
  traces in valid models. The probe images differ from each other far more than valid models
  do, so pooling them would hide the model-level signal.

A model's score is the median over probe images of that kNN distance. Calibration is
leave-one-group-out (group = run when the base has several runs, else model), so a valid model
is never scored against its own traces. Invalid = score above the round's distance threshold or
coverage below the round's coverage threshold.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from sl_probe_cae import CAESLProbe
from sc.slmin.activation_probe import check_schema, schema_hash

CASE_BASE_VERSION = 2


# ---------------------------------------------------------------------------
# Probe set and tracing
# ---------------------------------------------------------------------------

def load_probe_images(image_dir: str, n: int, img_size: int):
    """``n`` images spread evenly over the sorted directory, so every client's shard is covered.

    Returns ``(batch, file_names, probe_hash)``. The hash pins the probe set: a case base is only
    comparable with traces taken on the same images.
    """
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if len(files) < n:
        raise ValueError(f"{image_dir} has {len(files)} images, need {n} for the probe set")
    picks = [files[i] for i in np.linspace(0, len(files) - 1, n).round().astype(int)]
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    batch = torch.stack([tf(Image.open(os.path.join(image_dir, f)).convert('RGB')) for f in picks])
    return batch, picks, schema_hash([f"{img_size}"] + picks)


class ModelTracer:
    """Trace a state dict: load it into a scratch network and run the probe set through it."""

    def __init__(self, net: torch.nn.Module, probe_batch: torch.Tensor, probe_hash: str,
                 scope: str = "role", bands: int = 1):
        self.net = net.eval()
        self.batch = probe_batch.to(next(net.parameters()).device)
        self.probe_hash = probe_hash
        self.probe = CAESLProbe(self.net, scope=scope, bands=bands)
        self.names: list[str] = []
        self._lock = threading.Lock()  # one scratch network, traced from executor threads

    def trace(self, state_dict) -> np.ndarray:
        """``(n_probe, n_features)`` traces for this model, rows in probe-image order.

        On GPU, cuDNN's TF32 convolutions are switched off while tracing: with them, traces differ from
        CPU traces (which case bases may be built from) by ~5e-5 typically; without, ~1e-7.
        """
        tf32 = torch.backends.cudnn.allow_tf32
        try:
            with self._lock, torch.no_grad():
                torch.backends.cudnn.allow_tf32 = False
                self.net.load_state_dict(state_dict)
                self.probe.reset()
                with self.probe.capture():
                    self.net(self.batch)
                Z, self.names = self.probe.features()
        finally:
            torch.backends.cudnn.allow_tf32 = tf32
        return Z


# ---------------------------------------------------------------------------
# Case base
# ---------------------------------------------------------------------------

class SLCaseBase:
    """One-class, round-aware case base of valid models' SL traces, with a calibrated gate."""

    def __init__(self, k: int = 5, k_pca: int = 0, window: int = 1, q: float = 1.0,
                 margin: float = 1.25, lo_pct: float = 2.5, hi_pct: float = 97.5,
                 match_images: bool = True, calibration: str = "max", coverage_rule: bool = True):
        """``window`` >= the number of rounds and ``match_images=False`` give the round-agnostic,
        image-pooled comparison (kept for ablation).

        ``calibration`` sets how held-out valid scores become thresholds (see ``set_thresholds``):
        "max" takes the ``q`` quantile of the models within ``window`` rounds; "pooled" normalises
        every held-out score by its round's median, takes one ``q`` quantile over all rounds, and
        scales it back per round.

        ``coverage_rule`` makes low feature coverage a reason to reject on its own. Measured on the
        live runs it almost only added false positives (the distance rule caught nearly every attack
        it caught), so new case bases are built without it; coverage is still reported as evidence.
        """
        if calibration not in ("max", "pooled"):
            raise ValueError(f"calibration must be 'max' or 'pooled', not {calibration!r}")
        self.k, self.k_pca, self.window, self.q, self.margin = k, k_pca, window, q, margin
        self.match_images, self.calibration_method = match_images, calibration
        self.coverage_rule = coverage_rule
        self.lo_pct, self.hi_pct = lo_pct, hi_pct
        self.meta: dict = {}

    # -- space ---------------------------------------------------------------

    def _fit_space(self, Z):
        self.mu = Z.mean(0)
        sd = Z.std(0)
        self.keep = sd > 1e-9
        self.sd = np.where(self.keep, sd, 1.0)
        zs = self.z(Z)
        if self.k_pca:
            w, v = np.linalg.eigh(np.cov(zs, rowvar=False))
            self.comp = v[:, np.argsort(w)[::-1][:min(self.k_pca, zs.shape[1])]]
        else:
            self.comp = np.eye(zs.shape[1])
        return zs

    def z(self, Z):
        return ((Z - self.mu) / self.sd)[:, self.keep]

    # -- scoring -------------------------------------------------------------

    def _candidates(self, round_num: int, exclude=None):
        """Valid cases from rounds near ``round_num`` (clamped to the rounds the base covers)."""
        r = int(np.clip(round_num, self.rounds.min(), self.rounds.max()))
        mask = np.abs(self.case_round - r) <= self.window
        if exclude is not None:
            mask &= ~exclude
        return mask, r

    def _score(self, zq, round_num: int, exclude=None):
        """``(score, coverage, nearest model per image, lo, hi)`` of one model's standardised traces."""
        cand, _ = self._candidates(round_num, exclude)
        zc = self.Zs[cand]
        lo = np.percentile(zc, self.lo_pct, axis=0)
        hi = np.percentile(zc, self.hi_pct, axis=0)
        live = (hi - lo) > 1e-6
        coverage = float(((zq >= lo) & (zq <= hi))[:, live].mean())

        pq, pc = zq @ self.comp, zc @ self.comp
        img, models = self.case_image[cand], self.case_model[cand]
        dist, nearest = np.empty(len(zq)), []
        for i in range(len(zq)):
            same = img == i if self.match_images else np.ones(len(img), bool)
            d = np.linalg.norm(pc[same] - pq[i], axis=1)
            k = min(self.k, len(d))
            dist[i] = np.partition(d, k - 1)[:k].mean()
            nearest.append(models[same][d.argmin()])
        return float(np.median(dist)), coverage, nearest, lo, hi

    # -- fit / decide --------------------------------------------------------

    def fit(self, Z, model_ids, rounds, images, groups, names, meta=None):
        """Rows of ``Z`` are probe traces of valid models.

        ``model_ids``/``rounds``/``images`` tag each row; ``groups`` controls calibration -- every
        model in a group is scored against a base built without that group (pass run ids when
        there are several runs).
        """
        self.names = list(names)
        self.case_model = np.asarray(model_ids).astype(str)
        self.case_round = np.asarray(rounds, int)
        self.case_image = np.asarray(images, int)
        groups = np.asarray(groups).astype(str)
        self.rounds = np.unique(self.case_round)
        self.Zs = self._fit_space(Z)

        cal = []
        for m in np.unique(self.case_model):
            rows = self.case_model == m
            g, r = groups[rows][0], int(self.case_round[rows][0])
            s, c, _, _, _ = self._score(self.Zs[rows], r, exclude=groups == g)
            cal.append(dict(model=str(m), group=str(g), round=r, score=s, coverage=c))
        self.calibration = cal
        self.meta = dict(meta or {}, n_models=int(len(np.unique(self.case_model))), n_cases=int(len(Z)),
                         n_groups=int(len(np.unique(groups))), rounds=[int(r) for r in self.rounds])
        self.set_thresholds()
        return self

    def set_thresholds(self, calibration: str = None, q: float = None, margin: float = None):
        """(Re)derive per-round thresholds from the held-out valid scores in ``self.calibration``.

        "max": per round, the ``q`` quantile of held-out models within ``window`` rounds x ``margin``.
        With ``q=1`` that is the single most atypical valid model, so one outlier sets the threshold
        for 2*window+1 rounds, and the threshold grows as more runs are held out.

        "pooled": each held-out score is divided by the median held-out score of its round window,
        one ``q`` quantile is taken over all rounds together (all held-out models instead of ~1/3 of
        them), and the cut is scaled back by each round's median. Coverage is treated the same way
        with the ``1 - q`` quantile.
        """
        self.calibration_method = calibration or self.calibration_method
        self.q = self.q if q is None else q
        self.margin = self.margin if margin is None else margin
        cal = self.calibration
        near = {int(r): [c for c in cal if abs(c["round"] - r) <= self.window] for r in self.rounds}
        self.thresholds = {}
        if self.calibration_method == "max":
            for r, sel in near.items():
                s = np.array([c["score"] for c in sel])
                cv = np.array([c["coverage"] for c in sel])
                self.thresholds[r] = dict(score=float(np.quantile(s, self.q) * self.margin),
                                          coverage=float(np.quantile(cv, 1.0 - self.q) / self.margin),
                                          n_models=len(sel))
            return self
        s_scale = {r: float(np.median([c["score"] for c in sel])) for r, sel in near.items()}
        c_scale = {r: float(np.median([c["coverage"] for c in sel])) for r, sel in near.items()}
        zs = np.array([c["score"] / s_scale[c["round"]] for c in cal])
        zc = np.array([c["coverage"] / c_scale[c["round"]] for c in cal])
        s_cut = float(np.quantile(zs, self.q) * self.margin)
        c_cut = float(np.quantile(zc, 1.0 - self.q) / self.margin)
        for r, sel in near.items():
            self.thresholds[r] = dict(score=s_cut * s_scale[r], coverage=c_cut * c_scale[r], n_models=len(cal))
        return self

    def decide(self, Zq: np.ndarray, round_num: int) -> dict:
        """Gate one model's probe traces (rows in probe-image order). Returns verdict and evidence."""
        zq = self.z(Zq)
        s, c, nearest, lo, hi = self._score(zq, round_num)
        _, r = self._candidates(round_num)
        thr = self.thresholds[r]
        too_far, low_cov = s > thr["score"], self.coverage_rule and c < thr["coverage"]

        kept = [n for n, keep in zip(self.names, self.keep) if keep]
        zm = zq.mean(0)
        excess = np.maximum(lo - zm, 0.0) + np.maximum(zm - hi, 0.0)
        top = np.argsort(-excess)[:5]
        reasons = (["trace distance above threshold"] if too_far else []) + \
                  (["feature coverage below threshold"] if low_cov else [])
        return {
            "verdict": "invalid" if (too_far or low_cov) else "valid",
            "round": int(round_num),
            "compared_rounds": [int(x) for x in self.rounds if abs(x - r) <= self.window],
            "score": s,
            "score_threshold": thr["score"],
            "score_ratio": s / thr["score"],
            "coverage": c,
            "coverage_threshold": thr["coverage"],
            "coverage_rule": self.coverage_rule,
            "reasons": reasons,
            "nearest_valid": [{"model": str(m), "probe_images": int(n)}
                              for m, n in Counter(nearest).most_common(3)],
            "out_of_tolerance": [{"feature": kept[i], "z": float(zm[i]),
                                  "bounds": [float(lo[i]), float(hi[i])]}
                                 for i in top if excess[i] > 0],
        }

    # -- persistence ---------------------------------------------------------

    def check_compatible(self, names, probe_hash: str) -> None:
        """Refuse traces taken with a different feature schema or probe set."""
        check_schema(self.names, list(names), "SL case base")
        if probe_hash != self.meta.get("probe_hash"):
            raise ValueError(f"Probe set differs from the case base's ({probe_hash} vs {self.meta.get('probe_hash')})")

    _ARRAYS = ("mu", "sd", "keep", "comp", "Zs", "case_model", "case_round", "case_image", "rounds")

    def save(self, path: str) -> None:
        params = dict(k=self.k, k_pca=self.k_pca, window=self.window, q=self.q, margin=self.margin,
                      lo_pct=self.lo_pct, hi_pct=self.hi_pct, match_images=self.match_images,
                      calibration=self.calibration_method, coverage_rule=self.coverage_rule)
        header = dict(version=CASE_BASE_VERSION, params=params, meta=self.meta,
                      schema=schema_hash(self.names), thresholds=self.thresholds,
                      calibration=self.calibration)
        np.savez(path, header=json.dumps(header), names=np.array(self.names),
                 **{a: getattr(self, a) for a in self._ARRAYS})

    @classmethod
    def load(cls, path: str) -> "SLCaseBase":
        d = np.load(path, allow_pickle=False)
        header = json.loads(str(d["header"]))
        if header["version"] != CASE_BASE_VERSION:
            raise ValueError(f"Case base version {header['version']}, expected {CASE_BASE_VERSION}; rebuild it")
        cb = cls(**header["params"])
        cb.names = [str(n) for n in d["names"]]
        if schema_hash(cb.names) != header["schema"]:
            raise ValueError("Case base names do not match their recorded schema hash")
        for a in cls._ARRAYS:
            setattr(cb, a, d[a])
        cb.meta = header["meta"]
        cb.thresholds = {int(r): t for r, t in header["thresholds"].items()}
        cb.calibration = header["calibration"]
        return cb


# ---------------------------------------------------------------------------
# Trace archive written by the server
# ---------------------------------------------------------------------------

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
