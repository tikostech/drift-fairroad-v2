"""
SynapsesLogger-minimised activation features for the CAE.

Adapts the vendored Tikos ``sc/slmin/sl_minimize.py`` (written for a decoder-only LM)
to the convolutional autoencoder:

- Hooks every weight-bearing layer (Conv2d, ConvTranspose2d, Linear), not just Linear.
- Per image, each layer's output is read as tokens x features: spatial positions x
  channels for conv layers, a single token for Linear layers. The statistics are
  ``activation_probe._layer_stats`` (per_head off -- the CAE has no attention heads).
- Layers are named the way ``minimize()`` parses an LM stack: ``conv1..conv5`` under
  ``encoder`` become depths 0..4 of family ``encoder.conv``, likewise ``decoder.deconv``.
  Layers without a trailing index (``encoder.fc``, ``decoder.fc``) have no depth and are
  carried through unreduced, as ``lm_head`` is for the LM.
- ``minimize()`` then reduces depth within each family (``scope="role"``) or across
  all conv layers (``scope="net"``).

With the small CAE and default settings this gives 2 families x 7 stats x 6 SL stats
+ 2 fc layers x 7 stats = 98 columns per image.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

# The `sc` package lives at the repo root; fl-payload scripts run with cwd=fl-payload/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sc.slmin.activation_probe import BASE_STATS, EXTRA_STATS  # noqa: E402
from sc.slmin.sl_minimize import SL_STATS, minimize  # noqa: E402

TRACED_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)

# "encoder.conv3" -> ("encoder.conv", 3)
_INDEXED_RE = re.compile(r"^(.*?)(\d+)$")


def batch_layer_stats(tokens: torch.Tensor, extra: bool = True) -> torch.Tensor:
    """``_layer_stats`` for a whole batch at once: ``(B, T, D) -> (B, n_stats)``.

    Same definitions as the vendored per-image ``_layer_stats`` (per_head off), including the exact
    p90 order statistic -- the vendored ``_batch_layer_stats`` approximates p90 as mean + 1.28 sd,
    which would change the features and invalidate case bases built from earlier traces. Calling
    ``_layer_stats`` once per image cost ~0.65 s per traced model; this is ~10x faster.
    """
    t = tokens.to(torch.float32)
    T = t.shape[1]
    x2 = t * t
    norms = torch.sqrt(x2.sum(dim=2))                                   # (B, T) per-token L2
    mu = norms.mean(dim=1)
    one = torch.ones_like(mu)
    if T > 1:
        sd = norms.std(dim=1, unbiased=False)
        p90 = torch.quantile(norms, 0.9, dim=1) / (mu + 1e-12)
    else:                                                               # as _layer_stats: one token
        sd, p90 = mu * 0, one
    vals = [mu, sd, norms.max(dim=1).values, p90]
    if extra:
        flat = x2.flatten(1)                                            # (B, T*D)
        ms = flat.mean(dim=1)
        m4 = (flat * flat).mean(dim=1)
        mx2 = flat.max(dim=1).values
        vals += [(flat < 1e-4 * mx2.unsqueeze(1)).to(torch.float32).mean(dim=1),
                 m4 / (ms * ms + 1e-24),
                 torch.sqrt(mx2) / (torch.sqrt(ms) + 1e-12)]
    return torch.stack(vals, dim=1)


def probe_name(module_name: str) -> str:
    """Map a CAE module name onto the ``model.layers.<depth>.<group>`` form minimize() parses."""
    m = _INDEXED_RE.match(module_name)
    if not m:
        return module_name  # ungrouped: no depth axis
    return f"model.layers.{int(m.group(2)) - 1}.{m.group(1)}"


class CAESLProbe:
    """Collect per-image layer statistics during forward passes and reduce them with minimize()."""

    def __init__(self, net: nn.Module, scope: str = "role", bands: int = 1,
                 stats=SL_STATS, extra: bool = True):
        self.net = net
        self.scope, self.bands, self.stats, self.extra = scope, bands, tuple(stats), extra
        self.stat_names = list(BASE_STATS) + (list(EXTRA_STATS) if extra else [])
        self.layers = [(name, layer) for name, layer in net.named_modules()
                       if isinstance(layer, TRACED_TYPES)]
        self._rows: dict[str, list[torch.Tensor]] = {}
        self.reset()

    def reset(self) -> None:
        self._rows = {name: [] for name, _ in self.layers}

    def _make_hook(self, name: str):
        def hook(module, module_in, module_out):
            out = module_out.detach()
            if out.dim() == 4:
                tokens = out.flatten(2).transpose(1, 2)   # (B, H*W, C)
            else:
                tokens = out.unsqueeze(1)                 # (B, 1, F)
            self._rows[name].append(batch_layer_stats(tokens, extra=self.extra))
        return hook

    @contextlib.contextmanager
    def capture(self):
        """Hooks are registered only inside this block, so normal forward passes are untouched."""
        handles = [layer.register_forward_hook(self._make_hook(name)) for name, layer in self.layers]
        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    def raw(self):
        """``(X, names)``: (n_images, n_layers * n_stats) before reduction."""
        counts = {name: sum(len(r) for r in rows) for name, rows in self._rows.items()}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"Layers captured different numbers of images: {counts}")
        if not next(iter(counts.values()), 0):
            return np.zeros((0, 0)), []
        blocks, names = [], []
        for name, _ in self.layers:
            blocks.append(torch.cat(self._rows[name]).cpu().numpy().astype(np.float64))
            names += [f"{probe_name(name)}.{s}" for s in self.stat_names]
        return np.concatenate(blocks, axis=1), names

    def features(self):
        """``(Z, names)``: the SL-minimised block, one row per image."""
        X, names = self.raw()
        if not names:
            return X, []
        return minimize(X, names, scope=self.scope, stats=self.stats, bands=self.bands)
