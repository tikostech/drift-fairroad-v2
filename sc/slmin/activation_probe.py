"""Capture per-layer / per-head activation statistics during a forward pass.

Why this exists
---------------
``Wv`` is one scalar per token, and the ``Γa`` matrix that accompanies it is
rank-1 with a direction that is constant across prompts -- so the whole
SynapsesLogger *case* carries exactly as much information as ``Wv`` alone
(measured; see the README). Two unrelated feature families built on ``Wv`` both
plateau at ~0.55 balanced accuracy once prompt length is controlled. Getting more
signal therefore means capturing activations **before** they are reduced.

It does not, however, mean changing ``SynapsesLoggerV47``. Its ``__call__``
already retains every Linear layer's full input and output; the collapse to
``Wv`` happens later, in ``generateSynapsesLoggerBasedSC``. So this module adds a
*second, independent* hook rather than touching a component the production
service shares. Nothing here can alter ``Wv``, and ``main.py parity`` stays
meaningful.

Two measured constraints shape the design:

**Memory.** On Phi-4-mini, 129 Linear modules × 8 forward passes retained
121.8 MB for a single prompt; at 32 new tokens that is ~487 MB, and a 3,870-row
corpus of raw activations is out of the question. So this reduces each tensor to
a handful of numbers *inside* the hook and never stores the tensor.

**Prefill-only.** ``scripts/selftest_trace.py`` measured the trace as
prefill-only on this build -- 32 and 512 new tokens give the same ``Wv`` to ~1e-6.
Only the first forward pass carries prompt information, so this captures that
pass and ignores every decode step, which cuts the work by the number of
generated tokens.

What comes out
--------------
Per prompt: ``(n_hooked_layers × n_stats)`` floats, length-invariant because the
token axis is aggregated away. With the default statistics that is ~774 numbers
per prompt versus the 4 the current pipeline uses -- about 6 KB, so a 3,870-row
corpus costs ~24 MB.

Per-head statistics are available for fused attention projections by slicing the
output into heads; pass ``head_dim`` to enable it.
"""

from __future__ import annotations

import contextlib
import hashlib

import numpy as np


def _nullcontext():
    """Used when a caller passes a client without a lock (tests, scripts)."""
    return contextlib.nullcontext()


def schema_hash(names) -> str:
    """Fingerprint of the feature list, order included.

    The probe's feature order is whatever order the forward hooks fire in, which
    is a property of the model graph and the library that walks it -- not
    something this code controls. A different transformers release or model
    revision can reorder modules, and then feature *i* means something else while
    every array still has the right shape. Nothing would raise; a PCA basis fitted
    on the old order would simply produce quiet nonsense.

    So the order is pinned to a hash that travels with every artifact, and is
    compared before features fitted under one order are used under another.
    """
    h = hashlib.sha256()
    for n in names:
        # A separator that cannot appear in a module path keeps ["ab","c"] and
        # ["a","bc"] from hashing alike.
        h.update(str(n).encode("utf-8") + b"\x1f")
    return h.hexdigest()[:16]


class SchemaMismatch(RuntimeError):
    """Runtime probe features do not match the ones an artifact was fitted on."""


def check_schema(expected_names, actual_names, what: str = "artifact") -> None:
    """Raise unless the two feature lists are identical, order included."""
    if list(expected_names) == list(actual_names):
        return
    exp, act = list(expected_names), list(actual_names)
    detail = f"{len(exp)} expected vs {len(act)} produced"
    if len(exp) == len(act):
        i = next((j for j, (a, b) in enumerate(zip(exp, act)) if a != b), None)
        detail = f"same length, first difference at index {i}: {exp[i]!r} vs {act[i]!r}"
    raise SchemaMismatch(
        f"activation feature schema does not match {what}: {detail}. "
        f"The probe's feature order follows the model graph, so a different "
        f"model revision or transformers version reorders it. Re-profile the "
        f"corpus and refit, or restore the original environment -- the stored "
        f"PCA basis is not valid under a different order.")

# Aggregations applied across tokens, after the per-token reduction. Each keeps
# the feature independent of how many tokens the prompt had.
TOKEN_AGGS = ("mean", "std", "max", "p90_over_mean")


# Stat names in the fixed order _layer_stats emits them. The order is part of
# the contract: values stay as an unnamed device tensor until the very end, so
# names are reattached positionally.
BASE_STATS = ("mean", "std", "max", "p90_over_mean")
EXTRA_STATS = ("sparsity", "kurtosis", "max_over_rms")
HEAD_STATS = ("head_entropy", "head_max_share", "n_heads")


def _layer_stats(t, extra: bool = True, per_head: bool = False, head_dim: int = 128):
    """One layer's statistics as a **device tensor**, never a Python float.

    Two things dominate the cost of profiling a corpus, and both are addressed
    here rather than in the caller.

    *Passes over the data.* Everything is derived from a single elementwise
    square: ``x2 = t*t`` gives per-token squared norms by summing the feature
    axis, its mean is the mean square, and the mean of ``x2*x2`` is the fourth
    moment, so kurtosis about zero comes free. Sparsity tests a squared
    threshold rather than taking ``abs``.

    *Device synchronisation.* Calling ``float()`` on a CUDA tensor blocks until
    the GPU catches up. With 129 hooked layers and ~7 statistics each that is
    ~900 stalls per prompt, which measured at 2.5 prompt/s on a Blackwell card --
    the syncs, not the arithmetic. So every value stays on the device and is
    concatenated; the single transfer happens once per prompt in
    :meth:`ActivationProbe.features`.
    """
    import torch

    if t.dim() == 1:
        t = t.unsqueeze(0)
    t = t.to(torch.float32)
    x2 = t * t                                   # the one expensive pass
    norms = torch.sqrt(x2.sum(dim=1))            # per-token L2, what Wv approximates
    mu = norms.mean()
    one = torch.ones((), device=t.device, dtype=torch.float32)
    vals = [
        mu,
        norms.std(unbiased=False) if norms.numel() > 1 else one * 0,
        norms.max(),
        (torch.quantile(norms, 0.9) / (mu + 1e-12)) if norms.numel() > 1 else one,
    ]
    if extra:
        ms = x2.mean()                           # mean square
        m4 = (x2 * x2).mean()                    # fourth moment, reusing x2
        mx2 = x2.max()
        vals += [
            # fraction of units effectively off -- a sparsity signature
            (x2 < 1e-4 * mx2).to(torch.float32).mean(),
            # heavy-tailedness about zero: a few units dominating shows up here
            m4 / (ms * ms + 1e-24),
            torch.sqrt(mx2) / (torch.sqrt(ms) + 1e-12),
        ]
    if per_head:
        if t.shape[1] % head_dim == 0:
            h = t.reshape(t.shape[0], -1, head_dim)
            per = torch.linalg.norm(h, dim=2).mean(dim=0)
            p = per / (per.sum() + 1e-12)
            n_h = p.numel()
            vals += [
                -(p * torch.log(p + 1e-12)).sum() / float(np.log(max(n_h, 2))),
                p.max(),
                one * float(n_h),
            ]
        else:
            # Emit placeholders so every layer contributes the same number of
            # values -- a ragged feature vector cannot be stacked into a matrix.
            vals += [one, one, one * 0]
    return torch.stack([v.to(torch.float32).reshape(()) for v in vals])


class ActivationProbe:
    """A forward hook that reduces each Linear output to a few numbers.

    Attach with :meth:`attach`, run one generation, then read :meth:`features`.
    The probe records only the **first** forward pass per layer -- the prefill --
    and ignores decode steps, so the result depends on the prompt alone.
    """

    def __init__(self, per_head: bool = False, head_dim: int = 128,
                 extra: bool = True, max_layers: int = 0):
        self.per_head = per_head
        self.head_dim = head_dim
        self.extra = extra
        self.max_layers = max_layers
        self._rows: dict[str, dict] = {}
        self._seen: set[str] = set()
        self._handles: list = []
        self._order: list[str] = []

    # ------------------------------------------------------------------ #
    def _make_hook(self, name: str):
        def hook(module, module_in, module_out):
            # Prefill only: the first call per layer is the prompt pass, every
            # later call is a single decode step and carries no prompt info.
            if name in self._seen:
                return
            self._seen.add(name)

            out = module_out[0] if isinstance(module_out, tuple) else module_out
            if out is None or not hasattr(out, "detach"):
                return
            t = out.detach()
            if t.dim() == 3:              # (batch, tokens, d) -> first sequence
                t = t[0]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            # Stays on the device; the one transfer happens in features().
            self._rows[name] = _layer_stats(t, extra=self.extra,
                                            per_head=self.per_head,
                                            head_dim=self.head_dim)
            self._order.append(name)
        return hook

    def attach(self, model) -> int:
        """Hook every ``torch.nn.Linear``; returns how many were hooked."""
        import torch

        self.reset()
        n = 0
        for name, layer in model.named_modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            if self.max_layers and n >= self.max_layers:
                break
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))
            n += 1
        return n

    def detach(self) -> None:
        """Always call this, including on failure -- hooks left on a shared
        model accumulate across calls and corrupt later captures."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self._rows.clear()
        self._seen.clear()
        self._order.clear()

    # ------------------------------------------------------------------ #
    def features(self) -> tuple[np.ndarray, list[str]]:
        """Flatten to ``(vector, names)`` -- the single device transfer per prompt."""
        import torch

        if not self._order:
            return np.zeros(0), []
        stat_names = list(BASE_STATS)
        if self.extra:
            stat_names += list(EXTRA_STATS)
        if self.per_head:
            stat_names += list(HEAD_STATS)
        vec = torch.cat([self._rows[layer] for layer in self._order]).to("cpu").numpy()
        vec = np.nan_to_num(vec.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        names = [f"{layer}.{st}" for layer in self._order for st in stat_names]
        return vec, names

    @property
    def n_layers(self) -> int:
        return len(self._rows)


def _f32():  # retained for callers; stats now run in torch
    import torch
    return torch.float32


def _batch_layer_stats(t, mask, extra: bool, per_head: bool, head_dim: int):
    """Per-sequence statistics for a padded batch: ``(B, T, D) -> (B, n_stats)``.

    Every reduction is masked. Padding tokens are real positions in the tensor
    but carry no prompt, so including them would make a statistic depend on the
    longest prompt in the batch -- i.e. reintroduce a length artifact through the
    back door, and a batch-composition artifact on top of it.
    """
    import torch

    t = t.to(torch.float32)
    m2 = mask.to(torch.float32)                       # (B, T)
    n_tok = m2.sum(dim=1).clamp(min=1.0)              # real tokens per sequence
    D = t.shape[-1]

    # Memory discipline: reduce over the feature axis first, then mask the (B, T)
    # result. Masking the full (B, T, D) tensor instead -- which is what the
    # obvious formulation does -- allocates several more copies of a tensor that
    # reaches 32 x 1024 x 16384 floats here, and that is what exhausted a 97 GB
    # card on the longest prompts.
    x2 = t * t                                        # the one full-size temporary
    sq_sum = x2.sum(dim=2)                            # (B, T)
    quad_sum = (x2 * x2).sum(dim=2)                   # (B, T)
    norms = torch.sqrt(sq_sum) * m2                   # zero on padding
    mu = norms.sum(dim=1) / n_tok
    var = ((norms - mu.unsqueeze(1)) ** 2 * m2).sum(dim=1) / n_tok
    mx = (norms + (m2 - 1.0) * 1e30).max(dim=1).values             # padding -> -inf
    # A masked quantile has no vectorised form, so approximate p90 by the mean
    # plus 1.2822 sd (the normal 90th percentile). The feature is a spread ratio;
    # it does not need to be an exact order statistic.
    p90 = mu + 1.2822 * torch.sqrt(var.clamp(min=0))
    vals = [mu, torch.sqrt(var.clamp(min=0)), mx, p90 / (mu + 1e-12)]

    if extra:
        denom = n_tok * D
        ms = (sq_sum * m2).sum(dim=1) / denom
        m4 = (quad_sum * m2).sum(dim=1) / denom
        mx2 = (x2.amax(dim=2) * m2).amax(dim=1)
        small = ((x2 < 1e-4 * mx2.view(-1, 1, 1)).sum(dim=2).to(torch.float32)
                 * m2).sum(dim=1) / denom
        vals += [small, m4 / (ms * ms + 1e-24), torch.sqrt(mx2) / (torch.sqrt(ms) + 1e-12)]

    if per_head:
        if D % head_dim == 0:
            h = t.reshape(t.shape[0], t.shape[1], -1, head_dim)
            hn = torch.linalg.norm(h, dim=3)                       # (B, T, heads)
            per = (hn * m2.unsqueeze(-1)).sum(dim=1) / n_tok.unsqueeze(1)   # (B, heads)
            p = per / (per.sum(dim=1, keepdim=True) + 1e-12)
            n_h = p.shape[1]
            vals += [-(p * torch.log(p + 1e-12)).sum(dim=1) / float(np.log(max(n_h, 2))),
                     p.max(dim=1).values,
                     torch.full_like(mu, float(n_h))]
        else:
            ones = torch.ones_like(mu)
            vals += [ones, ones, torch.zeros_like(mu)]
    return torch.stack(vals, dim=1)                   # (B, n_stats)


class BatchActivationProbe:
    """Batched counterpart of :class:`ActivationProbe`.

    At batch size 1 the GPU is launch-latency bound, not compute bound: a bare
    Phi-4-mini prefill measured 7.5 prompt/s on a Blackwell card, and the probe's
    129 layers of small kernels took that to 2.5. Batching does the same number
    of launches for many more prompts, which is where the speedup comes from.
    """

    def __init__(self, per_head: bool = True, head_dim: int = 128, extra: bool = True):
        self.per_head, self.head_dim, self.extra = per_head, head_dim, extra
        self._rows: dict[str, object] = {}
        self._order: list[str] = []
        self._seen: set[str] = set()
        self._handles: list = []
        self._mask = None

    def _make_hook(self, name: str):
        def hook(module, module_in, module_out):
            if name in self._seen or self._mask is None:
                return
            out = module_out[0] if isinstance(module_out, tuple) else module_out
            if out is None or not hasattr(out, "detach") or out.dim() != 3:
                return
            if out.shape[1] != self._mask.shape[1]:
                return                     # not a per-token tensor; skip
            self._seen.add(name)
            self._rows[name] = _batch_layer_stats(out.detach(), self._mask,
                                                  self.extra, self.per_head, self.head_dim)
            self._order.append(name)
        return hook

    def attach(self, model) -> int:
        import torch

        self.detach()
        n = 0
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                self._handles.append(layer.register_forward_hook(self._make_hook(name)))
                n += 1
        return n

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def begin(self, mask) -> None:
        """Start a batch: clears per-batch state and installs the padding mask."""
        self._rows.clear()
        self._order.clear()
        self._seen.clear()
        self._mask = mask

    def features(self) -> tuple[np.ndarray, list[str]]:
        import torch

        if not self._order:
            return np.zeros((0, 0)), []
        stat_names = list(BASE_STATS)
        if self.extra:
            stat_names += list(EXTRA_STATS)
        if self.per_head:
            stat_names += list(HEAD_STATS)
        M = torch.cat([self._rows[l] for l in self._order], dim=1).to("cpu").numpy()
        M = np.nan_to_num(M.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        names = [f"{l}.{s}" for l in self._order for s in stat_names]
        return M, names


def probe_batch(client, prompts: list[str], probe: "BatchActivationProbe" = None,
                per_head: bool = True, head_dim: int = 128):
    """Profile many prompts in one forward pass. Returns ``(B, d)`` and names.

    Right-padded with an attention mask; every statistic is masked, so a
    sequence's features do not depend on what else shared its batch.
    """
    import torch

    # Right padding with an explicit mask: real tokens then sit at positions
    # 0..n-1 and attend normally, which is the safe arrangement for a
    # non-generation forward pass.
    if client.tokenizer.pad_token is None:
        client.tokenizer.pad_token = client.tokenizer.eos_token
    client.tokenizer.padding_side = "right"
    texts = [client.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts]
    enc = client.tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048,
                           add_special_tokens=False).to(client.model.device)

    # Hold the client's lock across attach -> forward -> read -> detach. The
    # hooks live on the shared model, so an overlapping call would swap hooks
    # under this one and hand back another request's activations. Tokenisation
    # is deliberately outside the lock: it touches only the tokenizer and is the
    # cheap part.
    lock = getattr(client, "model_lock", None)
    with (lock if lock is not None else _nullcontext()):
        own = probe is None
        if own:
            probe = BatchActivationProbe(per_head=per_head, head_dim=head_dim)
            probe.attach(client.model)
        probe.begin(enc["attention_mask"])
        try:
            with torch.no_grad():
                client.model(**enc)
            return probe.features()
        finally:
            if own:
                probe.detach()


def probe_prompt(client, prompt: str, per_head: bool = True,
                 head_dim: int = 128) -> tuple[np.ndarray, list[str]]:
    """Run one prompt through ``client``'s model and return its activation features.

    ``client`` is an :class:`util.llm_client.LLMExtractClient`. The SynapsesLogger
    is deliberately not involved: this is an independent capture path, so it
    cannot perturb ``Wv``.
    """
    import torch

    probe = ActivationProbe(per_head=per_head, head_dim=head_dim)
    inputs = client.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to(client.model.device)
    probe.attach(client.model)
    try:
        with torch.no_grad():
            # One forward pass is enough: the probe only records prefill, so
            # there is no reason to pay for generation at all.
            client.model(**inputs)
    finally:
        probe.detach()
    return probe.features()
