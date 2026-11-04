"""SynapsesLogger-style minimisation of the activation probe's layer axis.

What this is for
----------------
``util.activation_probe`` emits ``129 modules x 10 statistics = 1290`` numbers
per prompt, and the pipeline then fits a PCA to get that down to something a
Euclidean SOM can organise. The thesis's SynapsesLogger does its own reduction,
and it does it *structurally* rather than by fitting anything: A 5.2 averages a
synapses node group over the postsynaptic neurones, A 5.4 averages the result
over every layer to one causation-probability scalar per superficial feature,
and A 5.5 keeps, per layer, only the single strongest neurone -- a sparse path
``C = [(lambda, tau, r)]`` -- from which the posteriors are derived.

Mapping the thesis onto this probe
----------------------------------
The three axes line up once you notice the probe has already done A 5.2:

| thesis                          | probe                                      |
| ------------------------------- | ------------------------------------------ |
| superficial feature ``iota``    | the statistic channel (mean, std, ...)      |
| node group ``Gamma(lambda,tau)``| one Linear module                           |
| postsynaptic neurone ``tau``    | *already reduced* -- ``_layer_stats`` collapses the ``d_model`` axis to a per-token norm, then collapses tokens |
| layer ``lambda``                | depth in the stack, 0..31                   |

So the neurone axis is gone before this module sees anything, and A 5.2 is a
no-op here. The axis that is *not* reduced -- and which is therefore the entire
source of the 1290 -- is depth. That is what A 5.4 and A 5.5 reduce, and that is
what this module does.

Depth is reduced **within a role family**, not across all 129 modules at once.
``qkv_proj``, ``o_proj``, ``gate_up_proj`` and ``down_proj`` are four distinct
postsynaptic populations with different widths and different jobs; the thesis
groups per postsynaptic neurone layer, and averaging a 16384-wide MLP output
together with a 3072-wide projection would be the same mistake the PCA docstring
in ``build_activation_features`` warns about -- it would rank the groups by
width. ``scope="net"`` pools all 129 anyway, as the aggressive variant, so the
choice is measured rather than assumed.

What is emitted, and why these six
----------------------------------
Per (group, statistic), the depth series ``v[0..D-1]`` becomes six numbers. Two
of them are A 5.4, four are A 5.5's path ``C`` collapsed to scalars:

``phi``     mean over depth of ``log10 v`` -- A 5.4, the causation probability
            abstraction. In logs because the raw statistics span orders of
            magnitude across depth, so a linear mean is a report on whichever
            layer happens to be largest.
``spread``  standard deviation of the same. The thesis asks for exactly this:
            "the variation of Synapses node value of input feature per layer
            will be calculated as a density function and the variation of the
            density will be used to calculate the causality abstraction".
``peak``    ``max_d r[d]`` where ``r = v / sum(v)`` is the relevance share over
            depth -- A 5.5's highest successful relevance ``r'``.
``depth``   ``argmax_d r[d] / (D-1)`` -- A 5.5's path location. The full ``C``
            is a list of (layer, neurone, relevance); the neurone is already
            gone, so the path collapses to *where in the stack* this statistic
            peaks.
``drift``   ``sum_d r[d] * d / (D-1)`` -- the relevance-weighted centre of mass
            of the path. A smoother read of the same thing as ``depth``, which
            is a hard argmax and therefore jumpy.
``conc``    normalised entropy of ``r``, inverted: 1 when the path is localised
            in one layer, 0 when relevance is flat across the stack. This is the
            single number that says whether the feature *has* a path at all.

``r`` is a share and so is invariant to the overall scale of the statistic,
which matters here for the same reason it matters everywhere else in this
repository: prompt length moves magnitudes, and a feature that moves with it is
a length detector. ``phi`` and ``spread`` deliberately keep scale, so the two
halves of the block are complementary rather than redundant.

The thesis's S and F are not emitted, on purpose
------------------------------------------------
A 5.5 returns ``(I, A, S, F, C)`` with

    A = mean_over_C(r)
    S = mean_over_C(r * 1 / A)        = (sum r) / (A * |C|) = 1
    F = mean_over_C((1 - r) / A)      = (1 - A) / A

``S`` is identically 1 for every feature and every prompt, and ``F`` is a
deterministic function of ``A``, so neither is a feature -- carrying them would
add two columns of which one is constant and one is a reparameterisation of a
column already present. ``A`` itself is ``mean_d r[d]``, which is ``1/D`` once
``r`` is a share, so it is constant too. This is a property of the algorithm as
written, not of this implementation; ``peak`` and ``conc`` are what is left of
A 5.5 once the degenerate parts are removed.

Cost
----
``scope="role"`` gives ``4 groups x 10 stats x 6 + 10 (lm_head, ungrouped) =
250`` columns, a 5.2x reduction. ``scope="net"`` gives ``60``, a 21.5x
reduction -- small enough to feed a SOM with no PCA at all, which would remove
the fitted transform from the serving path.

One property worth having beyond the size: the reduction is invariant to the
order of modules *within* a role family. The activation probe's feature order
follows the model graph, which is why ``schema_hash`` and ``SchemaMismatch``
exist -- a different transformers release can reorder modules and silently
change what feature *i* means. Depth here is parsed from the module name, not
taken from hook-firing order, so a reordering cannot move a value into the wrong
column.

This is a pure function of the stored 1290 columns, so it needs no model and no
re-profiling: it runs over an existing ``*_profiles.npz``.

A 5.4 in the hook
-----------------
:class:`SLMinProbe` and :class:`SLMinBatchProbe` at the bottom of this module do
the same averaging *during* the forward pass, accumulating into a preallocated
cell instead of retaining a tensor per module. They are exact -- the equivalence
against :func:`minimize` is asserted in ``tests/test_sl_minimize.py`` and
re-checked on the real model by ``scripts/bench_sl_probe.py``.

They are also not faster, which is the measured answer to the obvious question.
The tensor the unreduced probe retains per module is ``(B, n_stats)`` -- ten
floats per prompt per layer -- so retaining and concatenating them is nothing
next to computing them, and A 5.4 needs them all computed before it can average.
The forward-pass cost is in how many modules are read and how much activation
each carries; ``--depth-stride`` in ``scripts/sl_minimize_profiles.py`` is the
knob that changes that, and unlike everything else here it is an approximation.
"""

from __future__ import annotations

import re

import numpy as np

# Emitted per (group, statistic), in this order. Positional, like every other
# feature contract in this package.
SL_STATS = ("phi", "spread", "peak", "depth", "drift", "conc")

# Keeps log10 finite where a statistic is exactly zero (``sparsity`` is, on
# layers where nothing falls under the threshold). Far below anything observed.
EPS = 1e-12

# ``model.layers.<depth>.<block>.<role>.<stat>`` -- the only shape the probe
# produces for a decoder stack. Anything that does not match (``lm_head.*``) has
# no depth and is carried through unreduced.
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_names(names):
    """Split probe feature names into ``(group, depth, stat)`` triples.

    ``group`` is the role family (``self_attn.qkv_proj``), ``depth`` is the
    integer layer index, ``stat`` is the trailing statistic. Modules with no
    ``model.layers.N`` prefix get ``depth = -1`` and are treated as ungrouped.
    """
    out = []
    for n in names:
        module, _, stat = str(n).rpartition(".")
        m = _LAYER_RE.match(module)
        if m:
            out.append((m.group(2), int(m.group(1)), stat))
        else:
            out.append((module, -1, stat))
    return out


def _reduce_series(V: np.ndarray) -> np.ndarray:
    """``(n, D)`` of one statistic across depth -> ``(n, 6)``.

    Vectorised over prompts; ``D`` is 32 here, so this is cheap.
    """
    D = V.shape[1]
    if D == 1:
        # Nothing to reduce over. phi is the value itself, the path is trivial.
        lg = np.log10(np.abs(V) + EPS)
        z = np.zeros_like(lg)
        return np.column_stack([lg, z, np.ones_like(z), z, z, np.ones_like(z)])

    lg = np.log10(np.abs(V) + EPS)
    phi = lg.mean(axis=1)                                   # A 5.4
    spread = lg.std(axis=1)

    # Relevance share over depth. Built from |v| rather than the log so that it
    # is a genuine share of the statistic and not a share of its exponent.
    A = np.abs(V)
    tot = A.sum(axis=1, keepdims=True)
    # A statistic that is zero at every depth has no path; a flat share is the
    # honest encoding of "no information about where", and keeps conc at 0.
    r = np.where(tot > 0, A / np.where(tot > 0, tot, 1.0), 1.0 / D)

    peak = r.max(axis=1)
    idx = r.argmax(axis=1).astype(np.float64)               # A 5.5 path location
    depth = idx / (D - 1)
    drift = (r * np.arange(D, dtype=np.float64)).sum(axis=1) / (D - 1)
    ent = -(r * np.log(r + EPS)).sum(axis=1) / np.log(D)
    conc = 1.0 - np.clip(ent, 0.0, 1.0)

    return np.column_stack([phi, spread, peak, depth, drift, conc])


def minimize(X: np.ndarray, names, scope: str = "role", prefix: str = "slmin.",
             stats=SL_STATS, bands: int = 1):
    """Reduce ``(n, 1290)`` probe features to the SL-minimised block.

    ``scope="role"`` reduces depth within each role family; ``scope="net"``
    pools every depth-bearing module into a single group. Returns
    ``(Z, out_names)``.

    ``stats`` selects which of :data:`SL_STATS` to emit, which is what makes the
    two halves of the thesis separable: ``("phi", "spread")`` is A 5.4 alone,
    ``("peak", "depth", "drift", "conc")`` is A 5.5's path alone. They are
    measured separately because they turn out not to behave the same way.

    ``bands`` splits each family's depth series into that many contiguous
    windows and reduces each separately. ``bands=1`` is A 5.4 as written --
    average over the whole stack. Larger values read "node group" at a finer
    grain than "the entire layer family", which is the reading the measurements
    prefer: ``bands=8, stats=("phi",)`` scores *above* the unreduced 1290-column
    block at a quarter of the width.

    Ungrouped modules (``lm_head``) have no depth axis to reduce, so their
    columns are carried through unchanged rather than dropped -- the output
    layer is one module, and averaging it into a 32-deep family would be
    inventing a series that does not exist.
    """
    if scope not in ("role", "net"):
        raise ValueError(f"scope must be 'role' or 'net', not {scope!r}")
    stats = tuple(stats)
    unknown = [s for s in stats if s not in SL_STATS]
    if unknown:
        raise ValueError(f"unknown SL statistics {unknown}; known: {list(SL_STATS)}")
    take = [SL_STATS.index(s) for s in stats]
    if bands < 1:
        raise ValueError(f"bands must be >= 1, not {bands}")

    X = np.asarray(X, dtype=np.float64)
    triples = parse_names(names)
    if len(triples) != X.shape[1]:
        raise ValueError(f"{len(triples)} names for {X.shape[1]} columns")

    # column index -> (group, depth, stat); build the (group, stat) -> [(depth,
    # col)] index once.
    series: dict[tuple[str, str], list[tuple[int, int]]] = {}
    passthrough: list[int] = []
    for col, (group, depth, stat) in enumerate(triples):
        if depth < 0:
            passthrough.append(col)
            continue
        key = ("net" if scope == "net" else group, stat)
        series.setdefault(key, []).append((depth, col))

    blocks, out_names = [], []
    for (group, stat) in sorted(series):
        entries = sorted(series[(group, stat)])
        cols = [c for _, c in entries]
        # Pooling across roles means several modules share a depth; keeping them
        # as separate positions in the series is right -- the reduction is over
        # the whole family, and A 5.4 averages over every node group in it.
        edges = np.linspace(0, len(cols), bands + 1).astype(int)
        for b in range(bands):
            lo, hi = int(edges[b]), int(edges[b + 1])
            if hi <= lo:
                continue                     # more bands than depths in this family
            blocks.append(_reduce_series(X[:, cols[lo:hi]])[:, take])
            tag = f"{group}.{stat}" if bands == 1 else f"{group}.{stat}.b{b}"
            out_names += [f"{prefix}{tag}.{s}" for s in stats]

    if passthrough:
        blocks.append(X[:, passthrough])
        out_names += [f"{prefix}{triples[c][0]}.{triples[c][2]}" for c in passthrough]

    Z = np.concatenate(blocks, axis=1) if blocks else np.zeros((X.shape[0], 0))
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z, out_names


# ---------------------------------------------------------------------------
# A 5.4 folded into the forward hook
# ---------------------------------------------------------------------------
#
# Everything above reduces a matrix that has already been built. That cuts
# storage and PCA width but not the forward pass, because the 1290 columns are
# still materialised first: ``ActivationProbe`` keeps one small tensor per
# module, appends it to a list, and concatenates 129 of them at the end.
#
# A 5.4 does not need any of that kept. It is a mean over a group, and a mean is
# a running sum -- so the hook can add each layer's contribution straight into a
# preallocated cell and let the layer's own tensor die. Nothing is retained per
# module, nothing is concatenated, and the transfer at the end is the reduced
# width rather than the full one.
#
# The per-layer statistics themselves are computed by the *same* functions the
# unreduced probe uses, imported rather than reimplemented. That is what makes
# the equivalence test in ``tests/test_sl_minimize.py`` meaningful: if these
# probes agree with ``minimize(...)`` to floating-point, the only thing that
# changed is when the averaging happens, not what is averaged.


def band_edges(n_depths: int, bands: int):
    """Depth -> band index, partitioned exactly as :func:`minimize` does.

    Shared rather than reimplemented on purpose: the in-hook probe has to land
    on the same partition as the post-hoc reduction or the two disagree by a
    layer at the band boundaries, which is the kind of difference that would
    show up as a small unexplained drift in the features rather than as an
    error.
    """
    edges = np.linspace(0, n_depths, bands + 1).astype(int)
    out = {}
    for b in range(bands):
        for d in range(int(edges[b]), int(edges[b + 1])):
            out[d] = b
    return out


class _SLCells:
    """The (group, band) -> row mapping, and the names that come out of it.

    Built once from the module names at attach time, which is what lets depth be
    parsed from the *name* rather than taken from hook-firing order -- the
    invariance that removes the ``SchemaMismatch`` class for this block.
    """

    def __init__(self, module_names, stat_names, bands: int, scope: str,
                 prefix: str = "slmin."):
        self.bands = bands
        self.stat_names = list(stat_names)

        # group -> sorted depths, in the order minimize() would see them.
        depths: dict[str, list[int]] = {}
        self.passthrough: list[str] = []
        self.of_module: dict[str, tuple[str, int] | None] = {}
        for name in module_names:
            m = _LAYER_RE.match(name)
            if not m:
                self.passthrough.append(name)
                self.of_module[name] = None
                continue
            group = "net" if scope == "net" else m.group(2)
            depths.setdefault(group, []).append(int(m.group(1)))

        self.groups = sorted(depths)
        self._band_of = {g: band_edges(len(sorted(set(depths[g]))), bands)
                         for g in self.groups}
        # Bands that actually contain a layer. ``minimize`` skips empty ones --
        # asking for 8 bands over 2 sampled depths leaves six with nothing in
        # them -- so emitting cells for them here would shift every name by a
        # band relative to the post-hoc path. Found by the depth-stride test.
        self._live = {}
        for g in self.groups:
            n_d = len(set(depths[g]))
            edges = np.linspace(0, n_d, bands + 1).astype(int)
            self._live[g] = [b for b in range(bands)
                             if int(edges[b]) < int(edges[b + 1])]
        # Position of a depth within its group's sorted depth list -- not the
        # raw layer index, because minimize() bands by position in the series.
        self._pos = {g: {d: i for i, d in enumerate(sorted(set(depths[g])))}
                     for g in self.groups}

        for name in module_names:
            m = _LAYER_RE.match(name)
            if not m:
                continue
            group = "net" if scope == "net" else m.group(2)
            pos = self._pos[group][int(m.group(1))]
            self.of_module[name] = (group, self._band_of[group][pos])

        self.cell_of = {(g, b): i for i, (g, b) in
                        enumerate((g, b) for g in self.groups
                                  for b in self._live[g])}
        self.n_cells = len(self.cell_of)

        # Emission order is minimize()'s: sorted (group, stat), bands inner,
        # passthrough last. Kept identical so an archive from either path is
        # interchangeable with one from the other.
        self.names = []
        for group in self.groups:
            for stat in sorted(self.stat_names):
                for b in self._live[group]:
                    tag = f"{group}.{stat}" if bands == 1 else f"{group}.{stat}.b{b}"
                    self.names.append(f"{prefix}{tag}.phi")
        for name in self.passthrough:
            for stat in self.stat_names:
                self.names.append(f"{prefix}{name}.{stat}")

    def gather(self, buf, count, raw):
        """``(n_cells, n_stats)`` sums -> the flat vector, in emission order."""
        cols = []
        for group in self.groups:
            for stat in sorted(self.stat_names):
                si = self.stat_names.index(stat)
                for b in self._live[group]:
                    ci = self.cell_of[(group, b)]
                    n = count[ci]
                    cols.append(buf[..., ci, si] / n if n else buf[..., ci, si])
        for name in self.passthrough:
            if name in raw:
                for si in range(len(self.stat_names)):
                    cols.append(raw[name][..., si])
            else:
                for _ in range(len(self.stat_names)):
                    cols.append(None)
        return cols


def _hooked(linears, stride: int):
    """Module names to hook, after depth subsampling.

    Ungrouped modules (``lm_head``) are always kept: there is no depth series to
    sample from, and dropping the output layer would silently change the block's
    width rather than its resolution.
    """
    if stride <= 1:
        return list(linears)
    out = []
    for name in linears:
        m = _LAYER_RE.match(name)
        if not m or int(m.group(1)) % stride == 0:
            out.append(name)
    return out


class SLMinProbe:
    """A 5.4 accumulated inside the forward hook -- one prompt at a time.

    Same contract as :class:`util.activation_probe.ActivationProbe`: attach,
    run one forward pass, read :meth:`features`. The difference is what happens
    between those steps. ``ActivationProbe`` keeps a tensor per module and
    concatenates 129 of them; this adds each module's contribution into a
    preallocated ``(n_cells, n_stats)`` accumulator and keeps nothing.

    The per-layer statistics come from ``activation_probe._layer_stats``,
    imported rather than reimplemented, so the only thing that differs from the
    post-hoc path is *when* the averaging happens.
    """

    def __init__(self, bands: int = 8, per_head: bool = True, head_dim: int = 128,
                 extra: bool = True, scope: str = "role", acc_dtype: str = "float64",
                 depth_stride: int = 1):
        self.bands, self.scope = bands, scope
        # Hook only every Nth layer. This is the one knob here that changes the
        # forward pass rather than what is done with its output -- and it is an
        # approximation: a band then averages the layers it sampled, not all of
        # them. Its accuracy cost is measured in the README.
        self.depth_stride = depth_stride
        self.per_head, self.head_dim, self.extra = per_head, head_dim, extra
        # The accumulator's precision. float64 reproduces the post-hoc reduction
        # bit for bit, which is what the equivalence test asserts; float32 is
        # there because consumer and workstation NVIDIA parts run fp64 at a
        # fraction of fp32 rate, so on those the exactness has a measurable
        # price. See scripts/bench_sl_probe.py.
        self.acc_dtype = acc_dtype
        self._handles: list = []
        self._cells = None
        self._buf = self._count = None
        self._raw: dict = {}
        self._seen: set[str] = set()

    def _stat_names(self):
        from util.activation_probe import BASE_STATS, EXTRA_STATS, HEAD_STATS
        names = list(BASE_STATS)
        if self.extra:
            names += list(EXTRA_STATS)
        if self.per_head:
            names += list(HEAD_STATS)
        return names

    def _make_hook(self, name: str):
        from util.activation_probe import _layer_stats

        def hook(module, module_in, module_out):
            # Prefill only, for the same reason the unreduced probe is: later
            # calls are single decode steps and carry no prompt information.
            if name in self._seen:
                return
            self._seen.add(name)
            out = module_out[0] if isinstance(module_out, tuple) else module_out
            if out is None or not hasattr(out, "detach"):
                return
            t = out.detach()
            if t.dim() == 3:
                t = t[0]
            if t.dim() == 1:
                t = t.unsqueeze(0)

            v = _layer_stats(t, extra=self.extra, per_head=self.per_head,
                             head_dim=self.head_dim)
            cell = self._cells.of_module.get(name)
            if cell is None:
                # Ungrouped (lm_head): no depth series, carried raw.
                self._raw[name] = v.to(self._buf.dtype)
                return
            ci = self._cells.cell_of[cell]
            # The running sum. ``v`` goes out of scope here and its memory is
            # reclaimed -- nothing per-module survives the hook.
            self._buf[ci] += (v.to(self._buf.dtype).abs() + EPS).log10()
            self._count[ci] += 1
        return hook

    def attach(self, model) -> int:
        import torch

        self.detach()
        linears = _hooked([n for n, m in model.named_modules()
                           if isinstance(m, torch.nn.Linear)], self.depth_stride)
        self._keep = set(linears)
        self._cells = _SLCells(linears, self._stat_names(), self.bands, self.scope)
        dev = next(model.parameters()).device
        n_stats = len(self._stat_names())
        self._buf = torch.zeros((self._cells.n_cells, n_stats),
                                dtype=getattr(torch, self.acc_dtype), device=dev)
        self._count = [0] * self._cells.n_cells
        self._raw, self._seen = {}, set()

        n = 0
        for name, layer in model.named_modules():
            # A module that is not hooked costs nothing: no statistics are
            # computed for it at all, which is where the saving comes from.
            if isinstance(layer, torch.nn.Linear) and name in self._keep:
                self._handles.append(layer.register_forward_hook(self._make_hook(name)))
                n += 1
        return n

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        if self._buf is not None:
            self._buf.zero_()
            self._count = [0] * self._cells.n_cells
        self._raw, self._seen = {}, set()

    def features(self):
        """``(vector, names)`` -- one device transfer, at the reduced width."""
        import torch

        if self._cells is None:
            return np.zeros(0), []
        cols = self._cells.gather(self._buf, self._count, self._raw)
        zero = torch.zeros((), dtype=self._buf.dtype, device=self._buf.device)
        vec = torch.stack([zero if c is None else c.reshape(()) for c in cols])
        out = vec.to("cpu").numpy().astype(np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), list(self._cells.names)


class SLMinBatchProbe(SLMinProbe):
    """Batched counterpart -- the one that matters for profiling a corpus.

    ``begin(mask)`` installs the padding mask, exactly as
    :class:`util.activation_probe.BatchActivationProbe` requires, and every
    statistic underneath is masked, so a prompt's features do not depend on what
    else shared its batch.
    """

    def __init__(self, bands: int = 8, per_head: bool = True, head_dim: int = 128,
                 extra: bool = True, scope: str = "role", acc_dtype: str = "float64",
                 depth_stride: int = 1):
        super().__init__(bands, per_head, head_dim, extra, scope, acc_dtype,
                         depth_stride)
        self._mask = None
        self._batch = 0

    def _make_hook(self, name: str):
        from util.activation_probe import _batch_layer_stats

        def hook(module, module_in, module_out):
            if name in self._seen or self._mask is None:
                return
            out = module_out[0] if isinstance(module_out, tuple) else module_out
            if out is None or not hasattr(out, "detach") or out.dim() != 3:
                return
            if out.shape[1] != self._mask.shape[1]:
                return                        # not a per-token tensor; skip
            self._seen.add(name)
            v = _batch_layer_stats(out.detach(), self._mask, self.extra,
                                   self.per_head, self.head_dim)    # (B, n_stats)
            cell = self._cells.of_module.get(name)
            if cell is None:
                self._raw[name] = v.to(self._buf.dtype)
                return
            ci = self._cells.cell_of[cell]
            self._buf[:, ci, :] += (v.to(self._buf.dtype).abs() + EPS).log10()
            self._count[ci] += 1
        return hook

    def attach(self, model) -> int:
        import torch

        self.detach()
        linears = _hooked([n for n, m in model.named_modules()
                           if isinstance(m, torch.nn.Linear)], self.depth_stride)
        self._keep = set(linears)
        self._cells = _SLCells(linears, self._stat_names(), self.bands, self.scope)
        self._dev = next(model.parameters()).device
        n = 0
        for name, layer in model.named_modules():
            # A module that is not hooked costs nothing: no statistics are
            # computed for it at all, which is where the saving comes from.
            if isinstance(layer, torch.nn.Linear) and name in self._keep:
                self._handles.append(layer.register_forward_hook(self._make_hook(name)))
                n += 1
        return n

    def begin(self, mask) -> None:
        import torch

        B = int(mask.shape[0])
        self._batch = B
        n_stats = len(self._stat_names())
        self._buf = torch.zeros((B, self._cells.n_cells, n_stats),
                                dtype=getattr(torch, self.acc_dtype), device=self._dev)
        self._count = [0] * self._cells.n_cells
        self._raw, self._seen = {}, set()
        self._mask = mask

    def features(self):
        import torch

        if self._cells is None or self._buf is None:
            return np.zeros((0, 0)), []
        cols = self._cells.gather(self._buf, self._count, self._raw)
        zero = torch.zeros((self._batch,), dtype=self._buf.dtype,
                           device=self._buf.device)
        M = torch.stack([zero if c is None else c for c in cols], dim=1)
        out = M.to("cpu").numpy().astype(np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), list(self._cells.names)
