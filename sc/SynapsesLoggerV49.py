# (c) Dr Don SMA Liyanage
# DO NOT REMOVE THE COPYRIGHT STATEMENT
import math
import torch
import torch.nn as nn
import numpy as np
from sc.SequentialCollection import SequentialCollection

# from sl.tutils import multiply as tutils_multiply
# from sl.tutils import divide as tutils_divide

TIKOS_AVAILABLE = True
# # Attempt to import Tikos Utils (handle import error if library is missing)
# try:
#     # Adjust this import if your project structure requires 'tikos_utils_os.tutils'
#     import tutils
#
#     TIKOS_AVAILABLE = True
# except ImportError:
#     tutils = None
#     TIKOS_AVAILABLE = False

# CUDA statement
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA Enabled: {CUDA_AVAILABLE}")
print(f"Tikos Utils Available: {TIKOS_AVAILABLE}")


class TikosLibAdapter:
    """
    Adapter for Mode 2 (Hybrid Streaming).
    - Uses tutils for heavy element-wise ops (GPU Streaming).
    - Uses NumPy for everything else (CPU).
    """

    def __init__(self):
        self.np = np

    def multiply(self, a, b):
        # Offload to GPU via Tikos Utils
        # Ensure inputs are numpy (safety check)
        a_np = a if isinstance(a, np.ndarray) else a.numpy()
        b_np = b if isinstance(b, np.ndarray) else b.numpy()
        # return tutils_multiply(a_np, b_np)
        return np.multiply(a_np, b_np)

    def divide(self, a, b):
        # Offload to GPU via Tikos Utils
        a_np = a if isinstance(a, np.ndarray) else a.numpy()
        b_np = b if isinstance(b, np.ndarray) else b.numpy()
        # return tutils_divide(a_np, b_np)
        return np.divide(a_np, b_np)

    # --- Fallbacks to NumPy (CPU) ---
    # Since tutils returns NumPy arrays, we must use NumPy for the rest of the chain.
    def sum(self, x, dim=None, axis=None, keepdim=False, **kwargs):
        # Map 'dim' (torch style) to 'axis' (numpy style)
        target_axis = dim if dim is not None else axis
        res = np.sum(x, axis=target_axis, keepdims=keepdim, **kwargs)
        return res

    def mean(self, x, dim=None, axis=None, keepdim=False, **kwargs):
        target_axis = dim if dim is not None else axis
        res = np.mean(x, axis=target_axis, keepdims=keepdim, **kwargs)
        return res

    def matmul(self, a, b):
        return np.matmul(a, b)

    def abs(self, x):
        return np.abs(x)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def expand_dims(self, x, axis):
        return np.expand_dims(x, axis)


class SynapsesLoggerV49:
    def __init__(self, inputs, hidden, output, cItems, vInputs=None, ntype=0, useOpenTikosUtil=0):
        """
        useOpenTikosUtil (int):
            0 = CPU (Standard NumPy) - Best for small networks (nType 0,1,2).
            1 = GPU (PyTorch Standard) - Best for general acceleration if VRAM allows.
            2 = GPU (Tikos Utils) - Streaming Mode for massive tensors (OOM safe).
        """
        self.inputN = inputs
        self.hiddenN = hidden
        self.outputN = output
        self.layersN = 1 + 1 + len(hidden) - 1
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.cItems = cItems
        self.vInputs = vInputs
        self.ntype = ntype
        # ntype 5 (MoE Transformer): per-layer router/gate probabilities.
        # Dict {layer_idx: 1D array [num_experts]}. Optional; falls back to
        # router logits captured in output tuples, then to uniform weighting.
        self.gates = {}
        # ntype 6 (Convolutional Autoencoder): indices of ConvTranspose layers.
        # Their weights are [in_ch, out_ch, kH, kW], so channel dims are swapped
        # to the Conv2d convention after spatial aggregation. Mark layers via
        # markTransposed() or addWeights(..., transposed=True).
        self.transposed = set()

        # Mode Selection Logic
        self.mode = int(useOpenTikosUtil)

        # Fallback Logic
        if self.mode == 1 and not CUDA_AVAILABLE:
            print("Warning: CUDA not available. Reverting to Mode 0 (CPU).")
            self.mode = 0
        elif self.mode == 2:
            if not TIKOS_AVAILABLE:
                print("Warning: 'tutils' not found. Reverting to Mode 0 (CPU).")
                self.mode = 0
            # Note: tutils uses pycuda, which is distinct from torch.cuda.

        # Device is only relevant for Mode 1 (PyTorch GPU)
        self.device = torch.device("cuda" if self.mode == 1 else "cpu")

        # Initialize Adapter for Mode 2
        self.tikos_adapter = TikosLibAdapter() if self.mode == 2 else None

    def _get_lib(self):
        """Returns the appropriate library interface."""
        if self.mode == 0:
            return np  # Direct NumPy
        elif self.mode == 1:
            return torch  # Direct PyTorch
        elif self.mode == 2:
            return self.tikos_adapter  # Hybrid Adapter
        return np

    # def __call__(self, module, module_in, module_out):
    #     """Captures activations."""
    #     # Process Input
    #     if isinstance(module_in, tuple) and len(module_in) > 0 and isinstance(module_in[0], torch.Tensor):
    #         first_input = module_in[0][0].detach()
    #
    #         if self.mode == 1:  # PyTorch GPU Mode
    #             self.inputs.append(first_input)  # Keep on GPU
    #         else:
    #             # Mode 0 (NumPy) & Mode 2 (Tutils) both need CPU/NumPy data storage
    #             self.inputs.append(first_input.cpu().to(torch.float32).numpy())
    #     else:
    #         self.inputs.append(module_in)
    #
    #     # Process Output
    #     if isinstance(module_out, torch.Tensor):
    #         first_output = module_out[0].detach()
    #         if self.mode == 1:
    #             self.outputs.append(first_output)
    #         else:
    #             self.outputs.append(first_output.cpu().to(torch.float32).numpy())
    #     elif isinstance(module_out, tuple):
    #         processed_output_tuple = tuple(
    #             (t[0].detach() if self.mode == 1 else t[0].detach().cpu().to(torch.float32).numpy())
    #             if isinstance(t, torch.Tensor) else t
    #             for t in module_out
    #         )
    #         self.outputs.append(processed_output_tuple)
    #     else:
    #         self.outputs.append(module_out)

    def __call__(self, module, module_in, module_out):
        """Captures activations with strict Float32 casting."""
        # --- Process Input ---
        if isinstance(module_in, tuple) and len(module_in) > 0:
            # Grab first element. If it's a tensor, handle it.
            elem = module_in[0]
            if isinstance(elem, torch.Tensor):
                # Original logic took [0] (first item in batch)
                t_in = elem[0].detach()
                if self.mode == 1:
                    self.inputs.append(t_in.to(torch.float32))  # Force Float32
                else:
                    self.inputs.append(t_in.cpu().to(torch.float32).numpy())
            else:
                self.inputs.append(elem)
        elif isinstance(module_in, torch.Tensor):
            # Handle direct Tensor input (Missing in original code)
            t_in = module_in.detach()
            if self.mode == 1:
                self.inputs.append(t_in.to(torch.float32))
            else:
                self.inputs.append(t_in.cpu().to(torch.float32).numpy())
        else:
            self.inputs.append(module_in)

        # --- Process Output ---
        if isinstance(module_out, torch.Tensor):
            # Handle single Tensor output
            first_output = module_out[0].detach()
            if self.mode == 1:
                self.outputs.append(first_output.to(torch.float32))
            else:
                self.outputs.append(first_output.cpu().to(torch.float32).numpy())

        elif isinstance(module_out, tuple):
            # Handle Tuple output
            processed_list = []
            for t in module_out:
                if isinstance(t, torch.Tensor):
                    # Take first item in batch [0], detach, cast float32
                    t_proc = t[0].detach()
                    if self.mode == 1:
                        processed_list.append(t_proc.to(torch.float32))
                    else:
                        processed_list.append(t_proc.cpu().to(torch.float32).numpy())
                else:
                    processed_list.append(t)
            self.outputs.append(tuple(processed_list))
        else:
            self.outputs.append(module_out)

    def addWeights(self, weights, transposed=False):
        """Registers weights.

        transposed=True marks this layer as ConvTranspose (decoder) for
        ntype 6; its channel dims are swapped during causation."""
        if transposed:
            self.transposed.add(len(self.weights))
        if self.mode == 1:  # PyTorch GPU
            if isinstance(weights, torch.Tensor):
                self.weights.append(weights.detach().to(self.device))
            elif isinstance(weights, np.ndarray):
                self.weights.append(torch.from_numpy(weights).to(self.device))
            else:
                self.weights.append(weights)
        else:  # Mode 0 & 2 (NumPy storage)
            if isinstance(weights, torch.Tensor):
                self.weights.append(weights.detach().cpu().numpy())
            else:
                self.weights.append(weights)

    def setWeights(self, weights):
        self.weights = []
        self.transposed = set()
        for w in weights: self.addWeights(w)

    def markTransposed(self, layer_indices):
        """ntype 6: marks weight indices as ConvTranspose (decoder) layers."""
        for idx in layer_indices:
            self.transposed.add(int(idx))

    def _coerce_capture(self, item):
        """Coerce a captured input/output element to the active backend.

        In Mode 1 (PyTorch) the computation methods call torch ops (e.g.
        torch.sum(x, dim=...)) which reject numpy arrays. Inputs/outputs set
        directly (bypassing the __call__ hook) arrive as numpy arrays, so
        convert them here to match addWeights() behaviour."""
        if self.mode in (1,2):
            if isinstance(item, np.ndarray):
                return torch.from_numpy(item).to(torch.float32).to(self.device)
            if isinstance(item, torch.Tensor):
                return item.to(torch.float32).to(self.device)
            if isinstance(item, tuple):
                return tuple(self._coerce_capture(sub) for sub in item)
        return item

    def setInputs(self, inputs):
        self.inputs = [self._coerce_capture(x) for x in inputs]

    def setOutputs(self, outputs):
        self.outputs = [self._coerce_capture(x) for x in outputs]

    def setVInputs(self, vInputs):
        self.vInputs = vInputs

    # --- MoE SUPPORT (ntype 5) ---

    def addExpertGroup(self, expert_weights):
        """Registers one MoE layer from a list of per-expert 2D weights.

        Stacks each expert's [out, in] weight into a single 3D tensor
        [num_experts, out, in] and registers it as ONE layer via addWeights().
        A pre-stacked 3D tensor can also be passed directly to addWeights();
        any 3D weight is auto-detected as an MoE expert stack."""
        if expert_weights is None or len(expert_weights) == 0:
            return
        processed = []
        for w in expert_weights:
            if isinstance(w, torch.Tensor):
                processed.append(w.detach().cpu().to(torch.float32).numpy())
            else:
                processed.append(np.asarray(w))
        stacked = np.stack(processed, axis=0)  # [E, out, in]
        if self.mode == 1:
            self.addWeights(torch.from_numpy(stacked).to(torch.float32))
        else:
            self.addWeights(stacked)

    def addGates(self, layer_idx, gate_probs):
        """Registers router/gate probabilities for one MoE layer index.

        Accepts [num_experts], [Seq, num_experts] or [Batch, Seq, num_experts];
        token dims are averaged out. Values are normalized to sum to 1."""
        if isinstance(gate_probs, torch.Tensor):
            g = gate_probs.detach().cpu().to(torch.float32).numpy()
        else:
            g = np.asarray(gate_probs, dtype=np.float32)
        if g.ndim > 1:
            g = g.reshape(-1, g.shape[-1]).mean(axis=0)
        s = float(np.sum(g))
        if s > 0:
            g = g / s
        self.gates[int(layer_idx)] = g

    def setGates(self, gates):
        """Sets gate probabilities for all MoE layers.

        Accepts a dict {layer_idx: probs} or a list aligned with the weight
        list (use None entries for dense layers)."""
        self.gates = {}
        if gates is None:
            return
        items = gates.items() if isinstance(gates, dict) else enumerate(gates)
        for idx, g in items:
            if g is not None:
                self.addGates(idx, g)

    def _extract_router_probs(self, layer_idx, num_experts):
        """Recovers gate probabilities from captured outputs, if possible.

        MoE blocks typically return tuples like (hidden_states, router_logits).
        Scans the captured output tuple for an element whose last dim equals
        num_experts, softmaxes it over experts and averages over tokens."""
        if layer_idx >= len(self.outputs):
            return None
        out = self.outputs[layer_idx]
        if not isinstance(out, tuple):
            return None
        for elem in out[1:]:
            cand = elem
            if isinstance(cand, torch.Tensor):
                cand = cand.detach().cpu().to(torch.float32).numpy()
            if isinstance(cand, np.ndarray) and cand.ndim >= 1 and cand.shape[-1] == num_experts:
                logits = cand.reshape(-1, num_experts).astype(np.float32)
                z = logits - np.max(logits, axis=-1, keepdims=True)
                ez = np.exp(z)
                probs = ez / (np.sum(ez, axis=-1, keepdims=True) + 1e-9)
                return probs.mean(axis=0)  # [E]
        return None

    def _get_gate_probs(self, layer_idx, num_experts):
        """Gate resolution order: explicit setGates -> captured router logits
        -> uniform (1/num_experts)."""
        g = self.gates.get(int(layer_idx)) if isinstance(self.gates, dict) else None
        if g is None:
            g = self._extract_router_probs(layer_idx, num_experts)
        if g is None or len(g) != num_experts:
            g = np.full(num_experts, 1.0 / num_experts, dtype=np.float32)
        s = float(np.sum(g))
        if s > 0:
            g = g / s
        return g

    def _resolve_layer_weight(self, layer_idx):
        """Returns an effective 2D weight for a layer.

        Dense (1D/2D) weights pass through unchanged. Stacked MoE expert
        weights [num_experts, out, in] are collapsed to [out, in] via a
        gate-weighted sum over experts."""
        w = self.weights[layer_idx]
        ndim = w.dim() if isinstance(w, torch.Tensor) else np.ndim(w)
        if ndim != 3:
            return w
        num_experts = w.shape[0]
        g = self._get_gate_probs(layer_idx, num_experts)
        if isinstance(w, torch.Tensor):
            g_t = torch.from_numpy(np.asarray(g, dtype=np.float32)).to(w.device)
            return (w.to(torch.float32) * g_t.view(-1, 1, 1)).sum(dim=0)
        else:
            return np.tensordot(np.asarray(g, dtype=np.float32), w, axes=1)

    def clear(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.gates = {}
        self.transposed = set()

    def clearWOW(self):
        self.inputs = []
        self.outputs = []
        self.gates = {}  # router probs are input-dependent

    # --- COMPUTATION METHODS ---

    def generateSynapsesNodeGroupIH(self):
        try:
            lib = self._get_lib()
            if self.ntype in [3, 4, 5, 6]: return None

            if self.ntype == 2:
                # Transformer Input Logic
                input_act = self.inputs[0]
                weights = self.weights[0]

                if self.mode == 1:  # PyTorch
                    w_sum = lib.sum(weights, dim=0)
                    alpha_input = self.outputs[0] * lib.sum(input_act, dim=1).unsqueeze(1)
                    alpha_reduced = lib.mean(alpha_input, dim=1)
                    return w_sum / (alpha_reduced.unsqueeze(1) + 1e-9)
                else:  # Numpy & Tutils
                    w_sum = lib.sum(weights, axis=0)
                    sum_inp = lib.sum(input_act, axis=1)

                    if self.mode == 2:
                        sum_inp = lib.expand_dims(sum_inp, 1)
                        alpha_input = lib.multiply(self.outputs[0], sum_inp)
                        alpha_reduced = lib.mean(alpha_input, axis=1)
                        denom = lib.expand_dims(alpha_reduced, 1) + 1e-9
                        return lib.divide(w_sum, denom)
                    else:
                        sum_inp = sum_inp[:, np.newaxis]
                        alpha_input = self.outputs[0] * sum_inp
                        alpha_reduced = np.mean(alpha_input, axis=1)
                        return w_sum / (alpha_reduced[:, np.newaxis] + 1e-9)

            else:  # Standard FeedForward
                weights = self.weights[0]
                input_feature = self.inputs[0] if self.ntype == 0 else self.inputs[0][0]

                if self.mode == 1:  # Torch
                    denom = input_feature * input_feature + 1e-9
                    return weights.t() / denom.unsqueeze(1)
                elif self.mode == 2:  # Tutils
                    denom = lib.multiply(input_feature, input_feature) + 1e-9
                    denom = lib.expand_dims(denom, 1)
                    return lib.divide(weights.T, denom)
                else:  # Numpy
                    denom = input_feature * input_feature + 1e-9
                    return weights.T / denom[:, np.newaxis]
        except Exception as e:
            print(f"IH Error: {e}")
            return None

    def generateSynapsesNodeGroupHH_Sequential(self, prev_delta, layer_idx):
        try:
            lib = self._get_lib()
            presynaptic_activation = self.inputs[layer_idx]
            weights = self.weights[layer_idx]

            if self.mode == 2:
                denominator = lib.multiply(prev_delta, presynaptic_activation)
                inv_denom = lib.divide(1.0, (denominator + 1e-9))
                w_proc = weights.T if weights.ndim == 2 else weights
                return lib.matmul(inv_denom, w_proc)
            elif self.mode == 1:
                denominator = prev_delta * presynaptic_activation
                inv_denom = 1 / (denominator + 1e-9)
                w_proc = weights.t() if weights.dim() == 2 else weights
                return lib.matmul(inv_denom, w_proc)
            else:
                denominator = prev_delta * presynaptic_activation
                return (1 / (denominator + 1e-9)) @ (weights.T if weights.ndim == 2 else weights)
        except Exception as e:
            print(f"HH Error: {e}")
            return None

    def generateSynapsesNodeGroupHO(self, presynaptic_delta):
        try:
            lib = self._get_lib()
            if self.ntype in [3, 4, 5, 6] or presynaptic_delta is None: return None

            raw_act = self.inputs[-1]

            if self.mode == 2:
                denominator = lib.multiply(presynaptic_delta, raw_act)
                w_t = self.weights[-1].T if self.weights[-1].ndim == 2 else self.weights[-1][:, np.newaxis]

                # Broadcasting manually for Tutils/Numpy structure
                w_expanded = lib.expand_dims(w_t, 0)  # [1, H, O]
                d_expanded = lib.expand_dims(denominator, 2)  # [In, H, 1]

                term = lib.divide(w_expanded, (d_expanded + 1e-9))
                return lib.sum(term, axis=1)

            elif self.mode == 1:
                denominator = presynaptic_delta * raw_act
                w_t = self.weights[-1].t() if self.weights[-1].dim() == 2 else self.weights[-1].unsqueeze(1)
                term = w_t.unsqueeze(0) / (denominator.unsqueeze(2) + 1e-9)
                return lib.sum(term, dim=1)
            else:
                denominator = presynaptic_delta * raw_act
                w_t = self.weights[-1].T if self.weights[-1].ndim == 2 else self.weights[-1][:, np.newaxis]
                term = w_t[np.newaxis, :, :] / (denominator[:, :, np.newaxis] + 1e-9)
                return np.sum(term, axis=1)
        except Exception as e:
            print(f"HO Error: {e}")
            return None

    def generateSynapsesCNN(self, cae=False):
        lib = self._get_lib()
        all_deltas = []

        # FIX 1: Bound the loop to the shortest list to prevent IndexError
        loop_len = min(len(self.weights), len(self.inputs), len(self.outputs))

        for i in range(loop_len):
            try:
                input_act = self.inputs[i]
                weights = self.weights[i]

                # FIX 2: Safely extract from tuples to prevent secondary index errors
                if isinstance(self.outputs[i], tuple):
                    if len(self.outputs[i]) > 0:
                        out_raw = self.outputs[i][0]
                    else:
                        continue  # Skip if the tuple is entirely empty
                else:
                    out_raw = self.outputs[i]

                if self.mode == 1:  # Torch
                    if input_act.dim() == 4:
                        inp_red = lib.mean(input_act[0], dim=(1, 2)).unsqueeze(0)
                    elif input_act.dim() == 3:
                        inp_red = lib.mean(input_act, dim=(1, 2)).unsqueeze(0)
                    else:
                        inp_red = input_act.unsqueeze(0)

                    agg_w = lib.mean(weights, dim=(2, 3)) if weights.dim() == 4 else weights
                    if cae and i in self.transposed and agg_w.dim() == 2:
                        agg_w = agg_w.t()  # ConvTranspose: [in_ch, out_ch] -> [out_ch, in_ch]

                    if out_raw.dim() == 3:
                        out_red = lib.mean(out_raw, dim=(1, 2)).unsqueeze(0)
                    else:
                        out_red = out_raw.unsqueeze(0)

                    row_sums = lib.sum(inp_red, dim=1)
                    alpha = lib.mean(out_red * row_sums.unsqueeze(1), dim=1)
                    denom = torch.where(alpha.abs() < 1e-9, alpha + 1e-9, alpha).unsqueeze(1)
                    delta = agg_w.t() / denom

                else:  # Numpy (Mode 0) & Tutils (Mode 2)
                    if input_act.ndim == 3:
                        inp_red = np.mean(input_act, axis=(1, 2))[np.newaxis, :]
                    else:
                        inp_red = np.atleast_2d(input_act)

                    agg_w = np.mean(weights, axis=(2, 3)) if weights.ndim == 4 else weights
                    if cae and i in self.transposed and agg_w.ndim == 2:
                        agg_w = agg_w.T  # ConvTranspose: [in_ch, out_ch] -> [out_ch, in_ch]

                    if out_raw.ndim == 3:
                        out_red = np.mean(out_raw, axis=(1, 2))[np.newaxis, :]
                    else:
                        out_red = np.atleast_2d(out_raw)

                    row_sums = np.sum(inp_red, axis=1)

                    if self.mode == 2:
                        prod = lib.multiply(out_red, row_sums[:, np.newaxis])
                        alpha = np.mean(prod, axis=1)
                        denom = (alpha[:, np.newaxis] + 1e-9)
                        delta = lib.divide(agg_w.T, denom)
                    else:
                        alpha = np.mean(out_red * row_sums[:, np.newaxis], axis=1)
                        delta = agg_w.T / (alpha[:, np.newaxis] + 1e-9)

                all_deltas.append(delta)
            except Exception as e:
                print(f"CNN Error at layer {i}: {e}")
                break
        return all_deltas

    def generateSynapsesCAE(self):
        """ntype 6: Convolutional Autoencoder.

        Same per-layer walk as the CNN (ntype 4) across encoder + decoder.
        Layers marked as ConvTranspose (markTransposed() or
        addWeights(..., transposed=True)) have their channel dims swapped to
        the Conv2d [out_ch, in_ch] convention after spatial aggregation."""
        return self.generateSynapsesCNN(cae=True)

    def generateSynapsesNodeGroupAll_Transformer(self, moe=False):
        lib = self._get_lib()
        all_deltas = []

        # FIX 1: Safety Check - Ensure all lists have elements to process
        if not self.weights or not self.inputs or not self.outputs:
            return []

        # FIX 2: Bound the loop to the shortest list to prevent IndexError
        loop_len = min(len(self.weights), len(self.inputs), len(self.outputs))

        for i in range(loop_len):
            try:
                # ==========================================================
                # STEP 1: PREPARE INPUT ACTIVATION
                # ==========================================================
                if i == 0:
                    raw_input = self.inputs[i]
                    # Handle Tuple Input
                    if isinstance(raw_input, tuple):
                        if len(raw_input) > 0:
                            input_act = raw_input[0]
                        else:
                            continue  # Empty tuple
                    else:
                        input_act = raw_input

                    # Force Input to be 2D [Seq, Hidden]
                    if self.mode == 1 and isinstance(input_act, torch.Tensor):
                        input_act = input_act.to(torch.float32)
                        if input_act.dim() == 1:
                            input_act = input_act.unsqueeze(0)
                        elif input_act.dim() == 3:
                            input_act = input_act.reshape(-1, input_act.shape[-1])
                    elif isinstance(input_act, np.ndarray):
                        if input_act.ndim == 1:
                            input_act = input_act[np.newaxis, :]
                else:
                    # If chain is broken or inputs missing, stop
                    if not all_deltas:
                        break

                    p_delta = all_deltas[-1]
                    p_w = self._resolve_layer_weight(i - 1) if moe else self.weights[i - 1]

                    # --- CRITICAL FIX 1: Handle 1D DELTA ---
                    if self.mode == 1:
                        if p_delta.dim() == 1:
                            p_delta = p_delta.unsqueeze(0)  # [Hidden] -> [1, Hidden]

                        # --- CRITICAL FIX 2: Handle 1D WEIGHTS (Biases) ---
                        # The crash "tuple index out of range" happens here if p_w is 1D
                        if p_w.dim() == 2:
                            w_t = p_w.t()
                        else:
                            # If 1D (Bias), reshape to [N, 1] so it acts like a column vector
                            w_t = p_w.reshape(-1, 1)

                        # Validate Shapes before MatMul
                        # p_delta: [Seq, In_Hidden]
                        # w_t:     [In_Hidden, Out_Hidden] (or [N, 1])
                        if p_delta.shape[1] != w_t.shape[0]:
                            continue

                        input_act = lib.matmul(p_delta, w_t)

                    else:  # Numpy / Tutils
                        if hasattr(p_delta, 'ndim') and p_delta.ndim == 1:
                            p_delta = p_delta[np.newaxis, :]

                        # NumPy Logic for 1D Weights
                        if p_w.ndim == 2:
                            w_t = p_w.T
                        else:
                            w_t = p_w.reshape(-1, 1)

                        input_act = lib.matmul(p_delta, w_t)

                # ==========================================================
                # STEP 2: PREPARE OUTPUT ACTIVATION
                # ==========================================================
                curr_out = self.outputs[i]
                if isinstance(curr_out, tuple):
                    if len(curr_out) > 0:
                        out_act = curr_out[0]
                    else:
                        continue
                else:
                    out_act = curr_out

                # Force Float32 & Check 2D
                if self.mode == 1:
                    if isinstance(out_act, torch.Tensor):
                        out_act = out_act.to(torch.float32)
                        if out_act.dim() == 1:
                            out_act = out_act.unsqueeze(0)
                        elif out_act.dim() == 3:
                            out_act = out_act.reshape(-1, out_act.shape[-1])

                    if isinstance(input_act, torch.Tensor):
                        input_act = input_act.to(torch.float32)

                # ==========================================================
                # STEP 3: CALCULATE DELTA
                # ==========================================================
                if self.mode == 1:  # Torch
                    sum_inp = lib.sum(input_act, dim=1).unsqueeze(1)  # [Seq, 1]

                    # Broadcasting Safety
                    if out_act.shape[0] != sum_inp.shape[0]:
                        if out_act.shape[0] == 1:
                            out_act = out_act.expand(sum_inp.shape[0], -1)
                        else:
                            continue

                    alpha = lib.mean(out_act * sum_inp, dim=1)
                    denom = torch.where(alpha.abs() < 1e-9, alpha + 1e-9, alpha).unsqueeze(1)

                    agg_w = self._resolve_layer_weight(i) if moe else self.weights[i]
                    if agg_w.dim() > 1:
                        agg_w = lib.sum(agg_w, dim=0)

                    # Ensure 2D for broadcasting [1, Hidden]
                    agg_w_expanded = agg_w.unsqueeze(0)

                    delta = agg_w_expanded / denom
                    all_deltas.append(delta)

                else:  # Numpy / Tutils
                    w_i = self._resolve_layer_weight(i) if moe else self.weights[i]
                    sum_inp = np.sum(input_act, axis=1)[:, np.newaxis]

                    # Sync dims
                    if out_act.shape[0] != sum_inp.shape[0]:
                        if out_act.shape[0] == 1:
                            out_act = np.broadcast_to(out_act, (sum_inp.shape[0], out_act.shape[1]))

                    if self.mode == 2:
                        prod = lib.multiply(out_act, sum_inp)
                        alpha = np.mean(prod, axis=1)
                        agg_w = w_i if w_i.ndim == 1 else np.sum(w_i, axis=0)
                        denom = alpha[:, np.newaxis] + 1e-9
                        all_deltas.append(lib.divide(agg_w, denom))
                    else:
                        alpha = np.mean(out_act * sum_inp, axis=1)
                        agg_w = w_i if w_i.ndim == 1 else np.sum(w_i, axis=0)
                        all_deltas.append(agg_w / (alpha[:, np.newaxis] + 1e-9))

            except Exception as e:
                print(f"Transformer Logic Error Layer {i}: {e}")
                continue

        return all_deltas

    def generateSynapsesNodeGroupAll_MoE(self):
        """ntype 5: MoE Transformer.

        Identical layer walk to the dense transformer (ntype 3), except that
        stacked expert weights [num_experts, out, in] are collapsed into an
        effective dense weight via gate-weighted aggregation (see
        _resolve_layer_weight / _get_gate_probs)."""
        return self.generateSynapsesNodeGroupAll_Transformer(moe=True)

    # def generateSynapsesCNN(self):
    #     lib = self._get_lib()
    #     all_deltas = []
    #     for i in range(len(self.weights)):
    #         try:
    #             input_act = self.inputs[i]
    #             weights = self.weights[i]
    #             out_raw = self.outputs[i][0] if isinstance(self.outputs[i], tuple) else self.outputs[i]
    #
    #             if self.mode == 1:  # Torch
    #                 if input_act.dim() == 4:
    #                     inp_red = lib.mean(input_act[0], dim=(1, 2)).unsqueeze(0)
    #                 elif input_act.dim() == 3:
    #                     inp_red = lib.mean(input_act, dim=(1, 2)).unsqueeze(0)
    #                 else:
    #                     inp_red = input_act.unsqueeze(0)
    #
    #                 agg_w = lib.mean(weights, dim=(2, 3)) if weights.dim() == 4 else weights
    #
    #                 if out_raw.dim() == 3:
    #                     out_red = lib.mean(out_raw, dim=(1, 2)).unsqueeze(0)
    #                 else:
    #                     out_red = out_raw.unsqueeze(0)
    #
    #                 row_sums = lib.sum(inp_red, dim=1)
    #                 alpha = lib.mean(out_red * row_sums.unsqueeze(1), dim=1)
    #                 denom = torch.where(alpha.abs() < 1e-9, alpha + 1e-9, alpha).unsqueeze(1)
    #                 delta = agg_w.t() / denom
    #
    #             else:  # Numpy (Mode 0) & Tutils (Mode 2)
    #                 if input_act.ndim == 3:
    #                     inp_red = np.mean(input_act, axis=(1, 2))[np.newaxis, :]
    #                 else:
    #                     inp_red = np.atleast_2d(input_act)
    #
    #                 agg_w = np.mean(weights, axis=(2, 3)) if weights.ndim == 4 else weights
    #
    #                 if out_raw.ndim == 3:
    #                     out_red = np.mean(out_raw, axis=(1, 2))[np.newaxis, :]
    #                 else:
    #                     out_red = np.atleast_2d(out_raw)
    #
    #                 row_sums = np.sum(inp_red, axis=1)
    #
    #                 if self.mode == 2:
    #                     prod = lib.multiply(out_red, row_sums[:, np.newaxis])
    #                     alpha = np.mean(prod, axis=1)
    #                     denom = (alpha[:, np.newaxis] + 1e-9)
    #                     delta = lib.divide(agg_w.T, denom)
    #                 else:
    #                     alpha = np.mean(out_red * row_sums[:, np.newaxis], axis=1)
    #                     delta = agg_w.T / (alpha[:, np.newaxis] + 1e-9)
    #
    #             all_deltas.append(delta)
    #         except Exception as e:
    #             print(f"CNN Error: {e}")
    #             break
    #     return all_deltas
    #
    # def generateSynapsesNodeGroupAll_Transformer(self):
    #     lib = self._get_lib()
    #     all_deltas = []
    #
    #     if not self.weights:
    #         return []
    #
    #     for i in range(len(self.weights)):
    #         try:
    #             # ==========================================================
    #             # STEP 1: PREPARE INPUT ACTIVATION
    #             # ==========================================================
    #             if i == 0:
    #                 raw_input = self.inputs[i]
    #                 # Handle Tuple Input
    #                 if isinstance(raw_input, tuple):
    #                     if len(raw_input) > 0:
    #                         input_act = raw_input[0]
    #                     else:
    #                         continue  # Empty tuple
    #                 else:
    #                     input_act = raw_input
    #
    #                 # Force Input to be 2D [Seq, Hidden]
    #                 if self.mode == 1 and isinstance(input_act, torch.Tensor):
    #                     input_act = input_act.to(torch.float32)
    #                     if input_act.dim() == 1:
    #                         input_act = input_act.unsqueeze(0)
    #                     elif input_act.dim() == 3:
    #                         input_act = input_act.reshape(-1, input_act.shape[-1])
    #                 elif isinstance(input_act, np.ndarray):
    #                     if input_act.ndim == 1:
    #                         input_act = input_act[np.newaxis, :]
    #             else:
    #                 # If chain is broken or inputs missing, stop
    #                 if not all_deltas:
    #                     break
    #
    #                 p_delta = all_deltas[-1]
    #                 p_w = self.weights[i - 1]
    #
    #                 # --- CRITICAL FIX 1: Handle 1D DELTA ---
    #                 if self.mode == 1:
    #                     if p_delta.dim() == 1:
    #                         p_delta = p_delta.unsqueeze(0)  # [Hidden] -> [1, Hidden]
    #
    #                     # --- CRITICAL FIX 2: Handle 1D WEIGHTS (Biases) ---
    #                     # The crash "tuple index out of range" happens here if p_w is 1D
    #                     if p_w.dim() == 2:
    #                         w_t = p_w.t()
    #                     else:
    #                         # If 1D (Bias), reshape to [N, 1] so it acts like a column vector
    #                         w_t = p_w.reshape(-1, 1)
    #
    #                     # Validate Shapes before MatMul
    #                     # p_delta: [Seq, In_Hidden]
    #                     # w_t:     [In_Hidden, Out_Hidden] (or [N, 1])
    #                     if p_delta.shape[1] != w_t.shape[0]:
    #                         continue
    #
    #                     input_act = lib.matmul(p_delta, w_t)
    #
    #                 else:  # Numpy / Tutils
    #                     if hasattr(p_delta, 'ndim') and p_delta.ndim == 1:
    #                         p_delta = p_delta[np.newaxis, :]
    #
    #                     # NumPy Logic for 1D Weights
    #                     if p_w.ndim == 2:
    #                         w_t = p_w.T
    #                     else:
    #                         w_t = p_w.reshape(-1, 1)
    #
    #                     input_act = lib.matmul(p_delta, w_t)
    #
    #             # ==========================================================
    #             # STEP 2: PREPARE OUTPUT ACTIVATION
    #             # ==========================================================
    #             curr_out = self.outputs[i]
    #             if isinstance(curr_out, tuple):
    #                 if len(curr_out) > 0:
    #                     out_act = curr_out[0]
    #                 else:
    #                     continue
    #             else:
    #                 out_act = curr_out
    #
    #             # Force Float32 & Check 2D
    #             if self.mode == 1:
    #                 if isinstance(out_act, torch.Tensor):
    #                     out_act = out_act.to(torch.float32)
    #                     if out_act.dim() == 1:
    #                         out_act = out_act.unsqueeze(0)
    #                     elif out_act.dim() == 3:
    #                         out_act = out_act.reshape(-1, out_act.shape[-1])
    #
    #                 if isinstance(input_act, torch.Tensor):
    #                     input_act = input_act.to(torch.float32)
    #
    #             # ==========================================================
    #             # STEP 3: CALCULATE DELTA
    #             # ==========================================================
    #             if self.mode == 1:  # Torch
    #                 sum_inp = lib.sum(input_act, dim=1).unsqueeze(1)  # [Seq, 1]
    #
    #                 # Broadcasting Safety
    #                 if out_act.shape[0] != sum_inp.shape[0]:
    #                     if out_act.shape[0] == 1:
    #                         out_act = out_act.expand(sum_inp.shape[0], -1)
    #                     else:
    #                         continue
    #
    #                 alpha = lib.mean(out_act * sum_inp, dim=1)
    #                 denom = torch.where(alpha.abs() < 1e-9, alpha + 1e-9, alpha).unsqueeze(1)
    #
    #                 agg_w = self.weights[i]
    #                 if agg_w.dim() > 1:
    #                     agg_w = lib.sum(agg_w, dim=0)
    #
    #                 # Ensure 2D for broadcasting [1, Hidden]
    #                 agg_w_expanded = agg_w.unsqueeze(0)
    #
    #                 delta = agg_w_expanded / denom
    #                 all_deltas.append(delta)
    #
    #             else:  # Numpy / Tutils
    #                 sum_inp = np.sum(input_act, axis=1)[:, np.newaxis]
    #
    #                 # Sync dims
    #                 if out_act.shape[0] != sum_inp.shape[0]:
    #                     if out_act.shape[0] == 1:
    #                         out_act = np.broadcast_to(out_act, (sum_inp.shape[0], out_act.shape[1]))
    #
    #                 if self.mode == 2:
    #                     prod = lib.multiply(out_act, sum_inp)
    #                     alpha = np.mean(prod, axis=1)
    #                     agg_w = self.weights[i] if self.weights[i].ndim == 1 else np.sum(self.weights[i], axis=0)
    #                     denom = alpha[:, np.newaxis] + 1e-9
    #                     all_deltas.append(lib.divide(agg_w, denom))
    #                 else:
    #                     alpha = np.mean(out_act * sum_inp, axis=1)
    #                     agg_w = self.weights[i] if self.weights[i].ndim == 1 else np.sum(self.weights[i], axis=0)
    #                     all_deltas.append(agg_w / (alpha[:, np.newaxis] + 1e-9))
    #
    #         except Exception as e:
    #             print(f"Transformer Logic Error Layer {i}: {e}")
    #             continue
    #
    #     return all_deltas

    # def generateSynapsesNodeGroupAll_Transformer(self):
    #     lib = self._get_lib()
    #     all_deltas = []
    #
    #     if not self.weights:
    #         return []
    #
    #     for i in range(len(self.weights)):
    #         try:
    #             # ==========================================================
    #             # STEP 1: PREPARE INPUT ACTIVATION
    #             # ==========================================================
    #             if i == 0:
    #                 raw_input = self.inputs[i]
    #                 # Handle Tuple Input
    #                 if isinstance(raw_input, tuple):
    #                     if len(raw_input) > 0:
    #                         input_act = raw_input[0]
    #                     else:
    #                         continue
    #                 else:
    #                     input_act = raw_input
    #
    #                 # --- Input Standardization ---
    #                 if self.mode == 1 and isinstance(input_act, torch.Tensor):
    #                     # PyTorch Mode
    #                     input_act = input_act.to(torch.float32)
    #                     if input_act.dim() == 1:
    #                         input_act = input_act.unsqueeze(0)
    #                     elif input_act.dim() == 3:
    #                         input_act = input_act.reshape(-1, input_act.shape[-1])
    #
    #                 elif isinstance(input_act, np.ndarray):
    #                     # NumPy/Tutils Mode
    #                     if input_act.ndim == 1:
    #                         input_act = input_act[np.newaxis, :]
    #                     elif input_act.ndim == 3:
    #                         # Flatten Batch: [B, Seq, H] -> [B*Seq, H]
    #                         input_act = input_act.reshape(-1, input_act.shape[-1])
    #
    #                     # Fix Contiguity for Input
    #                     if not input_act.flags['C_CONTIGUOUS']:
    #                         input_act = np.ascontiguousarray(input_act)
    #
    #             else:
    #                 # Proceed only if we have a valid chain
    #                 if not all_deltas:
    #                     break
    #
    #                 p_delta = all_deltas[-1]
    #                 p_w = self.weights[i - 1]
    #
    #                 # ------------------------------------------------------
    #                 # CRITICAL FIX: CONTIGUITY & SHAPE CHECKING
    #                 # ------------------------------------------------------
    #                 if self.mode == 1:  # Torch
    #                     if p_delta.dim() == 1:
    #                         p_delta = p_delta.unsqueeze(0)
    #
    #                     # Handle Bias vs Weight
    #                     if p_w.dim() == 2:
    #                         w_t = p_w.t()
    #                     else:
    #                         w_t = p_w.reshape(-1, 1)
    #
    #                     if p_delta.shape[1] != w_t.shape[0]:
    #                         continue
    #
    #                     input_act = lib.matmul(p_delta, w_t)
    #
    #                 else:  # NumPy / Tutils (Mode 0 & 2)
    #                     # 1. Ensure Delta is 2D
    #                     if hasattr(p_delta, 'ndim') and p_delta.ndim == 1:
    #                         p_delta = p_delta[np.newaxis, :]
    #
    #                     # 2. Fix Contiguity for Delta (Mode 2 crash fix)
    #                     if isinstance(p_delta, np.ndarray) and not p_delta.flags['C_CONTIGUOUS']:
    #                         p_delta = np.ascontiguousarray(p_delta)
    #
    #                     # 3. Handle Weights & Transpose
    #                     if p_w.ndim == 2:
    #                         # .T creates non-contiguous view -> Must fix!
    #                         w_t = np.ascontiguousarray(p_w.T)
    #                     else:
    #                         w_t = p_w.reshape(-1, 1)
    #                         # Reshape usually returns view too
    #                         if not w_t.flags['C_CONTIGUOUS']:
    #                             w_t = np.ascontiguousarray(w_t)
    #
    #                     # 4. Strict Shape Check (Prevents matmul core dimension mismatch)
    #                     # p_delta: [N, A], w_t: [A, B] -> A must match
    #                     if p_delta.shape[1] != w_t.shape[0]:
    #                         continue
    #
    #                     input_act = lib.matmul(p_delta, w_t)
    #
    #             # ==========================================================
    #             # STEP 2: PREPARE OUTPUT ACTIVATION
    #             # ==========================================================
    #             curr_out = self.outputs[i]
    #             if isinstance(curr_out, tuple):
    #                 if len(curr_out) > 0:
    #                     out_act = curr_out[0]
    #                 else:
    #                     continue
    #             else:
    #                 out_act = curr_out
    #
    #             # Standardize Output
    #             if self.mode == 1:
    #                 if isinstance(out_act, torch.Tensor):
    #                     out_act = out_act.to(torch.float32)
    #                     if out_act.dim() == 1:
    #                         out_act = out_act.unsqueeze(0)
    #                     elif out_act.dim() == 3:
    #                         out_act = out_act.reshape(-1, out_act.shape[-1])
    #
    #                 if isinstance(input_act, torch.Tensor):
    #                     input_act = input_act.to(torch.float32)
    #
    #             elif isinstance(out_act, np.ndarray):
    #                 # Fix Contiguity for Output
    #                 if out_act.ndim == 3:
    #                     out_act = out_act.reshape(-1, out_act.shape[-1])
    #                 if not out_act.flags['C_CONTIGUOUS']:
    #                     out_act = np.ascontiguousarray(out_act)
    #
    #             # ==========================================================
    #             # STEP 3: CALCULATE DELTA
    #             # ==========================================================
    #             if self.mode == 1:  # Torch
    #                 sum_inp = lib.sum(input_act, dim=1).unsqueeze(1)
    #
    #                 if out_act.shape[0] != sum_inp.shape[0]:
    #                     if out_act.shape[0] == 1:
    #                         out_act = out_act.expand(sum_inp.shape[0], -1)
    #                     else:
    #                         continue
    #
    #                 alpha = lib.mean(out_act * sum_inp, dim=1)
    #                 denom = torch.where(alpha.abs() < 1e-9, alpha + 1e-9, alpha).unsqueeze(1)
    #
    #                 agg_w = self.weights[i]
    #                 if agg_w.dim() > 1:
    #                     agg_w = lib.sum(agg_w, dim=0)
    #
    #                 agg_w_expanded = agg_w.unsqueeze(0)
    #                 delta = agg_w_expanded / denom
    #                 all_deltas.append(delta)
    #
    #             else:  # NumPy / Tutils
    #                 sum_inp = np.sum(input_act, axis=1)[:, np.newaxis]
    #
    #                 # Broadcasting Check
    #                 if out_act.shape[0] != sum_inp.shape[0]:
    #                     if out_act.shape[0] == 1:
    #                         out_act = np.broadcast_to(out_act, (sum_inp.shape[0], out_act.shape[1]))
    #                     # Ensure broadcast result is contiguous if needed?
    #                     # Usually logic handles it, but let's be safe for Tutils
    #                     if self.mode == 2:
    #                         out_act = np.ascontiguousarray(out_act)
    #
    #                 if self.mode == 2:
    #                     prod = lib.multiply(out_act, sum_inp)
    #                     alpha = np.mean(prod, axis=1)
    #                     agg_w = self.weights[i] if self.weights[i].ndim == 1 else np.sum(self.weights[i], axis=0)
    #                     denom = alpha[:, np.newaxis] + 1e-9
    #
    #                     # Ensure Delta Result is contiguous for next iteration
    #                     res = lib.divide(agg_w, denom)
    #                     if isinstance(res, np.ndarray) and not res.flags['C_CONTIGUOUS']:
    #                         res = np.ascontiguousarray(res)
    #                     all_deltas.append(res)
    #                 else:
    #                     alpha = np.mean(out_act * sum_inp, axis=1)
    #                     agg_w = self.weights[i] if self.weights[i].ndim == 1 else np.sum(self.weights[i], axis=0)
    #                     all_deltas.append(agg_w / (alpha[:, np.newaxis] + 1e-9))
    #
    #         except Exception as e:
    #             print(f"Transformer Logic Error Layer {i}: {e}") # Optional: Comment out to suppress logs
    #             continue
    #
    #     return all_deltas

    def _process_layer_causation(self, delta_matrix, cicF):
        """
        Processes causation logic.
        - Fixes accumulation bug (uses += instead of recursive division).
        - Rounds float values to 8 decimals to remove scientific notation.
        """
        if delta_matrix is None or not cicF: return cicF

        # Mode 1 (Torch) needs conversion to Numpy for processing
        if self.mode == 1 and isinstance(delta_matrix, torch.Tensor):
            dm = delta_matrix.detach().cpu().numpy()
        else:
            dm = delta_matrix

        # Ensure dm is at least 1D
        if dm.ndim == 0:
            dm = np.expand_dims(dm, 0)

        num_layer_inputs = dm.shape[0]
        process_limit = min(len(cicF), num_layer_inputs)

        for i in range(process_limit):
            rD_vector = dm[i]
            rD_scalar = float(np.mean(np.abs(rD_vector)))

            record = cicF[i]
            record[4] += rD_scalar
            record[1] = (record[1] + rD_scalar) / (num_layer_inputs if num_layer_inputs > 0 else 1)

            # Normalize probabilities
            A_val = record[1] if record[1] != 0 else 1.0
            record[2] = float((record[2] + rD_scalar) / (A_val + 1e-9))
            record[3] = float((record[3] + (1.0 - rD_scalar)) / (A_val + 1e-9))

            if record[2] > 1.0: record[2] = 1.0
            if record[3] > 1.0: record[3] = 1.0

            # Store trace with clean rounding
            if self.ntype in [0, 1, 2]:
                clean_vector = [round(float(x), 15) for x in rD_vector]
                record[5].extend(clean_vector)
            else:
                record[5].append(round(rD_scalar, 8))
        return cicF

    # def selectRegresstion(self):
    #     valindexId = len(self.outputs) - 1
    #     out = self.outputs[valindexId]
    #     if isinstance(out, tuple): out = out[0]
    #
    #     if self.mode == 1 and isinstance(out, torch.Tensor):
    #         outArray = out.detach().cpu().numpy()
    #     else:
    #         outArray = out
    #
    #     if outArray.ndim > 1:
    #         return [(row.argmax(), np.amax(row)) for row in outArray]
    #     else:
    #         return [(outArray.argmax(), np.amax(outArray))]

    def selectRegresstion(self):
        # Safety Check: If no outputs, return a default safe value
        if not self.outputs:
            return [(0, 0.0)]

        valindexId = len(self.outputs) - 1
        out = self.outputs[valindexId]

        # Unpack tuple if necessary
        if isinstance(out, tuple):
            if len(out) > 0:
                out = out[0]
            else:
                return [(0, 0.0)]

        # Convert to Numpy if needed
        if self.mode == 1 and isinstance(out, torch.Tensor):
            outArray = out.detach().cpu().numpy()
        else:
            outArray = out

        # Ensure outArray is valid
        if outArray is None or outArray.size == 0:
            return [(0, 0.0)]

        if outArray.ndim > 1:
            return [(row.argmax(), np.amax(row)) for row in outArray]
        else:
            return [(outArray.argmax(), np.amax(outArray))]

    def caseCausationAbstration(self):
        # # Safety Check: Ensure inputs exist
        # if not self.inputs or len(self.inputs) == 0:
        #     # return [[0, 0.0, 0.0, 0.0, 0.0, []]]
        #     return []
        #
        # try:
        #     if self.ntype == 4:
        #         num_items = int(np.prod(self.inputs[0].shape))
        #     elif self.ntype == 0:
        #         num_items = len(self.inputs[0])
        #     elif self.ntype == 3:
        #         num_items = self.inputs[0].shape[0]
        #     else:
        #         num_items = len(self.inputs[0][0])
        # except:
        #     num_items = 0
        try:
            if not self.inputs:
                return []

            input_sample = self.inputs[0]

            if self.ntype in [4, 6]:  # CNN / Conv Autoencoder
                num_items = int(np.prod(input_sample.shape))
            elif self.ntype in [3, 5]:  # Transformer (3: Dense, 5: MoE)
                # If shape is (Seq, Hidden), use Seq (dim 0)
                # If shape is (Batch, Seq, Hidden), use Seq (dim 1)
                # Note: __call__ usually strips the first Batch dim [0][0]
                if hasattr(input_sample, 'shape'):
                    num_items = input_sample.shape[0]
                else:
                    num_items = len(input_sample)
            elif self.ntype == 0:
                num_items = len(input_sample)
            else:
                num_items = len(input_sample[0]) if hasattr(input_sample, '__len__') else 0
        except Exception as e:
            print(f"Error calculating num_items: {e}")
            num_items = 0

        cicF = [[i, 0.0, 0.0, 0.0, 0.0, []] for i in range(num_items)]

        if self.ntype in [4, 6]:
            deltas = (self.generateSynapsesCAE() if self.ntype == 6
                      else self.generateSynapsesCNN())
            for d in deltas:
                if d is not None:
                    if d.shape[0] < num_items and num_items % d.shape[0] == 0:
                        repeats = num_items // d.shape[0]
                        if self.mode == 1:
                            d_br = d.repeat_interleave(repeats, dim=0)
                        else:
                            d_br = np.repeat(d, repeats, axis=0)
                        cicF = self._process_layer_causation(d_br, cicF)
                    else:
                        cicF = self._process_layer_causation(d, cicF)
            return cicF
        elif self.ntype in [3, 5]:
            deltas = (self.generateSynapsesNodeGroupAll_MoE() if self.ntype == 5
                      else self.generateSynapsesNodeGroupAll_Transformer())
            for d in deltas:
                if d is not None:
                    cicF = self._process_layer_causation(d, cicF)
            return cicF
        else:
            sl_ih = self.generateSynapsesNodeGroupIH()
            cicF = self._process_layer_causation(sl_ih, cicF)
            last_delta = sl_ih
            for i in range(1, len(self.hiddenN)):
                sl_hh = self.generateSynapsesNodeGroupHH_Sequential(last_delta, i)
                cicF = self._process_layer_causation(sl_hh, cicF)
                last_delta = sl_hh if sl_hh is not None else last_delta
            sl_ho = self.generateSynapsesNodeGroupHO(last_delta)
            cicF = self._process_layer_causation(sl_ho, cicF)
            return cicF

    def caseItemCausality(self, A=0.0, S=0.0, F=0.0, l=0):
        τ2 = 1 if ((S >= F) and (S != 0.0) and (F != 0.0)) else 0
        τ2Str = "required information" if (τ2 == 1) else "non required information"
        iStr = "validated to full network" if (l == 1) else "invalid in the full network"
        Ф = (τ2, f"{τ2Str} {iStr} with a relevence of:{A}")
        return Ф

    # def generateSynapsesLoggerBasedSC(self, refCaseType=""):
    #     cicФ = self.caseCausationAbstration()
    #     objSC = SequentialCollection()
    #     cid = objSC.createNewCase()
    #     objCase = objSC.getCase(cid)
    #     objReg = self.selectRegresstion()
    #
    #     loop_len = min(len(cicФ), len(self.cItems)) if (cicФ and self.cItems) else 0
    #
    #     Γ, Γi, Γa, Γc, S, W, Wv, t, Vmin, Vmax, I, c, P = [], [], [], [], [], [], [], [], [], [], [], [], []
    #
    #     for j in range(loop_len):
    #         objc = cicФ[j]
    #         rval = float(objc[1])
    #         S.append(j);
    #         Γi.append(objc[0]);
    #         Γc.append("")
    #         h_val = float(objc[5][0]) if (objc[5] and len(objc[5]) > 0) else 0.0
    #         Γ.append(self.caseItemCausality(rval, float(objc[2]), float(objc[3]), (1 if h_val > 0 else 0)))
    #         Γa.append([float(x) for x in objc[5]])
    #         W.append(str(self.vInputs[j]));
    #         I.append(self.cItems[j]);
    #         Wv.append(rval)
    #         t.append(1);
    #         Vmin.append(0.0);
    #         Vmax.append(0.0);
    #         c.append(abs(rval) != 0);
    #         P.append(round(float(objc[2]), 8))
    #
    #     objCase.updateCase(refV=refCaseType, w=float(abs(objReg[-1][1])) if objReg else 0.0, Γ=Γ, Γi=Γi, Γa=Γa, Γc=Γc,
    #                        S=S, W=W, Wv=Wv, t=t, Vmin=Vmin, Vmax=Vmax, I=I, c=c, P=P)
    #     objSC.setCase(cid, objCase)
    #     return objSC

    def generateSynapsesLoggerBasedSC(self, refCaseType="", payloadId=""):
        cicФ = self.caseCausationAbstration()
        objSC = SequentialCollection()
        cid = objSC.createNewCase()
        objCase = objSC.getCase(cid)

        # Get Regression info safely
        objReg = self.selectRegresstion()

        # Safe access to weight w
        if objReg and len(objReg) > 0:
            # Check if objReg contains tuples/lists and has elements
            last_reg = objReg[-1]
            if isinstance(last_reg, (list, tuple, np.ndarray)) and len(last_reg) > 1:
                w_val = float(abs(last_reg[1]))
            else:
                w_val = 0.0
        else:
            w_val = 0.0

        loop_len = min(len(cicФ), len(self.cItems)) if (cicФ and self.cItems) else 0

        Γ, Γi, Γa, Γc, S, W, Wv, t, Vmin, Vmax, I, c, P = [], [], [], [], [], [], [], [], [], [], [], [], []

        for j in range(loop_len):
            objc = cicФ[j]
            rval = float(objc[1])
            S.append(j)
            Γi.append(objc[0])
            Γc.append("")
            h_val = float(objc[5][0]) if (objc[5] and len(objc[5]) > 0) else 0.0
            Γ.append(self.caseItemCausality(rval, float(objc[2]), float(objc[3]), (1 if h_val > 0 else 0)))
            Γa.append([float(x) for x in objc[5]])

            # Safe vInputs access
            w_str = str(self.vInputs[j]) if (self.vInputs and j < len(self.vInputs)) else ""
            W.append(w_str)

            I.append(self.cItems[j])
            Wv.append(rval)
            t.append(1)
            Vmin.append(0.0)
            Vmax.append(0.0)
            c.append(abs(rval) != 0)
            P.append(round(float(objc[2]), 8))

        if self.ntype in [0, 4, 6]:
            W.append(str(self.vInputs[len(self.vInputs) - 1]))

        objCase.updateCase(refV=refCaseType, w=w_val, Γ=Γ, Γi=Γi, Γa=Γa, Γc=Γc,
                           S=S, W=W, Wv=Wv, t=t, Vmin=Vmin, Vmax=Vmax, I=I, c=c, P=P, payloadId=payloadId)
        objSC.setCase(cid, objCase)
        return objSC