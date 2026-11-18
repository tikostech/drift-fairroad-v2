"""
Federated Learning Server

Coordinates training and aggregates model updates.
Pickle serialize/deserialize runs in a thread executor so the event loop
stays responsive for /status and other requests (avoids monitor timeouts).

SL gate (optional; the gate itself is IP and lives in the drift-fairroad-service repo): each client
update and each averaged checkpoint is traced over a fixed probe set. With --sl-traces the traces are archived; with --sl-case-base they are
also compared against a case base of valid models. In enforce mode an invalid client update is
left out of the average, and an invalid checkpoint is saved as server_round_N.rejected.pt while
the previous global model is kept (and saved as server_round_N.pt).

A round's client updates are decided together once all have arrived: at most --sl-max-rejections
(default: fewer than half the clients) are rejected, most anomalous first. Rejecting more would
mean a majority of attackers or, far more likely, drift that the gate itself caused -- rejecting
everything then freezes training. The first --sl-checkpoint-warmup round(s) of checkpoints are
traced but not gated: a round-1 average of independently initialised networks is erratic. After
--sl-checkpoint-stall-guard consecutive rejected checkpoints the next one is admitted, for the same
reason as the budget: repeated rejection without progress is drift, and would stall training.

The SL tracing and the case base run as web services (drift-fairroad-service, reached through
ip_client/, which holds no IP): --sl-service-url traces through the SL service, --cbr-service-url
gates against a case base held by the CBR service (--sl-case-base is then its name there). Without
them they run in-process, which needs the service repo's IP core checked out (see ip_core.py). If a service call fails mid-run, the model is admitted
ungated and the error is logged and recorded (fail-open): a gate outage must not stall training.
"""
import asyncio
import json
import pickle
import time
import torch
from aiohttp import web
import logging
import importlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from aggregation import FederatedAggregator
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def _debug(msg: str):
    pass  # print(f"[FL-SERVER] {msg}", flush=True)


class FLServer:
    """Federated Learning Server."""

    def __init__(
        self,
        host: str,
        port: int,
        num_clients: int,
        min_clients: int,
        aggregation_strategy: str,
        checkpoint_dir: str = None,
        sl_traces: bool = False,
        sl_case_base: str = None,
        sl_gate_mode: str = "enforce",
        sl_gate_scope: str = "both",
        sl_max_rejections: int = None,
        sl_checkpoint_warmup: int = 1,
        sl_checkpoint_stall_guard: int = 1,
        sl_service_url: str = None,
        cbr_service_url: str = None,
        ip_api_key: str = None,
    ):
        """Initialize FL Server.

        sl_gate_mode: "enforce" acts on invalid verdicts, "audit" only records them.
        sl_gate_scope: "both", "checkpoint" or "clients" -- which models are gated (all are traced).
        sl_max_rejections: most client updates rejected per round (default (num_clients - 1) // 2,
            the usual Byzantine bound of fewer than half the clients malicious). The most anomalous
            invalid updates are rejected first; any further invalid ones are admitted.
        sl_checkpoint_warmup: checkpoints of the first N rounds are traced but not gated.
        sl_checkpoint_stall_guard: after N consecutive rejected checkpoints the next one is admitted
            (logged). Repeated rejection at the same training progress is drift, not a fresh attack,
            and rejecting on would stall training. A negative value disables the guard.
        sl_service_url: trace through the SL web service instead of in-process.
        cbr_service_url: gate through the CBR web service; ``sl_case_base`` is then a case base name there.
        ip_api_key: API key for both services (default: $IP_SERVICE_API_KEY).
        """
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.min_clients = min_clients
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR

        # Load model dynamically from config
        logger.info(f"Loading model from {config.MODEL_PATH}.{config.MODEL_CLASS}")
        model_module = importlib.import_module(config.MODEL_PATH)
        model_class = getattr(model_module, config.MODEL_CLASS)

        # Initialize model
        self.model = model_class(**config.MODEL_CONFIG)
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        print(f"Model parameters: {self.model.count_parameters()}")
        print(f"Model: {self.model}")

        # Aggregator
        self.aggregator = FederatedAggregator(strategy=aggregation_strategy)

        # State
        self.current_round = 0
        self.client_weights = {}
        self.client_samples = {}
        self.client_losses = {}
        self.registered_clients = set()
        self.ready_clients = set()
        self.round_lock = asyncio.Lock()
        self._round_event = asyncio.Event()  # set (and replaced) each time a round completes

        # Loss tracking for plotting
        self.loss_history = {}  # {client_id: [losses per round]}
        self.round_losses = []  # Average loss per round

        # Flag to track if training has started
        self.training_started = False

        # Executor for CPU-bound pickle so event loop can still serve /status etc.
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.max_rejections = (num_clients - 1) // 2 if sl_max_rejections is None else sl_max_rejections
        self.checkpoint_warmup = sl_checkpoint_warmup
        self.checkpoint_stall_guard = sl_checkpoint_stall_guard
        self.ckpt_rejected_in_a_row = 0
        self.sl_service_url, self.cbr_service_url, self.ip_api_key = sl_service_url, cbr_service_url, ip_api_key
        self._init_sl_gate(model_class, sl_traces, sl_case_base, sl_gate_mode, sl_gate_scope)

        logger.info(f"Server initialized with {aggregation_strategy} aggregation")

    def _init_sl_gate(self, model_class, sl_traces, sl_case_base, mode, scope):
        """Set up tracing (and gating, if a case base is given). No-op when both are off."""
        self.tracer = self.trace_store = self.case_base = None
        self.gate_mode, self.gate_scope = mode, scope
        # Accepted checkpoints so far. Models are compared with valid models at the same training
        # progress (generation + 1), not the same round number: after a rejected round the next
        # round trains from the older model, and judging it by the round counter would reject it
        # too, freezing training.
        self.generation = 0
        self.pending = {}  # client_id -> traced update awaiting the round's decision (rejection budget)
        self._service_down = ()  # ip_client.ServiceError when a service is used: caught, fail-open
        if not (sl_traces or sl_case_base):
            return
        from trace_store import TraceStore
        if self.sl_service_url or self.cbr_service_url:
            # ip_client sits at the repo root locally and next to this file in the server image.
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root not in sys.path:
                sys.path.append(root)
            from ip_client import ServiceError
            self._service_down = ServiceError  # unreachable, 5xx, or a rejected call (e.g. case base removed)

        if self.sl_service_url:
            from ip_client import RemoteModelTracer, SLServiceClient
            sl = SLServiceClient(self.sl_service_url, api_key=self.ip_api_key)
            sl.wait_ready(timeout=120)
            self.tracer = RemoteModelTracer(sl)
            files, probe_hash = self.tracer.probe_files, self.tracer.probe_hash
            where = f"SL service {self.sl_service_url}"
        else:
            import ip_core  # noqa: F401  (in-process gate: needs the IP core, see ip_core.py)
            from sl_gate import ModelTracer, load_probe_images
            batch, files, probe_hash = load_probe_images(
                config.NOISY_IMAGE_DIR, config.SL_PROBE_IMAGES, config.AUTOENCODER_IMAGE_SIZE)
            # Scratch network: traced models are loaded into it, never into the global model.
            self.tracer = ModelTracer(model_class(**config.MODEL_CONFIG).model, batch, probe_hash)
            where = config.NOISY_IMAGE_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.trace_store = TraceStore(os.path.join(self.checkpoint_dir, "sl_traces.npz"), files, probe_hash)
        self.gate_log_path = os.path.join(self.checkpoint_dir, "sl_gate_log.jsonl")
        open(self.gate_log_path, "w").close()  # one log per run, like sl_traces.npz
        logger.info(f"SL tracing on: {len(files)} probe images ({where})")

        if sl_case_base:
            if self.cbr_service_url:
                from ip_client import CBRServiceClient, RemoteCaseBase
                cbr = CBRServiceClient(self.cbr_service_url, api_key=self.ip_api_key)
                cbr.wait_ready(timeout=120)
                self.case_base = RemoteCaseBase(cbr, sl_case_base)
                source = f"'{sl_case_base}' on CBR service {self.cbr_service_url}"
            else:
                import ip_core  # noqa: F401
                from sl_gate import SLCaseBase
                self.case_base = SLCaseBase.load(sl_case_base)
                source = sl_case_base
            # Trace once now so a schema or probe-set mismatch fails at startup, not mid-training.
            self.tracer.trace(self.model.get_weights())
            self.case_base.check_compatible(self.tracer.names, probe_hash)
            logger.info(f"SL gate on ({mode}, scope={scope}): {source} "
                        f"({self.case_base.meta['n_models']} valid models); at most {self.max_rejections} "
                        f"update(s) rejected per round; checkpoint warm-up {self.checkpoint_warmup} round(s)")

    def _trace_and_decide(self, weights, round_num: int, source: str, included=None) -> dict:
        """Trace one model and score it if this source is in scope. Nothing is acted on or written yet.

        Runs in the executor (CPU-bound forward pass + statistics).
        """
        t0 = time.time()
        is_global = source == "global"
        compare_round = self.generation + 1
        in_scope = self.case_base is not None and (
            self.gate_scope == "both" or self.gate_scope == ("checkpoint" if is_global else "clients"))
        Z = decision = error = None
        try:
            Z = self.tracer.trace(weights)
            decision = self.case_base.decide(Z, compare_round) if in_scope else None
        except self._service_down as e:
            # Fail-open: a failing gate service must not stall training. Recorded, and logged loudly.
            error = f"IP service error, model admitted ungated: {e}"
            logger.error(f"SL gate r{round_num}/{source}: {error}")
        record = {"model": f"r{round_num}/{source}", "round": round_num, "compare_round": compare_round,
                  "source": source, "included": included, "gated": in_scope and error is None,
                  "accepted": True, "note": None,
                  "verdict": decision["verdict"] if decision else None,
                  "score_ratio": decision["score_ratio"] if decision else None,
                  "coverage": decision["coverage"] if decision else None,
                  "trace_seconds": round(time.time() - t0, 3)}
        if error:
            record["error"] = error
        return dict(Z=Z, record=record, decision=decision,
                    invalid=decision is not None and decision["verdict"] == "invalid")

    def _commit(self, traced: dict, accepted: bool, note: str = None) -> bool:
        """Archive and log a traced model with its final decision."""
        record, decision = traced["record"], traced["decision"]
        record["accepted"], record["note"] = accepted, note
        if traced["Z"] is not None:  # None when tracing through the SL service failed
            self.trace_store.add(traced["Z"], self.tracer.names, record)
        with open(self.gate_log_path, "a") as f:
            f.write(json.dumps(dict(record, decision=decision)) + "\n")
        if decision:
            msg = (f"SL gate {record['model']}: {decision['verdict'].upper()} "
                   f"(score {decision['score_ratio']:.2f}x threshold, coverage {decision['coverage']:.3f})"
                   + ("" if accepted or not traced["invalid"] else " -> REJECTED") + (f" [{note}]" if note else ""))
            (logger.warning if traced["invalid"] else logger.info)(msg)
        return accepted

    def _apply_rejection_budget(self) -> set:
        """Decide this round's client updates together; returns the client ids rejected.

        Invalid updates are rejected most anomalous first, at most ``max_rejections`` per round.
        If more look invalid than the budget allows, the model's whole population has moved (drift,
        not a majority of attackers), and rejecting them all would stall training.
        """
        flagged = [cid for cid, t in self.pending.items() if t["invalid"]]
        if self.gate_mode != "enforce":
            rejected = set()
        else:
            flagged.sort(key=lambda cid: self.pending[cid]["decision"]["score_ratio"], reverse=True)
            rejected = set(flagged[:self.max_rejections])
        for cid, traced in sorted(self.pending.items()):
            note = None
            if traced["invalid"] and cid not in rejected and self.gate_mode == "enforce":
                note = f"admitted: rejection budget ({self.max_rejections} per round) used by more anomalous updates"
            self._commit(traced, cid not in rejected, note)
        self.pending = {}
        return rejected

    async def handle_register(self, request: web.Request) -> web.Response:
        """Handle client registration."""
        data = await request.json()
        client_id = data.get("client_id")
        _debug(f"register request from {client_id}")

        async with self.round_lock:
            if client_id not in self.registered_clients:
                self.registered_clients.add(client_id)
                self.loss_history[client_id] = []  # Initialize loss history
                logger.info(f"{client_id} registered ({len(self.registered_clients)}/{self.num_clients})")

            # Check if minimum clients reached and training should start
            if not self.training_started and len(self.registered_clients) >= self.min_clients:
                self.training_started = True
                logger.info(f"✓ Minimum {self.min_clients} clients registered. Training can start!")

        _debug(f"register done: {client_id} -> {len(self.registered_clients)}/{self.num_clients}, can_start={self.training_started}")
        return web.json_response({
            "status": "registered",
            "current_round": self.current_round,
            "can_start": self.training_started,
            "message": f"{len(self.registered_clients)}/{self.num_clients} clients registered"
        })

    async def handle_get_model(self, request: web.Request) -> web.Response:
        """Send global model to client."""
        data = await request.json()
        client_id = data.get("client_id")
        requested_round = data.get("round", 0)
        _debug(f"get_model: {client_id} requests round {requested_round} (current={self.current_round})")

        if client_id not in self.registered_clients:
            return web.json_response({"error": "Not registered"}, status=403)

        # Wait for training to start (minimum clients)
        while not self.training_started:
            await asyncio.sleep(0.5)

        # Wait for the requested round to be ready (aggregation completed): woken by the round event
        # rather than polling, so the next round starts as soon as the checkpoint exists.
        max_wait = 300  # 5 minutes (clients may be staggered)
        deadline = asyncio.get_running_loop().time() + max_wait
        while self.current_round < requested_round:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            event = self._round_event
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break

        if self.current_round < requested_round:
            logger.warning(f"{client_id} timed out waiting for round {requested_round} (current: {self.current_round})")
            return web.json_response(
                {"error": "Round not ready", "current_round": self.current_round},
                status=503
            )

        # Read model under lock so we don't read during set_weights in aggregation
        _debug(f"get_model: acquiring lock for {client_id}")
        async with self.round_lock:
            weights = self.model.get_weights()
        _debug(f"get_model: pickling for {client_id} (executor)")
        loop = asyncio.get_running_loop()
        weights_bytes = await loop.run_in_executor(
            self._executor, lambda: pickle.dumps(weights)
        )
        _debug(f"get_model: sending {len(weights_bytes)} bytes to {client_id}")
        return web.Response(
            body=weights_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Round": str(self.current_round)
            }
        )

    async def handle_submit_update(self, request: web.Request) -> web.Response:
        """Receive model updates."""
        client_id = request.headers.get("X-Client-ID")
        num_samples = int(request.headers.get("X-Num-Samples", 0))
        loss = float(request.headers.get("X-Loss", 0.0))
        _debug(f"submit_update: {client_id} body_size=reading...")

        if client_id not in self.registered_clients:
            return web.json_response({"error": "Invalid client"}, status=403)

        weights_bytes = await request.read()
        _debug(f"submit_update: {client_id} read {len(weights_bytes)} bytes, unpickling (executor)")
        loop = asyncio.get_running_loop()
        client_weights = await loop.run_in_executor(
            self._executor, lambda: pickle.loads(weights_bytes)
        )

        traced = None
        if self.tracer:
            round_num = self.current_round + 1
            traced = await loop.run_in_executor(
                self._executor, lambda: self._trace_and_decide(client_weights, round_num, client_id)
            )

        _debug(f"submit_update: {client_id} acquiring lock")
        async with self.round_lock:
            # Every update is held until the round is complete; the SL gate then decides the round's
            # updates together (rejection budget) and drops the rejected ones before averaging.
            self.client_weights[client_id] = client_weights
            self.client_samples[client_id] = num_samples
            self.client_losses[client_id] = loss
            self.ready_clients.add(client_id)
            if traced is not None:
                self.pending[client_id] = traced

            logger.info(
                f"Update from {client_id}: Loss={loss:.6f} "
                f"({len(self.ready_clients)}/{len(self.registered_clients)})"
                + (" -- flagged by SL gate" if traced is not None and traced["invalid"] else "")
            )

            # Aggregate when ALL registered clients have submitted for this round
            if len(self.ready_clients) >= len(self.registered_clients):
                _debug(f"submit_update: all {len(self.ready_clients)} submitted -> aggregating")
                # Track loss for plotting BEFORE aggregation (when all clients have submitted)
                for cid in self.client_losses:
                    if cid in self.loss_history:
                        self.loss_history[cid].append(self.client_losses[cid])

                await self._aggregate_and_update()
                _debug(f"submit_update: aggregation done, round={self.current_round}")

                return web.json_response({
                    "status": "aggregated",
                    "round": self.current_round
                })
            else:
                waiting = len(self.registered_clients) - len(self.ready_clients)
                _debug(f"submit_update: {client_id} -> waiting ({waiting} more)")
                return web.json_response({
                    "status": "waiting",
                    "round": self.current_round,
                    "waiting_for": waiting
                })

    async def _aggregate_and_update(self):
        """Aggregate and update global model. Heavy work runs in executor so event loop stays responsive.

        With the SL gate, the averaged weights are traced first and only promoted if accepted;
        otherwise the previous global model stays in force for the next round.
        """
        if not self.client_weights and not self.tracer:
            return

        try:
            loop = asyncio.get_running_loop()
            round_num = self.current_round + 1
            if self.pending:
                for cid in self._apply_rejection_budget():
                    self.client_weights.pop(cid, None)
                    self.client_samples.pop(cid, None)
                    logger.info(f"Update from {cid} excluded by SL gate")
            aggregated_weights = None
            if self.client_weights:
                weights_list = list(self.client_weights.values())
                samples_list = list(self.client_samples.values())
                _debug("_aggregate_and_update: running aggregate in executor")
                # Run aggregation in executor (CPU-heavy; blocks event loop otherwise)
                aggregated_weights = await loop.run_in_executor(
                    self._executor,
                    lambda: self.aggregator.aggregate(
                        weights_list,
                        samples_list if self.aggregator.strategy == "weighted" else None,
                    ),
                )
            else:
                logger.warning(f"Round {round_num}: every client update was rejected; keeping the global model")

            promote = aggregated_weights is not None
            if promote and self.tracer:
                included = sorted(self.client_weights)
                traced = await loop.run_in_executor(
                    self._executor,
                    lambda: self._trace_and_decide(aggregated_weights, round_num, "global", included),
                )
                # A round-1 checkpoint averages independently initialised networks, so its trace says
                # little and rejecting it would push training off every trajectory the case base knows.
                warmup = round_num <= self.checkpoint_warmup
                stalled = 0 <= self.checkpoint_stall_guard <= self.ckpt_rejected_in_a_row
                reject = traced["invalid"] and self.gate_mode == "enforce" and not warmup and not stalled
                note = None
                if traced["invalid"] and warmup:
                    note = f"warm-up: round <= {self.checkpoint_warmup} not gated"
                elif traced["invalid"] and stalled and self.gate_mode == "enforce":
                    note = (f"admitted: stall guard ({self.ckpt_rejected_in_a_row} consecutive checkpoint "
                            f"rejection(s) at this training progress)")
                promote = self._commit(traced, not reject, note)
                self.ckpt_rejected_in_a_row = 0 if promote else self.ckpt_rejected_in_a_row + 1
                if not promote:
                    await loop.run_in_executor(
                        self._executor,
                        lambda: self._save_checkpoint_sync(round_num, aggregated_weights, suffix=".rejected"),
                    )

            if promote:
                _debug("_aggregate_and_update: set_weights on main loop")
                self.model.set_weights(aggregated_weights)
                if self.tracer:
                    self.generation += 1

            avg_loss = sum(self.client_losses.values()) / len(self.client_losses)
            self.round_losses.append(avg_loss)  # Track average loss

            logger.info(
                f"✓ Round {round_num} complete - "
                f"Avg Loss: {avg_loss:.6f}, Clients: {len(self.ready_clients)}"
                + ("" if promote else " -- checkpoint rejected, previous global model kept")
            )

            self.current_round += 1
            self._round_event.set()           # wake clients waiting in /get_model
            self._round_event = asyncio.Event()
            self.client_weights.clear()
            self.client_samples.clear()
            self.client_losses.clear()
            self.ready_clients.clear()

            # Run checkpoint save in executor so I/O doesn't block the loop
            round_done = self.current_round
            state_dict = self.model.get_weights()
            _debug(f"_aggregate_and_update: saving checkpoint round {round_done} (executor)")
            await loop.run_in_executor(
                self._executor,
                lambda: self._save_checkpoint_sync(round_done, state_dict),
            )
            _debug("_aggregate_and_update: done")

        except Exception as e:
            logger.error(f"Aggregation error: {e}")

    def _save_checkpoint_sync(self, round_num: int, state_dict, suffix: str = ""):
        """Save checkpoint (sync, for use in executor)."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = f"{self.checkpoint_dir}/server_round_{round_num}{suffix}.pt"
        torch.save({"round": round_num, "model_state_dict": state_dict}, path)
        logger.info(f"Checkpoint saved: {path}")

    def save_loss_plot(self):
        """Save loss plot for all clients and average loss."""
        os.makedirs(config.PLOT_DIR, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Plot individual client losses
        for client_id, losses in self.loss_history.items():
            if losses:  # Only plot if client has loss data
                rounds = list(range(1, len(losses) + 1))
                plt.plot(rounds, losses, marker='o', label=f'{client_id}', alpha=0.7)

        # Plot average loss
        if self.round_losses:
            rounds = list(range(1, len(self.round_losses) + 1))
            plt.plot(rounds, self.round_losses, marker='s', linewidth=2,
                    label='Average', color='black', linestyle='--')

        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Federated Learning - Training Loss per Round', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = f"{config.PLOT_DIR}/training_loss.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Loss plot saved: {plot_path}")
        return plot_path

    async def handle_status(self, request: web.Request) -> web.Response:
        """Server status."""
        _debug("status: replying")
        return web.json_response({
            "current_round": self.current_round,
            "registered_clients": len(self.registered_clients),
            "ready_clients": len(self.ready_clients),
            "total_expected": self.num_clients,
            "aggregation_strategy": self.aggregator.strategy
        })

    async def handle_save_plot(self, request: web.Request) -> web.Response:
        """Generate and save loss plot."""
        try:
            plot_path = self.save_loss_plot()
            return web.json_response({
                "status": "success",
                "plot_path": plot_path
            })
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)

    def create_app(self) -> web.Application:
        """Create app."""
        # CAE model weights ~48MB; allow up to 64MB for submit_update
        app = web.Application(client_max_size=64*1024*1024)
        app.router.add_post("/register", self.handle_register)
        app.router.add_post("/get_model", self.handle_get_model)
        app.router.add_post("/submit_update", self.handle_submit_update)
        app.router.add_get("/status", self.handle_status)
        app.router.add_post("/save_plot", self.handle_save_plot)
        return app

    def run(self):
        """Start server."""
        app = self.create_app()
        logger.info(f"Starting FL Server on {self.host}:{self.port}")
        web.run_app(app, host=self.host, port=self.port, print=None)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--aggregation", default="fedavg")
    parser.add_argument("--checkpoint-dir", default=None, help=f"Checkpoint directory (default: {config.CHECKPOINT_DIR})")
    parser.add_argument("--sl-traces", action="store_true", help="Archive SL traces of every client update and checkpoint")
    parser.add_argument("--sl-case-base", default=None,
                        help="SL case base (.npz) to gate models against, or its name on --cbr-service-url; "
                             "implies --sl-traces")
    parser.add_argument("--sl-gate-mode", choices=["enforce", "audit"], default="enforce",
                        help="enforce: drop invalid updates / roll back invalid checkpoints; audit: record only")
    parser.add_argument("--sl-gate-scope", choices=["both", "checkpoint", "clients"], default="both",
                        help="Which models are gated (all are traced)")
    parser.add_argument("--sl-max-rejections", type=int, default=None,
                        help="Most client updates rejected per round (default: (num_clients - 1) // 2)")
    parser.add_argument("--sl-checkpoint-warmup", type=int, default=1,
                        help="Checkpoints of the first N rounds are traced but not gated (default: 1)")
    parser.add_argument("--sl-checkpoint-stall-guard", type=int, default=1,
                        help="Admit the next checkpoint after N consecutive rejections (default: 1; -1 disables)")
    parser.add_argument("--sl-service-url", default=os.environ.get("SL_SERVICE_URL"),
                        help="Trace through the SL web service at this URL instead of in-process ($SL_SERVICE_URL)")
    parser.add_argument("--cbr-service-url", default=os.environ.get("CBR_SERVICE_URL"),
                        help="Gate through the CBR web service at this URL ($CBR_SERVICE_URL); "
                             "--sl-case-base then names a case base held by that service")
    parser.add_argument("--ip-api-key", default=None, help="API key for the IP services (default: $IP_SERVICE_API_KEY)")
    args = parser.parse_args()

    server = FLServer(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        min_clients=args.min_clients,
        aggregation_strategy=args.aggregation,
        checkpoint_dir=args.checkpoint_dir,
        sl_traces=args.sl_traces,
        sl_case_base=args.sl_case_base,
        sl_gate_mode=args.sl_gate_mode,
        sl_gate_scope=args.sl_gate_scope,
        sl_max_rejections=args.sl_max_rejections,
        sl_checkpoint_warmup=args.sl_checkpoint_warmup,
        sl_checkpoint_stall_guard=args.sl_checkpoint_stall_guard,
        sl_service_url=args.sl_service_url,
        cbr_service_url=args.cbr_service_url,
        ip_api_key=args.ip_api_key,
    )

    server.run()


if __name__ == "__main__":
    main()
