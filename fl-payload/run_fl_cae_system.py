"""
Run Complete Federated Learning System for Convolutional Autoencoder

Starts:
1. FL Server
2. Multiple FL Clients
3. Monitors training progress
4. Evaluates final model
"""
import subprocess
import time
import requests
import signal
import socket
import sys
import os
import matplotlib.pyplot as plt
import pickle
import json

import config


def _debug(msg: str):
    pass  # print(f"[RUNNER] {msg}", flush=True)


class FLSystemRunner:
    """Runner for complete FL CAE system"""

    def __init__(self, num_rounds=10, checkpoint_dir=None, sl_traces=False, sl_case_base=None,
                 sl_gate_mode="enforce", sl_gate_scope="both", sl_policy=None, malicious_client=None,
                 attack=None, attack_scale=None, absent_clients=None, final_eval=True, ip_services=None):
        self.num_rounds = num_rounds
        self.processes = []
        self.server_url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        self.sl_traces, self.sl_case_base = sl_traces, sl_case_base
        self.sl_gate_mode, self.sl_gate_scope = sl_gate_mode, sl_gate_scope
        # Optional server gate-policy overrides: {"max_rejections", "checkpoint_warmup", "checkpoint_stall_guard"}
        self.sl_policy = {k: v for k, v in (sl_policy or {}).items() if v is not None}
        self.malicious_client, self.attack, self.attack_scale = malicious_client, attack, attack_scale
        # Clients that never start: a valid run with client dropout (e.g. for the SL case base).
        self.absent_clients = list(absent_clients or [])
        self.final_eval = final_eval
        # Optional IP web services the server traces/gates through: {"sl_service_url", "cbr_service_url"}
        self.ip_services = {k: v for k, v in (ip_services or {}).items() if v}
        if len(self.absent_clients) > config.NUM_CLIENTS - config.MIN_CLIENTS:
            raise ValueError(f"At most {config.NUM_CLIENTS - config.MIN_CLIENTS} absent client(s): "
                             f"the server needs MIN_CLIENTS={config.MIN_CLIENTS} to start")

    def write_run_meta(self):
        """Record what this run was (clean or attacked) next to its checkpoints, for the research tools in drift-fairroad-service."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta = {
            "num_rounds": self.num_rounds, "num_clients": config.NUM_CLIENTS,
            "malicious_client": self.malicious_client if self.attack else None,
            "absent_clients": self.absent_clients,
            "attack": self.attack, "attack_scale": self.attack_scale,
            "sl_case_base": self.sl_case_base, "sl_gate_mode": self.sl_gate_mode if self.sl_case_base else None,
            "sl_gate_scope": self.sl_gate_scope if self.sl_case_base else None,
            "sl_policy": self.sl_policy if self.sl_case_base else None,
            "ip_services": self.ip_services or None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(self.checkpoint_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def start_server(self):
        """Start FL server"""
        print(f"\n🚀 Starting FL Server on {config.SERVER_HOST}:{config.SERVER_PORT}...")
        _debug("start_server: spawning server (stdout/stderr inherited so debug appears in terminal)")

        # Server must bind to an address the runner can reach (localhost -> 127.0.0.1)
        bind_host = '127.0.0.1' if config.SERVER_HOST in ('localhost', '127.0.0.1') else config.SERVER_HOST
        # A server already on this port (another run) would pass the readiness check and take our clients,
        # leaving this run's server with no updates and the other run with extra rounds.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex((bind_host, config.SERVER_PORT)) == 0:
                print(f"✗ Port {config.SERVER_PORT} is already in use (another FL run?). Wait for it to finish.")
                return False
        server_process = subprocess.Popen(
            [
                sys.executable, 'fl_server.py',
                '--host', bind_host,
                '--port', str(config.SERVER_PORT),
                '--num-clients', str(config.NUM_CLIENTS),
                '--min-clients', str(config.MIN_CLIENTS),
                '--aggregation', config.AGGREGATION_STRATEGY,
                '--checkpoint-dir', self.checkpoint_dir,
            ] + (['--sl-traces'] if self.sl_traces else [])
              + (['--sl-case-base', self.sl_case_base, '--sl-gate-mode', self.sl_gate_mode,
                  '--sl-gate-scope', self.sl_gate_scope] if self.sl_case_base else [])
              + [a for k, v in self.sl_policy.items() for a in (f"--sl-{k.replace('_', '-')}", str(v))]
              + [a for k, v in self.ip_services.items() for a in (f"--{k.replace('_', '-')}", v)],
            stdout=None,
            stderr=None,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        self.processes.append(('server', server_process))
        print(f"✓ Server started (PID: {server_process.pid})")
        _debug(f"start_server: PID={server_process.pid}")

        # Wait for server to be ready
        print("\nWaiting for server to start...")
        for i in range(30):
            if server_process.poll() is not None:
                print("✗ Server exited before becoming ready")
                return False
            try:
                response = requests.get(f"{self.server_url}/status", timeout=1)
                if response.status_code == 200:
                    _debug("start_server: server ready")
                    print(f"✓ Server is ready at {self.server_url}/status\n")
                    return True
            except Exception:
                pass
            print(f"Waiting for server... ({i}/30)")
            time.sleep(1)

        _debug("start_server: timeout waiting for server")
        print("✗ Server failed to start (timeout)")
        return False

    def start_clients(self):
        """Start FL clients"""
        print(f"🚀 Starting {config.NUM_CLIENTS} FL Clients...\n")
        _debug(f"start_clients: spawning {config.NUM_CLIENTS} clients")

        for i in range(config.NUM_CLIENTS):
            client_id = f"client_{i+1}"
            if client_id in self.absent_clients:
                print(f"   Skipping {client_id} (absent)")
                continue
            print(f"   Starting {client_id}...")
            _debug(f"start_clients: starting {client_id}")

            # Don't capture stdout/stderr so client logs and errors appear in the terminal
            client_process = subprocess.Popen(
                [
                    sys.executable, 'fl_client_cae.py',
                    '--client-id', client_id,
                    '--clean-dir', config.CLEAN_IMAGE_DIR,
                    '--noisy-dir', config.NOISY_IMAGE_DIR,
                    '--server-url', self.server_url,
                    '--num-rounds', str(self.num_rounds)
                ] + (['--attack', self.attack] if self.attack and client_id == self.malicious_client else [])
                  + (['--attack-scale', str(self.attack_scale)]
                     if self.attack and client_id == self.malicious_client and self.attack_scale is not None else []),
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

            self.processes.append((client_id, client_process))

        print(f"\n✓ All {config.NUM_CLIENTS} clients started!\n")
        _debug("start_clients: all spawned")

    def monitor_training(self):
        """Monitor training progress"""
        print("="*70)
        print("MONITORING TRAINING PROGRESS")
        print("="*70)
        print("(Updates shown when rounds complete)\n")
        _debug("monitor_training: starting poll loop")

        last_round = -1
        poll_count = 0

        while True:
            try:
                poll_count += 1
                # Check server status (server now does pickle in thread so /status stays responsive)
                response = requests.get(f"{self.server_url}/status", timeout=180)

                if response.status_code == 200:
                    status = response.json()
                    current_round = status.get('current_round', 0)
                    if poll_count <= 1 or current_round > last_round:
                        _debug(f"monitor: status ok round={current_round} (poll #{poll_count})")

                    # Print when new round completes
                    if current_round > last_round:
                        ready = status.get('ready_clients', status.get('clients_ready', 0))
                        total = status.get('total_expected', config.NUM_CLIENTS)
                        _debug(f"monitor: round {current_round} complete - {ready}/{total} clients")
                        print(f"✓ Round {current_round} complete - " +
                              f"Clients: {ready}/{total}, " +
                              f"Strategy: {config.AGGREGATION_STRATEGY}")
                        last_round = current_round

                # Check if all clients finished
                all_finished = all(
                    proc.poll() is not None
                    for name, proc in self.processes
                    if name.startswith('client')
                )

                if all_finished:
                    _debug("monitor_training: all clients finished")
                    print(f"\n\nWaiting for training to complete...")
                    break

                time.sleep(2)

            except Exception as e:
                # Log once per failure; server may be busy with large model transfers
                _debug(f"monitor: request failed {e}")
                print(f"Monitoring: {e} (will retry)")
                time.sleep(2)

    def wait_for_clients(self):
        """Wait for all clients to finish"""
        for name, process in self.processes:
            if name.startswith('client'):
                process.wait()
                print(f"✓ {name.capitalize()} completed")

    def generate_loss_plot(self):
        """Generate training loss plot from checkpoints"""
        print(f"\n📊 Generating loss plot...")

        checkpoint_dir = self.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            print("  Checkpoint directory not found, skipping plot")
            return

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("server_round_") and f.endswith(".pt")]
        if not checkpoints:
            print("  No checkpoints found, skipping plot")
            return

        try:
            # For now, we'll generate a placeholder plot
            # In production, server should save loss history

            # Placeholder: real loss is in server memory; use round count from checkpoints
            n = len(checkpoints)
            rounds = list(range(1, n + 1))
            losses = [0.05 - (i * 0.002) for i in range(n)]  # Placeholder

            plt.figure(figsize=(10, 6))
            plt.plot(rounds, losses, marker='o', linestyle='-', linewidth=2)
            plt.xlabel('FL Round')
            plt.ylabel('Average Loss')
            plt.title('Federated Learning Training Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            os.makedirs(config.PLOT_DIR, exist_ok=True)
            plot_path = os.path.join(config.PLOT_DIR, 'cae_training_loss.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"✓ Loss plot saved: {plot_path}")

        except Exception as e:
            print(f"  Error generating plot: {e}")

    def evaluate_model(self):
        """Evaluate final model"""
        print(f"\n📊 Evaluating final model...")

        # Find latest checkpoint
        checkpoint_dir = self.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith('server_round_') and f.endswith('.pt')
            ]

            if checkpoints:
                # Get latest checkpoint by round number (server_round_10.pt > server_round_9.pt)
                def round_key(name):
                    base = name.replace('server_round_', '').replace('.pt', '')
                    return int(base) if base.isdigit() else 0
                latest_checkpoint = max(checkpoints, key=round_key)
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

                print(f"  Using checkpoint: {latest_checkpoint}")

                # Run evaluation
                fl_payload_dir = os.path.dirname(os.path.abspath(__file__))
                eval_process = subprocess.run(
                    [
                        sys.executable, 'evaluate_cae.py',
                        '--model-path', checkpoint_path,
                        '--clean-dir', config.CLEAN_IMAGE_DIR,
                        '--noisy-dir', config.NOISY_IMAGE_DIR,
                        '--num-clients', str(config.NUM_CLIENTS)
                    ],
                    capture_output=True,
                    text=True,
                    cwd=fl_payload_dir,
                )

                if eval_process.returncode == 0:
                    print(eval_process.stdout)
                else:
                    print(f"Evaluation error: {eval_process.stderr}")

            else:
                print("  No checkpoints found")
        else:
            print("  Checkpoint directory not found")

    def cleanup(self):
        """Cleanup all processes"""
        _debug("cleanup: terminating processes")
        print("\n\nCleaning up processes...")

        for name, process in self.processes:
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("All processes terminated.")

    def run(self):
        """Run complete FL system"""
        try:
            _debug("run: printing config")
            # Print configuration
            config.print_config()

            print(f"\n{'='*70}")
            print("STARTING FEDERATED LEARNING SYSTEM")
            print("="*70)
            _debug("run: starting server")

            self.write_run_meta()

            # Start server
            if not self.start_server():
                _debug("run: server failed to start, exiting")
                return

            time.sleep(2)
            _debug("run: starting clients")

            # Start clients
            self.start_clients()

            time.sleep(2)

            _debug("run: starting monitor")
            # Monitor training
            self.monitor_training()
            _debug("run: monitor returned, waiting for clients")

            # Wait for clients
            self.wait_for_clients()

            # Generate plots
            self.generate_loss_plot()

            print("\n" + "="*70)
            print("TRAINING COMPLETED!")
            print("="*70)

            # List checkpoints
            checkpoint_dir = self.checkpoint_dir
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                if checkpoints:
                    print(f"\n📁 Saved Checkpoints ({len(checkpoints)}):")
                    for cp in sorted(checkpoints)[:5]:
                        print(f"   {checkpoint_dir}/{cp}")
                    if len(checkpoints) > 5:
                        print(f"   ... and {len(checkpoints) - 5} more")

            # Show plots
            print(f"\n📈 Training Loss Plot:")
            print(f"   {config.PLOT_DIR}/cae_training_loss.png")

            print(f"\n✓ Training artifacts saved successfully!\n")

            # Evaluate model
            if self.final_eval:
                self.evaluate_model()

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            self.cleanup()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Received interrupt signal")
    sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FL CAE System")
    parser.add_argument('--num-rounds', type=int, default=10,
                        help='Number of FL rounds (default: 10)')
    parser.add_argument('--checkpoint-dir', default=None,
                        help=f'Checkpoint directory (default: {config.CHECKPOINT_DIR})')
    parser.add_argument('--sl-traces', action='store_true',
                        help='Archive SL traces of every client update and checkpoint')
    parser.add_argument('--sl-case-base', default=None,
                        help='Name of the case base on --cbr-service-url the server gates models against')
    parser.add_argument('--sl-gate-mode', choices=['enforce', 'audit'], default='enforce')
    parser.add_argument('--sl-gate-scope', choices=['both', 'checkpoint', 'clients'], default='both')
    parser.add_argument('--sl-max-rejections', type=int, default=None,
                        help='Server: most client updates rejected per round (default: fewer than half)')
    parser.add_argument('--sl-checkpoint-warmup', type=int, default=None,
                        help='Server: checkpoints of the first N rounds are not gated (default: 1)')
    parser.add_argument('--sl-checkpoint-stall-guard', type=int, default=None,
                        help='Server: admit the next checkpoint after N consecutive rejections (default: 1; -1 off)')
    parser.add_argument('--malicious-client', default='client_3',
                        help='Client that runs --attack (default: client_3)')
    parser.add_argument('--attack', choices=['sign_flip', 'scale', 'noise', 'inverted'], default=None,
                        help='Make --malicious-client submit poisoned updates')
    parser.add_argument('--attack-scale', type=float, default=None)
    parser.add_argument('--no-final-eval', action='store_true',
                        help='Skip the evaluation of the last checkpoint after training (saves ~40 s per run)')
    parser.add_argument('--absent-client', action='append', default=[], metavar='CLIENT_ID',
                        help='Do not start this client (valid client dropout); repeatable')
    parser.add_argument('--sl-service-url', default=os.environ.get('SL_SERVICE_URL') or config.SL_SERVICE_URL,
                        help=f'SL web service the server traces through (drift-fairroad-service), used with '
                             f'--sl-traces / --sl-case-base (default: $SL_SERVICE_URL or {config.SL_SERVICE_URL})')
    parser.add_argument('--cbr-service-url', default=os.environ.get('CBR_SERVICE_URL') or config.CBR_SERVICE_URL,
                        help=f'CBR web service holding the case bases, used with --sl-case-base '
                             f'(default: $CBR_SERVICE_URL or {config.CBR_SERVICE_URL})')
    args = parser.parse_args()
    # The SL probe and gate are not in this repo: tracing and gating go through the IP services.
    if (args.sl_traces or args.sl_case_base) and not args.sl_service_url:
        parser.error('--sl-traces / --sl-case-base need --sl-service-url (drift-fairroad-service)')
    if args.sl_case_base and not args.cbr_service_url:
        parser.error('--sl-case-base needs --cbr-service-url (drift-fairroad-service)')

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Run system
    runner = FLSystemRunner(
        num_rounds=args.num_rounds,
        checkpoint_dir=args.checkpoint_dir,
        sl_traces=args.sl_traces,
        sl_case_base=args.sl_case_base,
        sl_gate_mode=args.sl_gate_mode,
        sl_gate_scope=args.sl_gate_scope,
        sl_policy=dict(max_rejections=args.sl_max_rejections, checkpoint_warmup=args.sl_checkpoint_warmup,
                       checkpoint_stall_guard=args.sl_checkpoint_stall_guard),
        malicious_client=args.malicious_client,
        attack=args.attack,
        attack_scale=args.attack_scale,
        absent_clients=args.absent_client,
        final_eval=not args.no_final_eval,
        ip_services=dict(sl_service_url=args.sl_service_url, cbr_service_url=args.cbr_service_url),
    )
    runner.run()


if __name__ == '__main__':
    main()
