# SL gate demo

A 10-minute live demo: a poisoned client attacks federated training, the SL gate rejects its updates, and
the gate explains every decision in a web page.

## Before the meeting

From `fl-payload/`:

```
.venv-gpu/Scripts/python demo_sl_gate.py --no-open
```

This takes about 4 minutes on the GPU and proves everything works. It also leaves a finished page in
`sl_runs_demo/sl_gate_demo.html` that you can fall back on if anything goes wrong live. The page is a
single file that works offline, so you can copy it anywhere.

Checklist:

- No other FL run is active (the demo refuses to start if port 5544 is busy).
- `sl_runs_gpu/case_base_v2.npz` exists (the deployed v6 case base).
- The terminal font is large enough for the room.

## Running it live

```
.venv-gpu/Scripts/python demo_sl_gate.py --pause
```

`--pause` waits for Enter before each step so you can talk. If you are short of time, skip training and
present the last runs instead:

```
.venv-gpu/Scripts/python demo_sl_gate.py --reuse
```

## Talk track

**Opening (1 min).** Federated learning trains one model across several parties without sharing their
data: each client trains locally and the server averages their updates. The weakness is that the server
cannot see the data, so it has to trust every update. One poisoned client can damage the shared model.
The SL gate checks each update before it is averaged and explains every decision it makes.

**Step 1: without the gate (about 1 min of training).** Three clients train an image-denoising
autoencoder. Client 3 is malicious and scales its updates by 10 before sending them. Point at the line
`! malicious client: attack=scale` and at the training loss as rounds complete. Every update is
averaged in, including the attacker's.

**Step 2: with the gate (about 1 min).** Same clients, same attack. Point at the gate lines as they
appear, one per client per round:

- Honest clients come in around `0.5x threshold`: accepted.
- Client 3 comes in above the threshold with low coverage: `x ... REJECTED`, in every round. Some rounds
  are just over (1.5 to 3 times), others far beyond (tens to thousands of times).

The message: the gate does not need examples of attacks. It only knows what valid models look like.

Exact numbers change from run to run (random initialisation and timing), so read them off the screen
rather than quoting them from these notes.

**Step 3: quality (about 1 min).** The table shows reconstruction error per round on a fixed
validation set. Without the gate the error stalls around 0.055 and spikes in rounds where the
attacker's update dominates the average; with the gate it keeps falling, to about 0.031. That is about
40% lower, and it is the cost of the attack that the gate prevents.

**Step 4: the page (4-5 min).** The browser opens `sl_gate_demo.html`.

1. **Headline tiles:** attacker updates rejected 10/10, honest updates accepted 20/20, final error with
   and without the gate, and how many checkpoints of the run without the gate it would have flagged.
2. **Reconstruction error chart:** the solid line (with gate) drops well below the dashed line
   (without). Hover any round for exact values.
3. **Gate decisions chart:** every model, every round, as a multiple of the threshold on a log scale.
   Honest clients sit in the grey valid zone; client 3's crosses sit above the threshold line. Click
   the highest cross.
4. **Explanation panel:** this is the case-based reasoning part. Read the paragraph aloud. It says:
   - which valid models the update was compared with (about 100-150 models from the neighbouring rounds)
   - how many times further from them it is than the threshold allows
   - what share of its trace features are inside the valid range
   - its closest valid precedent, and that even that one is far away.

   The feature chart shows why. In a scaling attack it is the bottleneck (fully connected) activations:
   valid models sit between roughly -0.5 and 1.9; the attacker sits far to the right.
   **Checkpoints with and without the gate** (the chart above the explanation). Expect the question
   "why are no checkpoints rejected?". In this setup the gate checks client updates only; checkpoints
   are traced, not gated, because gating them caused false alarms and training freezes in earlier versions. With the
   attacker removed before averaging, every checkpoint is clean (solid line, in the valid zone). The
   dashed line is the run without the gate, scored against the same case base: once the attacker is
   averaged in, the gate would have flagged nearly every checkpoint. Click a red point on the dashed
   line to see why, then "Checkpoint (with gate)" under "Same round" to compare.
5. **Contrast:** under "Same round", click Client 1. The honest update is at about half the threshold,
   with over 85% coverage and at most one feature at the edge of its range. Use the left and right
   arrow keys to step through decisions.
6. **Borderline cases:** in the table, under "Flagged", pick the round with the smallest distance
   (around 1.5 to 2 times the threshold). The paragraph says "just over the limit" and the deviations
   are smaller. The explanation scales with the evidence.
7. **Reconstructions:** the same noisy input, decoded by the final model without and with the gate,
   next to the clean target.

**Close (30 s).** Three steps: trace each update on fixed probe images, retrieve the closest valid
precedents, decide against a threshold calibrated for a 0.5% false-alarm rate. Across 270 honest
updates in the evaluation runs, 1 was wrongly rejected; scaling attacks are caught almost every time.

## Likely questions

- **Does the server need the clients' data?** No. It runs every update on the same 32 probe images
  that it holds itself.
- **What does the gate know about attacks?** Nothing. The case base holds only valid models from 14
  clean training runs (8 with all clients, 6 with one client absent). That is why it can say "unlike any
  valid model" but cannot name the attack.
- **Which attacks does it miss?** Weak ones that barely change the model: Gaussian noise at 0.5 and
  inverted training mostly pass. They also barely hurt the model. Sign flip is the weak spot: 23 of 30
  updates caught when replayed offline, but only 9 of 30 in live runs.
- **How expensive is it?** Tracing one model takes about 12 ms on the GPU and 65 ms on the CPU.
- **Can it be fooled by an attacker who knows the case base?** Not tested here: an adaptive attacker
  that stays inside the valid range is open work.

## If something goes wrong

- **"Port 5544 is in use":** another run is still going. Wait for it to finish, or run with `--reuse`.
- **Training fails:** the full output is in `sl_runs_demo/gated.log` (or `ungated.log`).
- **No GPU:** the demo still runs on the CPU, more slowly.
- **Anything else:** open the page you built before the meeting.
