# G1 pure-RL wrist-recovery training plan

This file is the source of truth for the staged curriculum and its promotion
criteria. The watcher must read it before proposing or launching an experiment.
Training distributions may evolve; the held-out scenario definitions, seeds,
and acceptance thresholds must not be tuned against individual checkpoints.

## Data split and evaluation rule

- Training uses randomized commands, loads, pushes, reset states, and domain
  randomization. Training returns and training-episode metrics diagnose PPO but
  do not decide whether a policy is better.
- Development evaluation may use separate fixed seeds while designing a stage.
- Final manual held-out evaluation uses `scripts/evaluate_wrist_recovery.py`, fixed
  seeds `1103, 2207, 3301`, and all six scenarios. Every candidate is compared
  with the previous retained checkpoint under the identical evaluator command.
- Report both `nominal` (task variables only) and `robust` (startup domain
  randomization retained). Never train directly on these exact scenario tuples.
- Keep a change only when the targeted held-out metric improves without a
  material regression in wrist accuracy, falls, foot slip, or action quality.
- Autonomous exploratory groups do not run the complete held-out matrix. That
  matrix takes roughly two hours and is reserved for the user's final manual
  checkpoint review. The watcher may provisionally rank candidates from
  training diagnostics, but must preserve all checkpoints and must not label a
  training-metric winner as the final policy. An evaluation runs automatically
  only when its plan explicitly sets `evaluation.automatic=true`.

## Staged training curriculum

| Stage | Training mixture and purpose | Promotion gate on held-out evaluation |
|---|---|---|
| 0. Static hold | Zero base command, world-fixed wrists; initially no push or payload. Establish quiet standing and remove meaningless wrist/body motion. | `static_hold`: no falls; success >= 95%; mean wrist position < 2 cm and rotation < 0.15 rad; low foot motion/contact switching. |
| 1. Payload | Mix static hold with randomized bilateral, unilateral, and asymmetric wrist wrench. Ramp vertical load before lateral force/torque. | `payload`: no systematic collapse; success >= 90%; position p95 < 5 cm; no large increase in slip or yaw. |
| 2. Reactive recovery | Add random impulses in longitudinal, lateral, and yaw directions. Allow stepping; do not reward a prescribed step. | `push`: fall rate <= 5%; success >= 85%; peak and final wrist errors improve over baseline; recovery does not rely on continuous shuffling. |
| 3. Proactive support | Add symmetric and asymmetric reaches with target known before motion. Mix 0.12--0.30 m extension, lateral/vertical offsets, and modest wrist rotation. | Both reach scenarios: success >= 85%; final position < 4 cm; stable support change occurs without waiting for a fall; static hold remains quiet. |
| 4. Combined robustness | Mix reach, payload, and timed push, including asymmetric cases; retain easier cases to prevent forgetting. | `combined`: fall rate <= 10%, success >= 75%; robust-suite degradation is bounded; all earlier gates remain acceptable. |
| 5. Locomotion and height | Add a masked shoulder-line height target, ground-reaching wrist targets, and then introduce base twist gradually. Permit forward waist flexion, knee flexion, or mixtures; prohibit using backward torso lean to lower the shoulders and penalize excessive shoulder height difference. | During automated exploration use wrist/height/velocity errors, falls, slip, backward lean, shoulder level, and action smoothness only as provisional diagnostics. The user performs final visual and held-out judgment. |
| 6. EgoDex/Pico trajectories | Train on mapped continuous bimanual trajectories, with held-out objects/trajectory clips and latency/noise randomization. | Generalizes to unseen clips and asymmetric manipulation; simulator-to-real safety review precedes hardware execution. |

For automated promotion, “bounded” robust degradation means: static, payload,
and push success each remain at least 85% with fall rate at most 10%; both reach
scenarios remain at least 70% successful with fall rate at most 15%; and Stage
4 combined remains at least 65% successful with fall rate at most 15%. The
stricter nominal gates in the table still apply. During independent-seed
confirmation, apply gates to the aggregate across seeds and reject any seed
whose fall rate exceeds the corresponding limit by more than 5 percentage
points. These thresholds were fixed before reading the new robust matrix.

## Mixture discipline

- Within each new stage, start roughly 50% from the previous mastered stage,
  30% current moderate cases, and 20% current hard cases; adjust from held-out
  failure categories rather than total reward.
- Ramp one conceptual source of difficulty at a time. Payload, impulse, reach,
  velocity command, observation latency, and physics randomization are separate
  axes.
- Preserve easy static episodes through all stages so wrist drift and needless
  stepping cannot be hidden by high rewards on dynamic cases.
- A scenario matrix is suitable for evaluation. Training should sample around
  its ranges continuously, not replay the exact held-out Cartesian product.

## Statistical and GPU protocol

- `env.scene.num_envs` is per process. Four-GPU DDP at 1,024 environments per
  rank is 4,096 environments globally. The historical runs used 4,096 per rank,
  or 16,384 globally.
- Use four independent one-GPU runs for cheap parameter screening. Hold the
  training seed fixed across candidates so the parameter difference is the
  primary controlled factor. This ranks candidates but is not a final claim.
- Confirm the winning configuration with at least three independent training
  seeds, preferably concurrently on separate GPUs. Report mean, standard
  deviation, and individual seed results.
- Run the identical fixed held-out evaluator on every confirmation checkpoint.
  Treat a 1--3% single-run difference as inconclusive unless it is consistent
  across training seeds or substantially exceeds held-out variation.
- Use four-GPU DDP only when its larger effective batch or throughput is itself
  required. It produces one policy, not four independent experimental samples.
- Launch registered independent groups from a reviewed JSON plan using
  `tools/remote_sweep.py`. The watcher waits for every sibling and analyzes the
  group once, avoiding four separate model calls.

## Current state and next decision

The current retained checkpoint is the 5,000-iteration `yawpen_v1_full2` policy.
The first standard nominal held-out run (1,152 episodes) passes static hold,
payload, and push, but fails the reach gates: symmetric reach succeeds in 53.1%
of episodes with 46.4% falls, asymmetric reach succeeds in 38.5% with 37.5%
falls, and combined succeeds in 40.1% with 58.3% falls. The next controlled
training work should therefore target Stage 3 reach/support diversity, while
retaining mastered Stage 0--2 cases to prevent forgetting. Establish robust
suite numbers before claiming readiness for Stage 4.

The active Stage 3 budget is one four-candidate, fixed-seed screen followed by
one independent-seed confirmation group for the promoted configuration. The
screen is defined in `experiments/stage3_screen_v1.json`: it crosses reach
probability 0.50/0.80 with asymmetric-target probability 0.00/0.50. Each run
uses one H20, 4,096 environments, seed 1201, and 5,000 PPO iterations. This is
a candidate-ranking experiment, not multi-seed evidence. Do not launch the
confirmation group unless the fixed held-out suite shows a useful improvement
without regression of the already-passed static, payload, and push gates.

Stage 3 is confirmed across four independent seeds. Stage 4 is already covered
by the concurrent reach/payload/timed-push mixture. Stage 5 is authorized with
the command interface `[planar twist, bilateral wrist poses, shoulder-line
height, height mask]`. The shoulder line is measured from the two fixed shoulder
joint anchors. Do not command torso pitch or knee angles: forward waist flexion,
knee flexion, and mixtures are all acceptable. A soft penalty begins near five
degrees of backward torso lean. During initial learning, a height-task episode
terminates only after the target trajectory is halfway complete and backward
lean still exceeds twenty-five degrees; later stages may tighten this toward
the deployment constraint. Excessive shoulder height difference uses a dead zone so
lateral gait and push recovery remain possible. Start with stationary height
control and low wrist targets, then introduce planar motion as a separate
controlled factor. Stage 6 still requires the mapped trajectory split and
interface contract.

Example:

```bash
/mnt/hdd/miniforge3/envs/unitree_rl_mjlab/bin/python \
  scripts/evaluate_wrist_recovery.py \
  --checkpoint-file logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_12-12-08/model_4999.pt
```

For watcher/remote use, choose a unique evaluation name; the wrapper runs with
the immutable remote environment and copies the result directory back locally:

```bash
./tools/remote_evaluate_wrist.sh candidate_full5_nominal \
  /home/dev/unitree_rl_mjlab/logs/rsl_rl/g1_wrist_recovery_teacher/RUN/model_4999.pt
```
