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

Latest overnight authorization: `experiments/bilateral_ground_next.md` permits
stationary bilateral-low + shoulder-height training after successful leg8x
confirmation and bilateral smoke, without another user approval. Follow its
two-group limit; it supersedes the old post-leg-confirmation stop, not its gates.

| Stage | Training mixture and purpose | Promotion gate on held-out evaluation |
|---|---|---|
| 0. Static hold | Zero base command, world-fixed wrists; initially no push or payload. Establish quiet standing and remove meaningless wrist/body motion. | `static_hold`: no falls; success >= 95%; mean wrist position < 2 cm and rotation < 0.15 rad; low foot motion/contact switching. |
| 1. Payload | Mix static hold with randomized bilateral, unilateral, and asymmetric wrist wrench. Ramp vertical load before lateral force/torque. | `payload`: no systematic collapse; success >= 90%; position p95 < 5 cm; no large increase in slip or yaw. |
| 2. Reactive recovery | Add random impulses in longitudinal, lateral, and yaw directions. Allow stepping; do not reward a prescribed step. | `push`: fall rate <= 5%; success >= 85%; peak and final wrist errors improve over baseline; recovery does not rely on continuous shuffling. |
| 3. Proactive support | Add symmetric and asymmetric reaches with target known before motion. Mix 0.12--0.30 m extension, lateral/vertical offsets, and modest wrist rotation. | Both reach scenarios: success >= 85%; final position < 4 cm; stable support change occurs without waiting for a fall; static hold remains quiet. |
| 4. Combined robustness | Mix reach, payload, and timed push, including asymmetric cases; retain easier cases to prevent forgetting. | `combined`: fall rate <= 10%, success >= 75%; robust-suite degradation is bounded; all earlier gates remain acceptable. |
| 5. Locomotion and height | Add a masked shoulder-line height target, ground-reaching wrist targets, and then introduce base twist gradually. Permit forward waist flexion, knee flexion, or mixtures; prohibit using backward torso lean to lower the shoulders and penalize excessive shoulder height difference. | During automated exploration use wrist/height/velocity errors, falls, slip, backward lean, shoulder level, and action smoothness only as provisional diagnostics. The user performs final visual and held-out judgment. |
| 6A. Continuous wrist motion | Procedural smooth bilateral pose trajectories, including asymmetric motion, deliberate pauses, ground approach/lift/place, and retained static holds. | Whole-trajectory precision, lag, peaks, stability, and static regression; see `experiments/continuous_wrist_training_plan.md`. |
| 6B. Continuous loco-manipulation | Track continuous world trajectories with clutch OFF and relative operation trajectories composed with an independent transport reference with clutch ON. | Mode-conditioned tracking and body progress; preserve free support adjustment and target continuity. |
| 6C. EgoDex/Pico trajectories | Train on mapped continuous bimanual trajectories, with held-out objects/trajectory clips and latency/noise randomization. | Generalizes to unseen clips and asymmetric manipulation; simulator-to-real safety review precedes hardware execution. |

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
controlled factor. Stage 6A uses procedural task trajectories and does not require
EgoDex data, but needs a continuous command generator and whole-path telemetry.
Only Stage 6C requires the mapped trajectory split and interface contract. See
`experiments/continuous_wrist_training_plan.md`; this future plan does not extend
any current experiment budget or authorize an automatic launch.

The first Stage 5A screen is rejected despite completing normally. Its absolute
wrist-height and shoulder-height targets were sampled independently. The G1
shoulder-pitch-to-wrist-yaw kinematic chain is at most about 0.410 m before
joint-limit and collision margins, while 75.5% of the moderate samples and
95.3% of the deep samples exceeded that distance in the vertical axis alone.
The approximately twofold increase in aggregate wrist error when height-task
frequency increased from 25% to 50%, together with a final shoulder height near
1.07 m in every run, indicates that the policy preserved the old task and
sacrificed the height-active cases. The subsequent twist screen is therefore
also rejected as a Stage 5 promotion result.

Stage 5A repair v2 couples the targets: the final wrist height is its initial
height plus the commanded shoulder-height change plus only a small residual
offset. This preserves the initially feasible shoulder--wrist vertical
separation while permitting grasp-height variation. A height-active Huber loss
is added for both wrist position and shoulder height so large errors retain a
non-saturating gradient; the existing exponential/fine terms remain responsible
for final precision. Diagnostics must report height-active and non-height wrist
errors separately. Do not introduce planar twist until a stationary candidate
has height-active wrist error below 4 cm, height-active shoulder error below
5 cm, nearly full episode length, no systematic backward lean, and acceptable
non-height wrist retention. The repair screen is defined in
`experiments/stage5a_feasible_screen_v2.json`.

## Stage 5A v3 correction and bounded continuation

The v1/v2 conclusions above are superseded by a confirmed sampling bug:
`tensor[env_ids].uniform_()` updates an advanced-index copy, not the original
tensor. Shoulder samples and residual offsets stayed zero; symmetric reach
extension was also zero or stale. Resolved YAML alone did not prove that the
commands were applied. The v2 moderate/deep distribution comparison is invalid,
and preserving a shoulder--wrist vertical gap alone does not prove whole-body
or orientation feasibility. Previous fixed scripted held-out results remain
separate evidence, but training-distribution claims require this correction.

V3 assigns sampled values back explicitly and keeps all rewards unchanged.
Run `tools/check_wrist_command_sampling.py` with the training Python before
launching smoke/full training. It must check actual samples, noncontiguous reset
subsets, both height ranges, curriculum interpolation, initialized wrist/shoulder
targets, and fresh symmetric reaches. This CPU test is not a dynamics evaluator.
Smoke must also show plausible actual final shoulder/wrist targets, not merely
finite values. The reviewed first group is
`experiments/stage5a_sampling_screen_v3.json`.

Command metrics ending in `_masked` are population-average numerators, NOT
height-active conditional means. Use `tools/summarize_wrist_training.py` on the
remote machine to extract TensorBoard and divide numerator means by the actual
height command fraction (or its complement for non-height metrics). Historical
v2 names without `_masked` need the same normalization. Report unavailable
conditional means when the fraction is zero. These command-reset snapshots are
training diagnostics, not time-averaged held-out errors. Report backward lean
as a dimensionless sine/projection, not radians, and termination counts as
counts per logged batch, not fall percentages.

Authorized Stage 5A budget: at most THREE new registered groups, including v3
and any independent-seed confirmation; at most four one-GPU runs per group,
4,096 environments per run, and 5,000 PPO iterations per run. Validation is
excluded. Bugged v1/v2 groups do not consume this budget. Record consumed groups
in the journal. An unmet 5A gate is not by itself a stopping reason while this
budget remains and evidence supports a controlled intervention. Remain in 5A;
do not weaken its 4-cm wrist/5-cm shoulder requirements. Operationalize nearly
full episodes as final-100 mean length >=590/600, with non-height conditional
wrist error <2 cm and no systematic backward-lean failure. Final judgment is
still manual; no complete automatic held-out matrix is requested.

For a justified follow-up, vary only ONE conceptual factor: shoulder-depth
curriculum (shallower targets), curriculum ramp duration, or a common scaling
of the two height-active Huber loss weights. Pick the factor from actual failure
categories, preserve other settings, and validate risky changes. If v3 passes,
prefer independent-seed confirmation (at least three seeds) rather than
selecting a lucky seed. Stop on exhausted budget, repeated diagnosed crashes,
two genuine valid groups without targeted improvement, or absent evidence for
a safe next step; do not stop merely because a new JSON plan is not prewritten.
Stage 5B remains paused until stationary lowering is established.

## Stage 5B v3: explicit wrist transport clutch

Latest user-requested operation-first diagnosis and optional near-ground sampler
are documented in `experiments/operation_workspace_next.md`. This is a separate
development proposal; do not retroactively weaken the fixed v3 promotion gates
or restart full training from that proposal alone. The latest user authorization
now permits the workspace-only screen and conditional confirmation under the
TWO-group budget and gates in that file. Preserve old checkpoints; other Stage
5B budgets remain closed. Do not introduce gait changes in the workspace screen.

Stage 5A v3 passed all four independent training seeds (101/211/307/401).
The earlier twist-only 5B v2 completed, but its world-fixed wrists and sustained
travel commands conflict; its small wrist errors do not demonstrate locomotion.
Do not promote that screen to Stage 6 or repeat it unchanged.

The user authorized a binary wrist transport clutch and THREE episode types:
transport (clutch=1, 25%), anchored body adjustment (clutch=0, 25%), and
autonomous balance (clutch=0, zero command, 50%). These are terminal mixture
fractions; motion exposure and amplitude ramp over 30k warmup/60k ramp steps.
Transport wrist world poses are advected by an independent reference initialized
at reset and integrated from commanded planar twist, NEVER from actual body
motion. Shoulder height stays world-z. Anchored targets stay world-fixed.
Adjustment commands start after 3 s, last at most 2 s, and have a cumulative
commanded path-length budget of 8 cm and absolute yaw budget of 0.12 rad.
These are command budgets, not hard constraints on actual robot displacement;
recovery steps remain permitted. Modes stay constant within each episode.
Smooth clutch switching is deferred to a separate later experiment.

With the clutch enabled, append its binary value to actor/critic wrist commands.
This changes observation dimensions: train fresh policies; do not resume old
24-dimensional wrist-command checkpoints into the new model. Disabled defaults
retain old observation dimensions and behavior. Planar/yaw quiet-motion penalties
apply only when there is no requested motion, not during transport/adjustment;
feet are not position-constrained. Existing zero-command penalties remain soft.

Run both CPU checks (`check_wrist_command_sampling.py`, `check_wrist_clutch.py`),
then a one-GPU smoke with all three modes active and inspected TB mode fractions
and finite metrics before full training. The first group is
`experiments/stage5b_clutch_screen_v3.json`: one GPU and 4,096 global environments
per run, 16,384 concurrently across four runs, seed 1802, 5,000 iterations.
Only the common linear/yaw tracking reward scale changes (1/2/4/8).
Keep corrected stationary 5A height settings unchanged.

Report normalized per-mode wrist/shoulder errors and moving-only velocity errors;
divide `_masked` numerators by the matching mode/moving fraction. Do not treat
zero-command balance samples or post-adjustment stops as successful locomotion.
Stationary retention requires balance wrist <2 cm, height-active wrist <4 cm,
height shoulder <5 cm, final-100 episode length >=590/600, and no systematic
backward lean. Moving transport/adjustment must additionally have per-mode wrist
<4 cm and moving-only xy/yaw error <0.05 m/s and <0.10 rad/s respectively.
Moving-only wrist error must also be <4 cm; xy/yaw errors must each be below
50% of the respective moving-only commanded magnitude. This prevents a standing
policy from passing merely because the requested adjustment speed is small.
Moving statistics accumulate over each episode before reset; non-moving wrist
metrics remain command-reset snapshots. Unavailable moving fractions cannot pass.
These fixed gates are provisional training diagnostics, not held-out guarantees.
Compare per-mode precision, moving errors, terminations, and smoothness rather
than total reward. Preserve every checkpoint. No automatic two-hour evaluator.

Authorize at most THREE registered 5B v3 groups (screen, one justified follow-up
if needed, and independent-seed confirmation). No unlimited loop. If the gate
passes, confirm the unchanged winner on at least three independent seeds. If
unmet, allow one evidence-based single-factor follow-up within budget; do not
weaken gates. Stop after exhausted budget, diagnosed repeated crashes, absent
safe next intervention, or after successful confirmation. Stage 6C remains blocked
on mapped trajectory data/interface; procedural Stage 6A first needs implementation,
validation, and a separately approved bounded experiment budget. Send existing
group summaries by email.

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
