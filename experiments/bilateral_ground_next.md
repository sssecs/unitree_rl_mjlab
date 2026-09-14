# Authorized overnight bilateral low wrist + shoulder training

The user visually confirmed no obvious high-frequency shaking in the 8x leg
screen winner, then explicitly authorized automatic next-stage training without
waiting for another confirmation while asleep. This document overrides the old
instruction to STOP after `g1_ground_leg_smooth_confirm_v1`, ONLY if that group
passes its unchanged absolute workspace gates. Analyze all siblings and update
the existing journal/email first. If it fails, stop and diagnose, no promotion.

## Launch prerequisite

CPU sampling/reset tests must pass, and `g1_bilateral_ground_smoke_v1` must exit0
with finite PPO/new bilateral metrics and nonzero BOTH-wrists-low samples at
0.16--0.28 m. Read remote smoke TensorBoard and its resolved configuration.
Do not launch if this validation is missing or fails. Check remote GPU status:
the next four-way group must wait for all preceding three seeds to finish.
Source defaults remain bilateral probability0, so preceding runs are unchanged.

## Controlled experiment and budget

`bilateral_ground_screen_v1.json` is group1 of at most TWO new full groups.
Each of four independent single-H20 runs has 4096 global envs, 5000 iterations,
fresh training seed2001 (16384 envs concurrent). Keep all arguments from the
unchanged leg8x confirmation. Change only bilateral fraction within near-ground
episodes: 0/25/50/75%, giving 0/3.125/6.25/9.375% overall bilateral exposure
with height probability50% and conditional ground25%. Retain unilateral tasks.
The zero-exposure sibling is a control, NOT a bilateral winner.

For bilateral samples independently sample both wrist heights in 0.16--0.28 m
and existing bilateral horizontal ranges; derive one shoulder height from the
lower of the two initial-gap-implied shoulder heights, with existing residual.
Interpolate with the existing height curriculum; at full curriculum both final
heights are in range. This does not prove collision/joint-limit/whole-body
feasibility: runtime failures and learning failure remain meaningful outcomes.
Permit forward flexion, knee flexion or combinations; preserve existing backward
lean and shoulder-level constraints. No speed commands, continuous trajectories,
new reward, support-load balancing, actuator/filter, or DR changes this round.

## Fixed provisional gates before launch

Use final-100 TensorBoard samples normalized by actual command fractions via
`summarize_wrist_training.py`; these are reset snapshots, NOT held-out or
whole-path guarantees. All scalar histories must be finite and runs exit0.

- Nonzero bilateral exposure; normalized min and max bilateral target means
  BOTH in 0.16--0.28 m. Smoke checks individual targets, not only their mean.
- Bilateral wrist error (worse arm in each sample) <4 cm, bilateral shoulder error <5 cm.
- Preserve ground/overall-height wrist <4 cm and shoulder <5 cm; nonheight
  wrist <2 cm; episode length >=590/600; mean backward-lean termination
  <=0.01 per logged batch, with no systematic lean failure.
- Vs sibling00: nonheight error may increase at most0.3 cm, episode length may
  decrease at most3 steps, and leg acceleration RMS snapshot may increase at
  most20%. Inspect unchanged-weight slip, shoulder level and contacts; do not
  rank total reward or minimize support motion blindly.

If multiple candidates pass, select the lowest positive exposure whose bilateral
wrist error is within0.5 cm of the best passing candidate, considering retention.
Tiny same-seed differences do not justify a final policy claim.

Group2 only confirms the unchanged passing setting with three fresh independent
seeds2002/2003/2004 on GPUs0/1/2, 4096 envs/run, <=5000 iterations. Create a
descriptive `bilateral_ground_confirm_v1.json` locally, inspect diff, synchronize
and launch through `remote_sweep.py`; no extra smoke if configuration unchanged.
Every confirmation seed must pass absolute gates; report across-seed spread and
smoothness limits. Do not pretend unpaired seed comparisons prove causality.

If none passes, or budget exhausted or a diagnosed crash prevents safe progress,
stop and email the existing summary; never weaken gates or retry indefinitely.
After confirmation STOP for user visual review. Preserve checkpoints/aliases.
No automatic two-hour evaluator or additional stage launches. Subsequent steps
are bilateral-low + speed, then continuous wrist trajectories, separately
implemented/validated/budgeted. This budget authorizes neither step.
