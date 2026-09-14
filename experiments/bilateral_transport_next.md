# Authorized bilateral low wrists + shoulder height + low-speed mobility

User visually accepted the bilateral-low model and authorized this next stage.
This is a NEW bounded budget overriding the completed bilateral STOP, not an
extension of its gates. Preserve wrist sampling, shoulder coupling, 8x leg term,
PPO, actuator rates/gains, disturbances and rewards. No human motion reference,
continuous task trajectory generator, action filter, fixed feet or minimum stride.

## Screen and validation

`bilateral_transport_screen_v1.json` is group1 of at most TWO full groups.
Common mixture: 25% transport clutchON, 10% bounded anchored adjust clutchOFF,
65% free balance clutchOFF. These proportions are fixed across siblings; this
introduces mobility relative to the archived stationary task. Within this NEW
mixture, screen only transport envelope scale: longitudinal +/-0, .08, .12,
.18m/s; lateral and yaw bounds are half each longitudinal numerical bound
(lateral m/s, yaw rad/s). Zero transport is a control, not a mobile winner;
it still includes the shared anchored-adjust task, so is NOT an unchanged
stationary baseline. Keep archived stationary results as another reference.
Existing command curriculum ramps probability and magnitude after warmup.

Each sibling starts fresh, seed2101, one H20, 4096 global envs, <=5000
iterations; four-way group totals16384 concurrent. Teacher target transport
uses the independent command-integrated planar reference, not measured body.
ClutchOFF targets stay world anchored, with bounded .04m/s/.06rad/s adjust
commands using existing .08m/.12rad budgets. No mid-episode clutch switches.

Before launch run CPU sampling/reset/normalization checks and a 256-env,
20-iteration GPU smoke at the largest envelope, fully enabled curriculum,
all-bilateral/all-ground/all-height/all-transport. Require exit0, finite PPO and
new intersection metrics, nonzero bilateral-transport exposure. This smoke
may shorten reach-delay/duration to0/.10s ONLY to exercise telemetry before
random-policy falls; full runs retain original1/2s timing. Smoke v1 lacked
post-reach samples for precisely this reason; v2 must exercise the intersection.
This is runtime validation, not learned motion. Code changes add telemetry only;
observations/reward semantics do not change.

## Fixed provisional promotion gates (final100)

Normalize masked diagnostics with actual fractions via summarize_wrist_training.
New `bilateral_transport_moving_*` accumulates episode moving samples after the
reach transition reaches phase>=.99, with planar command norm>1e-3. Wrist
error is worse arm per sample. Projected speed is measured body velocity along
the current planar command, NOT verified world-reference endpoint displacement.
Do not mix these accumulated diagnostics with endpoint/reset-only metrics.

Positive candidates must satisfy ALL:

- exit0, finite histories; actual bilateral targets nonzero and min/max means
  both0.16--0.28m; actual transport and adjust fractions nonzero.
- bilateral-transport moving fraction>=0.001, conditional commandxy>=0.02m/s;
  worse-arm wrist <5cm and shoulder <5cm; projected actual speed >=40% of
  conditional commanded planar speed. A stopped robot cannot pass just from
  relaxed instantaneous velocity error. Planar velocity error <0.25m/s.
- total bilateral wrist <4cm/shoulder <5cm; ground/height wrist <4cm/shoulder
  <5cm; nonheight wrist <2cm; mean episode length>=590/600; backward-lean
  termination<=.01 per logged batch. Balance wrist<2cm, adjust moving wrist
  <4cm/shoulder<5cm with nonzero adjust moving fraction. These adjust gates
  test task retention, NOT successful body endpoint adjustment.
- Relative to NEW same-seed control: nonheight wrist regression<=.3cm,
  episode-length regression<=3, leg RMS snapshot increase<=25%.
  Inspect slip, contact switching, shoulder level and action acceleration;
  motion-required acceleration alone is not shaking evidence. No reward ranking
  or claims of large steps without verified swing/contact diagnostics.

Among passing candidates choose the highest speed envelope whose intersection
wrist error is within.5cm of the best passing error. This ranks useful command
coverage, not precision tracking of every instantaneous speed. Tiny single-seed
differences are provisional. Do not relax gates after seeing results.

## Confirmation and stop

Group2 only confirms unchanged selected common/per-run settings using fresh
seeds2102/2103/2104, GPUs0/1/2, 4096 envs/run, <=5000 iterations, 12288
concurrent. Create descriptive bilateral_transport_confirm_v1.json locally,
inspect diff, sync, idle-GPU check and launch via remote_sweep.py. No repeat
smoke if configuration/source unchanged. Require all absolute gates per seed;
report unpaired seed and trajectory coverage limitations honestly.

If none passes, stop and diagnose; no unlimited retries or automatic new factors.
Stop after confirmation regardless of outcome, emailing existing summaries.
No automatic two-hour held-out matrix or continuous wrist stage. Do not overwrite
local model aliases automatically; latest user permits replacing exploratory
best_ground WITHOUT making a local backup when a subsequent download is requested.
Remote checkpoints remain preserved under repository policy.
