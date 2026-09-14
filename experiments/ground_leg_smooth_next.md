# User-authorized leg jitter screen

LATEST authorization: after passing leg confirmation, follow
`experiments/bilateral_ground_next.md` instead of stopping for another user
approval. Its smoke prerequisite and separate two-group bilateral budget apply.
All leg confirmation gates remain unchanged; failure still requires stopping.

The user visually observed intermittent high-frequency leg shaking and authorized
the next training. Prioritize an isolated leg-smoothness screen before continuous
wrist-generator changes. This supersedes the completed workspace STOP decision
for this new budget only. Continuous trajectories remain the subsequent goal.

Hypothesis: existing whole-body acceleration regularization is too weak for legs.
This is not a diagnosed cause: PD/contact behavior or play/training differences
may also explain shaking. Keep actuator gains, control rate, action scaling,
sampling, PPO, pushes and payloads unchanged. Add a default-zero leg-only
acceleration term over twelve hip/knee/ankle joints. With the original -2.5e-7
whole-body term, the four total leg coefficients are 1x/2x/4x/8x. Arm penalties
do not change. No human motion reference, fixed feet, or low-pass action filter.

Budget: at most TWO registered full groups: the four-way same-seed screen in
`ground_leg_smooth_screen_v1.json`, then unchanged three-independent-seed
confirmation only if a nonzero candidate passes. Each run trains from scratch
for <=5000 iterations with 4096 envs on one GPU; screening is 16384 concurrent
envs, confirmation is 12288. Runtime smoke is a separate 256-env, 20-iteration
validation, not an optimization group. No automatic full held-out evaluator.

Before full launch verify twelve leg joints resolve, finite PPO and finite new
metrics. New `leg_joint_acc_rms_snapshot` and `leg_joint_vel_rms_snapshot` are
unweighted command-reset snapshots, NOT continuous frequency-spectrum evidence.
Use final-100 normalized wrist metrics from `summarize_wrist_training.py`.

Screen gates, fixed before launch:

- Preserve every existing workspace gate: ground/height wrist <4 cm;
  ground/height shoulder <5 cm; nonheight wrist <2 cm; episode length >=590;
  nonzero ground exposure with target 0.16--0.28 m; finite scalars; no systematic
  backward lean (mean backward-lean termination <=0.01 per logged batch).
- Relative to this group's same-seed control, leg acceleration RMS snapshot
  must decrease >=20%; mean action acceleration must not increase >5%.
- Ground wrist error must not exceed control by >0.5 cm; nonheight error must
  not exceed control by >0.3 cm; episode length must not drop >3 steps.
  Velocity RMS alone is not a success metric: required support motion is allowed.
- Compare foot-slip reward normalized by its unchanged coefficient, contacts,
  shoulder level and falls; flag deterioration, do not maximize total reward.

If no candidate passes, stop and diagnose; do not relax gates, retry blindly,
or add another variable. If several pass, prefer the weakest passing coefficient
unless a stronger one has clearly better task/smoothness evidence. Confirm its
unchanged configuration on seeds 1903/1904/1905. Every seed must meet absolute
workspace gates; compare aggregate smoothness with the screen and archived
confirmation, reporting limitations rather than claiming paired multi-seed
causal evidence. Stop after confirmation for user visual review. Do not silently
replace model aliases or launch Stage 6A within this budget. Email only existing
analysis summaries. Preserve all checkpoints.
