# Operation-first workspace and locomotion follow-up

## Evidence and scope

Short diagnostic `results/remote_eval/clutch12_nominal_diag_v1` uses the scale-12
checkpoint, no payload/push/startup DR/observation corruption, seed 9187, two
instances per case, one 12-second rollout. It is development diagnosis, not
multi-seed evidence or a change to the fixed held-out suite.
Forward/backward transport mean signed speed is +0.0883/-0.0950 m/s for
commands +/-0.1; wrist means 0.47/0.53 cm, no falls. Pure transport yaw responds
0.0451 rad/s to 0.1; anchored adjustment responds 0.0122 m/s to 0.04 and
0.0129 rad/s to 0.06, with wrist means 0.35/0.67 cm. Zero-command nominal
absolute yaw is 0.00967 rad/s, far below training mixture statistics. Thus
disturbances influence the latter, but adjustment under-response is real.
Only three >=100ms-flight touchdowns were observed across forward cases;
none in backward cases. This DOES NOT establish normal walking: inspect foot
slip and brief contact switches before inferring stride. Diagnostic v2 adds
contact-foot speed and displacement projected on initial heading, avoiding
misreading world-x under randomized initial yaw.

Enhanced v2 diagnostic: forward initial-heading displacement is +1.064 m
against +1.2 m integrated command; backward is -1.140 m against -1.2 m.
Contact-foot speed is 0.0661/0.0540 m/s with only 1/0 >=100ms-flight touchdowns
across two envs per case. The single same-foot displacement of 0.960 m is NOT
a trustworthy stride length: it includes displacement accumulated between sparse
events and can include slip. Sliding/dragging or brief-flight shuffling requires
visual inspection. Prioritize contact-slip and validated stepping before minimum
stride or cadence rewards. Balance contact-foot speed is only 0.00198 m/s.
Anchored forward adjustment finishes at +0.00766 m initial-heading displacement
despite a +0.08 m commanded budget; its ~0.0121 m/s response during movement
does not translate into adequate retained adjustment. Transport yaw response is
0.0466 rad/s to 0.1, and adjustment yaw 0.0135 to 0.06. These remain a single
diagnostic seed with only two instances/case, not robust generalization claims.

## Wrist workspace change (implemented; disabled by default)

The retained moderate sampler targets shoulders 0.78--0.98 m and wrist height
near 0.53 m. Height tasks previously used symmetric forward extensions and
coupled bilateral lowering, not a representative floor-pick workspace.

New `height_spatial_sampling` independently samples bilateral forward/lateral
offsets on height tasks. `ground_probability` is the fraction WITHIN height
episodes allocated to unilateral near-ground tasks. Randomize which hand is low;
derive shoulder height from that hand's initial shoulder/wrist vertical gap,
low wrist height, and small residual. Raise the other hand 0.10--0.20 m rather
than forcing an unreachable standing-height hand while shoulders crouch.
This is robot-geometry-based task sampling, NOT imitation or a joint-pose target.
Old settings/checkpoint dimensions remain unchanged. CPU tests verify target
values and noncontiguous reset isolation, NOT full-body feasibility.

First validate ground wrist range 0.16--0.28 m; keep ordinary moderate samples
and start with ground_probability 0.10--0.25 and zero base commands. Only after
joint-limit/collision/support and dynamics inspection advance toward 0.08--0.18 m.
The latter is the available sampler default, NOT a declared feasible range.
Ground wrist-body height is not palm/fingertip/object clearance: include the
wrist-to-palm transform and grasp orientation before calling this ground pickup.
Current torso/backward-lean, shoulder-level, pelvis-height and contact constraints
must remain in force; audit whether they exclude otherwise useful deep poses.
Do not simply sample independent wrist/shoulder heights or remove safety terms.

Actual engineering smoke `g1_ground_sampling_smoke_v1`: one H20 (GPU1),
256 envs, 20 iterations, zero twist, all-height/all-ground, spatial sampling,
ground wrist 0.16--0.28 m, extension 0.02--0.18 m, no warmup/1-step ramp.
TB has 20 finite samples; final-10 actual low target mean 0.2187 m, shoulder
target mean 0.5666 m, bilateral wrist target mean 0.2936 m. Random-policy wrist
error ~0.685 m and frequent falls do NOT demonstrate dynamics feasibility or
successful training. Both CPU checks pass. No formal screen has been started.

## Operation-first motion acceptance (proposal, not retroactive promotion)

Do not change the frozen held-out evaluator or declare old failed velocity gates
passed. Introduce a separate operation-oriented development test:
- Transport: progress along the requested path and accumulated yaw, lateral
  drift, final wrist pose, wrist peak/p95, slip, contact behavior, and falls.
  Judge steady motion over >=1s windows; permit within-step speed oscillation.
- Anchored adjustment: finite body displacement/yaw target from the integrated
  command; judge endpoint after stopping, with wrists world-fixed. Zero body
  movement cannot pass. The current speed-only interface provides no explicit
  terminal pose-error signal, so endpoint control needs an observable reference
  error and its own controlled implementation, not just a looser speed threshold.
- Balance: no prescribed feet or compulsory stepping. Track wrist precision,
  recovery and unnecessary contact chatter. Report nominal and pushed separately.

Set new numerical acceptance thresholds BEFORE screening the next configuration;
do not fit them to the scale-12 checkpoint. User visual review remains required.

## Large transport steps / small adjustments, without human motion references

Relevant primary sources:
- Walk These Ways: https://gmargo11.github.io/walk-these-ways/
  and https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py
  demonstrate behavior-conditioned procedural gait control, variable frequency,
  swing/posture/speed, but on a quadruped; G1 transfer is a hypothesis.
- Learning to Walk in Minutes: https://arxiv.org/abs/2109.11978
  and https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py
  provide task rewards/curricula including air-time rewards; air time alone is
  NOT a guarantee of useful large steps and can encourage hopping.
- WoCoCo: https://proceedings.mlr.press/v270/zhang25a.html
  demonstrates whole-body humanoid contact tasks without motion priors. Its
  explicitly staged contacts should not be imposed on autonomous balance here.

Recommended first intervention is task-conditioned, phase-free step economy:
1. Transport commands must include actual travel speeds (later 0.2--0.5 m/s),
   not exclusively <=0.1. At 0.1, a 0.3-m step requires roughly 3s between
   effective forward steps; forcing big steps at every tiny speed is unnatural.
2. Count validated lift-off/touchdown with contact debounce and flight duration.
   Reward useful directed progress per step; penalize redundant rapid touchdown
   only in sustained transport, jointly with slip and actual progress so sliding,
   standing, and jumping cannot game the term. Use modest, bounded coefficients.
   First audit/reweight slip independently: the nominal diagnostic already
   suggests appreciable contact-foot motion. Do NOT reward the raw 0.960-m
   sparse-event displacement as a large step. Landing displacement must be
   measured over a verified swing, excluding stance slip and contact chatter.
3. Adjustment uses endpoint achievement and a weak unnecessary-step cost, with
   no minimum stride; allow a small step or an ankle/hip/waist shift. Balance
   excludes minimum stride/contact schedules and allows urgent recovery steps.
4. If phase-free shaping still shuffles, a separate ablation can add procedural
   transport-only cadence/soft contact phase. This is NOT human-reference data,
   but does constrain gait timing; never apply it to anchored balance/recovery.

Do not simultaneously change workspace, reward scales, cadence, speeds, and DR.
Sequence: diagnose -> workspace-only smoke/screen -> operation progress interface
-> step-economy ablation -> optional cadence ablation -> independent seeds.
No new full training budget or watcher restart is authorized by this file.
Prepared, unlaunched workspace-only plan: `experiments/ground_workspace_screen_v1.json`.
Four one-GPU runs, 4096 envs/run, 5000 iterations, seed 1901, conditional ground
exposure 0/10/25/40%. All use zero base commands; this is NOT a locomotion
confirmation and temporarily isolates depth feasibility from gait changes.
