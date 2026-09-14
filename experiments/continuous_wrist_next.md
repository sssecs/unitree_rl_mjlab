# Authorized Stage6A1: bounded continuous wrist position cycles

## Latest user revision: capability-pack exploration

The user explicitly requested cancellation of all running3cm siblings and direct
operation capability-pack training. They were gracefully stopped; CANCEL the
automatic three-seed confirmation below and skip this cancelled group's analysis
trigger. Follow `operation_capability_pack_next.md`. The old Group2
instructions are historical and no longer authorize a confirmation launch.
No existing numerical gate is weakened; baseline failure is still recorded.
A learned-quality miss alone does not block a justified broader capability-pack
screen; unresolved runtime/NaN/reference-frame failures must be fixed first.

User requested the next training after all transport confirmation seeds passed.
This new two-group budget supersedes that completed STOP. First isolate continuous
operation with zero body commands, as Stage6A specifies; it does NOT confirm
retained locomotion. Stage6B will separately recombine mobility and continuous
targets. Preserve stationary bilateral25/ground25/height50 sampler, shoulder
coupling, leg8x, PPO, disturbances and every reward. No new reward or imitation.

## Implemented subset and limits

After original1s delay/2s reach, continuous episodes track horizontal segments
between the reached pose and one independently sampled nearby waypoint per hand.
Offsets are clamped to the same horizontal sampler box and <=.03m displacement.
World z and shoulder target remain unchanged: both low targets stay low. Hands
have independent4--6s periods. Quintic interpolation of triangular phase gives
C2-continuous go/return cycles with zero endpoint velocity/acceleration. Initial
reach also uses quintic ONLY for continuous episodes. Orientation reaches its
sampled goal then holds. Existing current pose/velocity-error observations suffice;
no future knots, actor mode labels or observation-dimension changes.

This is a bounded procedural POSITION tracking subset, not arbitrary6D, vertical
pick/lift/place, nonperiodic EgoDex/Pico, or full trajectory generalization.
Steady target speed<=.028125m/s and acceleration<=.045m/s^2, from3cm maximum
displacement, minimum2s half-period, quintic derivative bounds. Bounds do not
cover the existing initial reach. Sampling-box/height checks are not a complete
IK/contact/collision feasibility proof; learned success must be tested empirically.

## Validation and screen

CPU reset/low-height/displacement checks and existing telemetry checks must pass.
Before full launch require GPU256env/20iteration smoke, all-continuous/bilateral/
ground/height, full curriculum, exit0, finite PPO/new scalars, nonzero continuous
moving fraction. Smoke may use reach timing0/.10s to exercise post-reach logic
before random-policy falls; formal runs retain1/2s. Read target velocity, not
only resolved configuration. No full automatic two-hour evaluator.

`continuous_wrist_screen_v1.json`: four fresh seed2201 one-GPU candidates,
4096globalenvs/run,5000iterations,16384 concurrent. Change only continuous
probability0/30/60/90% among nonstatic reach/height episodes. Existing static
episodes remain static; effective exposure is lower than these conditional
probabilities. Zero candidate is control, not a continuous winner.

## Predeclared final100 gates

Use summarize_wrist_training conditional metrics. Continuous metrics accumulate
every post-reach sample, including failures before reset; moving metrics exclude
speed<=.001m/s. Wrist error is worse-arm per sample, orientation likewise.
Actual velocity projected onto commanded wrist direction prevents accepting
an immobile wrist solely because the3cm trajectory is small. Statistics are
episode-weighted training means, not pooled-frame p95, measured lag or a fixed
held-out guarantee; those remain final review diagnostics.

- exit0; finite histories; continuous fraction>=.02 and moving fraction>=.01.
- continuous worst-arm mean position<3cm, orientation<.2rad; conditional target
  speed>=.003m/s; projected actual wrist speed>=40% of target speed; mean
  velocity error<.15m/s. No zero-target or stopped-wrist candidate can pass.
- Preserve bilateral wrist<4cm/shoulder<5cm, ground/height wrist<4cm/shoulder
  <5cm, nonheight wrist<2cm, episode length>=590/600, backward-lean termination
  <=.01 per logged batch, nonzero low targets0.16--0.28m.
- Vs same-seed control: nonheight wrist regression<=.3cm, episode regression
  <=3steps, leg RMS snapshot increase<=25%. Inspect slip/contact/shoulder level
  and action acceleration; don't equate motion-required acceleration to shaking.

Among passing candidates choose the highest exposure whose continuous position
error is within.5cm of best passing error. Same-seed screening is provisional.
The former Group2 three-seed confirmation is cancelled by the latest revision.
Do not create or launch continuous_wrist_confirm_v1.json for this narrow subset.

If none passes, stop with diagnosis, no weakened gates or extra unbounded groups.
Email existing summaries. The latest capability-pack plan determines the next
step; never start unimplemented functionality just by changing JSON flags.
Preserve remote checkpoints and disclose no locomotion retention evidence for
this stationary subset. Do not overwrite mobile model aliases automatically.
# Latest override: cancelled small-motion screen

User requested cancellation and direct capability-pack training. The four
`g1_continuous_wrist_screen_v1` jobs were gracefully interrupted; checkpoints
remain remote and the group is marked handled. Do not launch or confirm these
jobs. Follow `operation_capability_pack_next.md` and its one-screen budget.
