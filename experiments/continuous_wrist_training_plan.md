# Future goal: continuous wrist trajectories

## Scope and hypothesis

The user requires wrists to keep moving along manipulation trajectories, rather
than just reach one sampled pose and hold. Smooth procedural task commands should
teach this without human motion references, joint-action imitation, or future
frames. Retain static holds so continuous motion does not become meaningless
drift. The current generator is still a single smooth reach followed by a hold:
this document is a future specification, NOT an implementation or launch order.
It does not extend the current two-group workspace budget.

## Command interface and reference frames

- Inputs remain bilateral Cartesian wrist poses, planar twist, shoulder-line
  height and mask, binary clutch, and current robot state; output is 29 joint actions.
- Clutch OFF: wrist targets follow a world-frame trajectory. A constant world
  pose is a deliberate subset, not the only supported case. Commanded small body
  adjustments and uncommanded balance steps must not drag wrist targets with them.
- Clutch ON: compose the relative operation trajectory with the independent,
  command-integrated planar transport reference. Do not attach targets to the
  measured robot body, double-advect them, or confuse operation with transport.
- Couple shoulder-height sampling to low wrist tasks. Keep the forward-flexion
  constraint and shoulder-level penalty; allow knee flexion or mixed solutions.
- Actor sees current commands and proprioception/history, not future trajectory
  knots or poses. Present target derivatives may be considered as a separate
  interface change; review observation dimensions/checkpoint compatibility first.
  Teacher privileged observations remain current-state simulation information.

## Stage 6A: procedural continuous motion, stationary base commands

Implement repeated smooth Cartesian segments or splines and shortest-arc
quaternion interpolation, with bounded target speed and acceleration. Initially
use smooth stop/start segments and deliberate pauses, then introduce nonzero
through-waypoint velocity with continuous bounded acceleration. Avoid target
jumps, quaternion sign flips, and discontinuous segment handoffs.

Randomize direction, amplitude, speed, timing, bilateral phase, and asymmetry.
An initial proposed mixture is 30% static holds, 50% continuous moderate-workspace
motion, and 20% low approach/lower/lift/place clips. These are task-space wrist
commands, not prescribed body, joint, or foot motions. Start low clips inside the
validated 0.16--0.28-m workspace; do not automatically deepen to 0.08--0.18 m.
Check the whole path for kinematic reachability, joint/collision margins, support
feasibility, and palm-frame conventions; shoulder-wrist vertical distance alone
is not proof of feasibility. Freeze numerical speed/acceleration limits from
these checks before screening, rather than inventing unvalidated values.

Test the generator as one conceptual factor. Preserve PPO, load/push settings,
reward scales, and existing safety constraints in the initial comparison.

## Stage 6B: continuous loco-manipulation

After stationary tracking passes, add transport, then anchored small body
adjustments, then smooth clutch switching as separate controlled changes.
Preserve world target pose and velocity continuity on switching; rebase the
command reference, not the measured robot state. Require observable body endpoint
error/progress before claiming successful anchored adjustment. Do not fix the
feet or impose a recovery cadence/minimum stride on balancing cases.

## Stage 6C: mapped EgoDex/Pico-like trajectories

Use continuous bilateral, asymmetric approach/pick/lift/carry/place task clips,
then mapped recordings once the mapping, coordinate/palm conventions, and split
are defined. Hold out objects and entire clips, not adjacent frames from the same
clip. Add latency/noise as separate factors. Recordings supply task commands,
not human joint-action supervision or an imitation teacher; no future frames.

## Measurement and promotion

Measure over the whole trajectory, not only command-reset/endpoint snapshots:

- Position and geodesic orientation error: mean, p95, peak; target/actual wrist
  velocity, tracking lag, response and recovery time. Separate startup/steady motion.
- Condition metrics on static, moderate, low, transport, and anchored-adjust cases.
  Exclude post-autoreset states from the previous episode, but include failed
  trajectories and falls rather than reporting only survivors.
- Monitor wrist/shoulder height, backward lean, shoulder level, slip/contact
  transitions, action smoothness, and commanded body progress. Do not reward
  pointless wrist motion or penalize intentionally commanded pauses.

Freeze numerical promotion gates before the first new screen, preserving current
static/height gates. Compare against the unchanged reach-and-hold baseline using
fixed diagnostic seeds; final claims require at least three independent training
seeds plus user visual/unchanged held-out judgment. No automatic two-hour evaluator.
Before a full run, test reset/segment/quaternion continuity and run a small GPU
validation for finite PPO, initialization, and throughput.

## Watcher handoff

After workspace confirmation, stop for user review as currently required. Next:
implement the generator and whole-path telemetry, validate, and define a separately
approved bounded screening/confirmation budget. Do not treat this document as
permission to chain additional runs after the current budget. Reuse existing
analysis summaries for emails; do not generate a second notification analysis.
