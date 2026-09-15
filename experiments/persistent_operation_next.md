# Active authorization: persistent operation screen v1

User approved the next training after visual replay was updated. Budget: ONE
four-way screen, then analyze/email existing summary and STOP for manual review.
No automatic confirmation, extra group, reward tuning or workspace expansion.

Hypothesis: continue independent-hand task clips after their endpoints instead
of holding until reset, so pure RL learns sustained tracking and whole-body
adaptation to moving self-obstacles. Do not reject a task because current legs
block it; no wrist-error termination. Existing fall/backward-lean termination,
self-collision reward and physical contacts stay unchanged. No human motion
references, imitation, future frames or prescribed feet.

`persistent_operation_screen_v1.json`: identical balanced policy+optimizer
warm-start `2026-09-14_11-27-20/model_9998.pt`, seed2401,4096envs per single
H20 /16384global concurrent,5000 NEW iterations (expected9998--14997).
Keep balanced curricula15000warmup/60000ramp for wrists and twist, physics,
PPO/rewards, loads/pushes, workspace and mode distribution unchanged. Control
uses one clip; other GPUs use persistent probabilities .30/.60/.90 conditional
on existing continuous episodes. Static holds remain, continuous probability.80.

Completed hands independently get new clips; start pose is their previous exact
endpoint, with zero boundary velocity/acceleration. New tasks stay relative to
the ORIGINAL command anchor, not accumulated offsets or measured body motion.
Existing6cm XY,0--12cm lift,20degree quaternion perturbations remain. Independent
reference rotation advects clip anchors and offsets once.20% successor first
segments pause; other knots vary and may reverse. Episode length remains12s
for all runs: this screen removes end holding, but does NOT establish minute-long
continuous operation. Check actual successor counts rather than assert coverage.

Diagnostics are common to all candidates, do not enter actor/critic/rewards:
dedicated max-force arm-leg contact sensor; contact-associated wrist error;
release/recovery count and time (<3cm for .20s after contact disappears);
finite-difference wrist velocity and disagreement with reported velocity.
Recovery averages include only successful recoveries; unrecovered/censored
episodes must not be hidden. Contact is a proxy, NOT proof of target obstruction.
Geometric near-miss without contact is not measured. Bounded CSV records first
two envs128steps each10000steps, poses/velocities, leg states and base commands.
This is sparse diagnostic sampling, not a population p95 or complete contact trace.

Before full launch: CPU old regressions + independent successor/reset/frame tests,
actual compiled-model contact audit (8arm/22leg active geoms;9synthetic arm-leg
penetrating contacts in128CPU probes; probes are not training data). Then256env
GPU smoke from balanced checkpoint with persistence1, all bilateral/low targets,
full curricula and mixed modes. Require finite PPO, compatible534dimensions,
nonzero successors and all mode coverage. Inspect CSV finite-difference behavior
before interpreting high velocity error. Actual contact coverage may be zero;
if so, report avoidance learning unverified, not artificially pass it.

Frozen provisional gates, final500iteration conditional metrics:
- all finite,5000new iterations, episode length>=580;
- continuous wrist<=.015m, rotation<=.25rad, shoulder<=.03m,
  mean episode peak wrist error<=.05m;
- conditional command speed>=.008m/s, actual projected speed gain>=.35;
- persistent cases require steady exposure>0 and mean hand continuations>=.05;
  persistent-only wrist<=.015m, orientation<=.25rad, shoulder<=.03m;
- bilateral moving low-target intersection>0, wrist<=.02m and shoulder<=.03m;
  initial low anchor range remains.16--.28m (lifts exceed this by design);
- backward lean<=.01rad. Velocity errors and action/leg smoothness must be
  compared with control; old.20m/s reported-velocity gate is reported separately,
  not silently weakened or replaced by FD averages.

Rank by task precision/stability, then successor coverage and contact/recovery
evidence; never total reward alone. No fixed held-out matrix or final best-policy
claim; same-seed screen cannot establish small causal improvement. p95/lag,
unblocked target reachability and object manipulation success remain unavailable.
If no arm-leg contact/release exposure, do not claim active avoidance acquired.
After this screen STOP regardless of rank; ask user for visual review.
Contact episodes ending before release cannot be assigned a recovery time; these
are unverified, not successes. FD compares interval-average position derivatives
with instantaneous reported velocities, so nonzero disagreement can be physical
acceleration, not automatically a sensor bug. Smoke may trace every step for its
bounded30iteration budget; formal runs use the sparse defaults above.
