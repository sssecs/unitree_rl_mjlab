# Latest authorization: rich operation capability pack

## Change in experiment discipline

User requested less conservative capability development. Single-factor/multi-seed
discipline remains appropriate for causal reward claims, but every small command
extension no longer requires a full three-seed milestone. The running3cm screen
is CANCELLED by the latest explicit request, not awaited or confirmed. All four
jobs were gracefully interrupted; its group is marked handled to avoid duplicate
completion-triggered launches. Implement and validate the capability pack, then
launch at most ONE registered four-way capability screening group. Stop after
that group for user visual review; no automatic multi-seed confirmation or further
groups within this authorization. Final robustness/best-policy claims still need
independent seeds and unchanged held-out evidence. Preserve all remote checkpoints.

## Shared capability distribution (not optional separate future stages)

All candidates must support continuous bilateral XYZ and quaternion targets,
coupled shoulder height, clutchON independent-reference transport and clutchOFF
world trajectories with balance/limited commanded adjustment. Keep wrist motion
distinct from base transport; no measured-body target dragging, double advection
or switching discontinuity. Initially retain episode-level clutch modes; smooth
mid-episode switches may remain outside this first pack.

Mix intentional static holds/pauses, moderate workspaces, unilateral and bilateral
low approach, lowering, lifting, carrying and placing task clips. Include
asynchronous/asymmetric hands and nonperiodic multiple segments, rather than only
two-point3cm cycles. Clips specify wrist tasks, not robot joints or prescribed feet.
No human action reference, imitation teacher, CVAE, VLA or future frames.
Actor interface stays current bilateral poses, current robot state/history,
shoulder target/mask and base/clutch commands. Review dimensions and checkpoint
compatibility before adding any present derivative observation.

Keep existing reward set, leg8x, PPO and physics fixed initially. Do not add
load-balancing, cadence, minimum stride or action filtering to this pack. Shoulder
targets must follow bilateral wrist feasibility throughout low-to-high segments,
not only at the first waypoint. Retain backward-lean/shoulder-level constraints.
Initial low bounds stay0.16--0.28m, not deeper unverified ranges. The workspace
must expand continuously through curriculum, not by abrupt infeasible jumps.

## Implementation and validation BEFORE launch

`operation_clip.py` adds independently timed approach/lift/place segments per
hand:0--4cm,8--12cm,then0cm vertical offsets; horizontal offsets<=6cm and
quaternion perturbations<=20degrees. All offsets scale with curriculum, and
final poses hold rather than repeat. This is a bounded first operation pack,
not full teleoperation/ground workspace or a validated collision-free IK envelope.
Implement genuinely
multi-segment XYZ/quaternion trajectories and shoulder coupling; shortest-arc
quaternions, bounded translational/angular speed and acceleration, reset/segment
pose and velocity continuity. Test the entire sampled path for sampling bounds,
bilateral shoulder consistency and obvious kinematic/joint/collision problems.
Vertical-gap checks alone are not complete feasibility proof. Explain limitations.

Use existing CPU regressions plus trajectory/frame/reset tests. Check wrist and
shoulder paths during low/lift/carry, independent transport composition and world
anchoring. Then GPU256env/20iteration runtime smoke with each new case actually
exercised, finite PPO/targets, nonzero scenario counts and no exceptions. A smoke
may shorten task timing only for branch coverage, not to certify physical success.
If safe implementation cannot be completed within the completion turn, record
specific remaining work and STOP, not launch the old generator under a new name.

Add scenario-conditioned whole-episode task metrics, not just reset snapshots:
position/geodesic orientation, peak errors, command/actual motion and body progress,
falls, slip/contact transitions, low-target/shoulder validity and static retention.
Where p95/lag is not yet available, explicitly report unavailable; it cannot be
invented or replaced by a pooled mean. No automatic two-hour held-out matrix.

## Four-card screening and bounded handoff

Before launch materialize one reviewed JSON plan and numerical provisional gates
for each scenario, then freeze them. Candidate quality compares independent task
metrics, never raw total reward. Rich coverage is shared across all four cards;
compare four coherent curriculum/mixing strategies (for example gentle ramp,
balanced, low-task emphasis, asymmetry emphasis), not tiny axis additions.
Changing multiple distribution parameters here tests a strategy, not a causal
effect of each knob; record this explicitly. Keep PPO/rewards/physics common.

Prefer warm-starting all four from the SAME confirmed pure-RL transport checkpoint
if actual launcher/runner semantics permit compatible policy loading, a fresh
curriculum and the exact incremental training budget. Verify optimizer, seed and
iteration-counter behavior first; otherwise use matched fresh initialization and
document why. Warm-start is RL fine-tuning, NOT action imitation. Allow4096global
envs per single-H20 run, <=5000 NEW PPO iterations, distinct GPUs0--3, explicit
matched screening seed;16384envs concurrent. Never overlap another full group.

For a broader task miss, report per-scenario failure and preserve the useful
baseline; do not demand every narrow baseline gate as prerequisite to exploration.
For safety/runtime failures stop and diagnose. After the one pack screen STOP
and email its existing analysis; user reviews whether full capabilities warrant
multi-seed confirmation or targeted ablation. No unbounded experiments, no package
changes, and all launch/sync operations use existing tools under AGENTS.md.

## Frozen provisional diagnostic gates (before screen launch)

Plan: `operation_capability_pack_screen_v1.json`. Warm start all four from
confirmed seed2104 `2026-09-14_07-00-19/model_4999.pt`, same seed2301,4096envs
each,5000 NEW iterations each. Runner resumes policy+optimizer+iteration4999;
new environments start command curricula at common_step_counter0. Expected
final iteration9998; inspect provenance rather than assume filename4999.
Gentle/balanced/fast compare wrist+velocity curriculum timing; low doubles ground
mix probability. Strategy comparisons do not isolate every parameter causally.

Use final500iteration conditional TensorBoard averages. Provisional acceptance:
episode length>=580; continuous_fraction>=.02, continuous_moving_fraction>=.01;
continuous commanded speed>=.008m/s; worst-hand mean position<=.05m,
orientation<=.35rad, shoulder<=.06m; mean episode peak wrist error<=.15m;
actual projected wrist speed / command speed>=.35; wrist velocity error<=.20m/s.
Lift and place fractions each>=.001; nonzero transport/adjust/balance and bilateral
moving intersection counts. Static hold wrist error<=.03m where reported;
backward lean<=.01rad where reported. Missing required metrics mean unverified,
not a pass. These broader NEW task gates do not retrospectively change old gates.
Rank wrist/shoulder accuracy and stability before body velocity compliance.
Ground anchor targets remain0.16--0.28m; dynamic lifts intentionally rise above
anchor range, shoulder lift is smooth and no greater than either wrist lift.
No p95, lag, object contact success or exhaustive IK/collision proof is available.
Do not claim per-lift error gates from lift counts alone. Manual review is needed.

After this ONE group, analyze all four, append existing summary, email via watcher
and write STOP_REQUESTED. No automatic confirmation or additional group.
