# `src/tasks/teleop`: privileged absolute-state teacher

This version is deliberately a **teacher-policy task**. Its first goal is to
learn accurate sparse whole-body tracking in simulation; deployable sensing is
deferred to a later student/distillation stage.

## Core change

The command is no longer represented relative to the robot's current shoulder
frame.

At the beginning of each episode, one fixed planar transform is computed:

```text
NPZ command world -- fixed SE(2) --> episode world
```

The transform aligns the command shoulder midpoint and shoulder-derived heading
to the **post-reset simulated G1 shoulder midpoint and heading** exactly once.

After that, the transform is frozen for the entire episode. Robot motion never
changes the command target.

For the sampled initial command frame:

```text
delta_yaw =
    sim_shoulder_heading_w(0)
    - npz_shoulder_heading_cw(0)

translation_xy =
    sim_shoulder_mid_xy_ew(0)
    - Rz(delta_yaw) * npz_shoulder_mid_xy_cw(0)
```

Then for every time `t`:

```text
cmd_position_ew(t)
    = Rz(delta_yaw) * npz_position_cw(t)
    + [translation_x, translation_y, 0]

cmd_wrist_quat_w(t)
    = q_align * npz_wrist_quat_cw(t)

cmd_shoulder_heading_w(t)
    = npz_shoulder_heading_cw(t) + delta_yaw
```

Z is **not** shoulder-aligned. NPZ ground `z=0` stays episode ground `z=0`.
Thus the human->G1-scaled shoulder/wrist heights are preserved.

## Why `_ew` instead of raw simulator `_w` for actor positions?

The task can run thousands of vectorized environments, each with a different
static `env_origin`. Feeding raw simulator-world XY would inject the arbitrary
environment-grid position into the network.

Therefore actor positions use:

```text
_ew = simulator world - static env_origin
```

This is still an **absolute, robot-independent, fixed frame**. It never follows
the robot. For a single env whose `env_origin` is zero, `_ew == _w`.

## Naming

```text
npz_*   source arrays loaded from NPZ
cmd_*   current fixed-world command target
sim_*   current MuJoCo measurement

_cw     NPZ command world
_ew     fixed per-environment world
_w      MuJoCo global simulator world
```

Examples:

```text
npz_left_wrist_pos_cw
cmd_left_wrist_pos_ew
cmd_left_wrist_pos_w
sim_left_wrist_pos_ew
sim_left_wrist_pos_w
```

There is no `_s` robot-relative task command in this teacher version.

## Teacher actor observations

The actor receives both the 24-D absolute command and the matching 24-D
privileged simulated task state.

Command:

```text
left wrist absolute position_ew               3
left wrist absolute orientation, rotation-6D  6
right wrist absolute position_ew              3
right wrist absolute orientation, rotation-6D 6
shoulder-mid absolute XY_ew                    2
left shoulder height                           1
right shoulder height                          1
shoulder heading vector [cos, sin]             2
                                                --
                                                24
```

Simulated task state has the same layout, but contains current MuJoCo values.

The actor additionally receives joint positions, joint velocities, IMU angular
velocity, projected gravity, and previous action.

This simulated absolute task state is privileged teacher information and is not
assumed to be available on the final real robot.

## Rewards

All task rewards compare absolute targets to absolute simulated states:

```text
cmd wrist world position       vs sim wrist world position
cmd wrist world orientation    vs sim wrist world orientation
cmd shoulder-mid world XY      vs sim shoulder-mid world XY
cmd shoulder heights           vs sim shoulder heights
cmd shoulder heading           vs sim shoulder heading
```

The wrist task remains stronger than the body-placement preferences.

## Reset randomization

Training resets randomize:

```text
x   in [-1, 1] m
y   in [-1, 1] m
yaw in [-pi, pi]
```

The NPZ is then aligned once to the reset shoulder frame. This makes the same
clip appear at many absolute placements/headings and reduces the incentive to
memorize one fixed world trajectory.

Play mode uses deterministic zero XY/yaw reset for easier visual debugging.

## mjlab v1.2 reset detail

`mjlab==1.2.0` runs command-manager reset before the reset qpos has been
forwarded through MuJoCo kinematics. Therefore this implementation intentionally
does **not** read `body_link_pos_w` inside `_resample_command()`.

Instead it marks alignment as pending and resolves it only after MuJoCo
`forward()`:

- explicit `env.reset()`: lazily when `command` is requested for the first
  post-reset observation;
- auto-reset inside `step()`: in `_update_command()` after `forward()`.

A just-reset environment is not advanced from `t=0` on that same update.

## MuJoCo debug visualization

The viewer shows:

```text
episode_world frame

cmd_left_wrist                  pastel frame
sim_left_wrist                  RGB frame
cmd_right_wrist                 pastel frame
sim_right_wrist                 RGB frame

cmd_shoulder_mid_frame          pastel frame
sim_shoulder_mid_frame          RGB frame

sim_torso_link                  optional stabilization frame
```

Position-error arrows are drawn from `sim_*` to `cmd_*`.

## Run

Copy the folder to the repository:

```bash
cp -r teleop /path/to/unitree_rl_mjlab/src/tasks/
```

Select the converted NPZ:

```bash
export UNITREE_TELEOP_COMMAND_FILE=/absolute/path/to/episode.npz
```

Smoke test:

```bash
python scripts/train.py Unitree-G1-Teleop \
  --env.scene.num-envs=32
```

Then scale up:

```bash
python scripts/train.py Unitree-G1-Teleop \
  --env.scene.num-envs=4096
```

The PPO experiment name is now `g1_teleop_teacher`.

## Later student policy

The intended next stage is distillation into a policy that receives only
deployable robot observations (joint encoders, IMU, available state estimator,
VR calibration/history, etc.). This package intentionally does not constrain
the teacher to those sensors.


## Multi-motion recursive NPZ sampling

Training can now use a directory tree instead of one episode.

```bash
export UNITREE_TELEOP_COMMAND_DIR=/absolute/path/to/g1_npz_dataset

python scripts/train.py Unitree-G1-Teleop \
  --env.scene.num-envs=4096
```

The task recursively scans every descendant `*.npz`:

```text
g1_npz_dataset/
├── task_a/
│   ├── episode_000.npz
│   └── episode_001.npz
├── task_b/session_2/
│   └── episode_127.npz
└── task_c/
    └── episode_423.npz
```

At every environment reset:

```text
env 0 -> randomly sample one motion id
env 1 -> randomly sample one motion id
...
env N -> randomly sample one motion id
```

Sampling is independent and with replacement, so thousands of vectorized
environments can train on different episodes simultaneously.

This follows TWIST's core `MotionLib` pattern, but preserves our NPZ timestamps
instead of assuming one fixed source FPS.

### Sampling weights

Default:

```python
teleop_cmd.motion_sampling_weight_mode = "uniform"
```

Every valid NPZ has equal probability per reset.

Optional:

```python
teleop_cmd.motion_sampling_weight_mode = "duration"
```

Longer clips then receive proportionally more probability.

### Start time inside the selected clip

Recommended teacher setting:

```python
teleop_cmd.sampling_mode = "start"
```

Every sampled episode starts from its first command frame. This is the current
default because the robot itself resets to a nominal state.

Optional random-state initialization:

```python
teleop_cmd.sampling_mode = "uniform"
```

This samples a random timestamp inside the selected clip, analogous to the
random reference-time option in TWIST. It is intentionally not the default.

### Single-file compatibility

The old workflow still works:

```bash
export UNITREE_TELEOP_COMMAND_FILE=/absolute/path/to/episode.npz
```

If both `UNITREE_TELEOP_COMMAND_DIR` and `UNITREE_TELEOP_COMMAND_FILE` are set,
the directory takes precedence.

### Play/debug one motion

Play mode fixes `motion_id=0` (the first file in sorted recursive path order)
and repeatedly runs `warm-up -> motion -> recovery -> next cycle` without a
physics reset. You can change:

```python
teleop_cmd.fixed_motion_id = 17
```

to inspect a specific motion.

### Validate the directory before training

```bash
python -m src.tasks.teleop.tools.inspect_npz_dataset \
  /absolute/path/to/g1_npz_dataset
```

Use `--show-files` to print every valid file.

Invalid/non-command NPZ files are skipped by default with warnings. Set:

```python
teleop_cmd.skip_invalid_files = False
```

for fail-fast behavior.

### Memory behavior

All valid sparse command clips are flattened and preloaded to the training
device once. There is no `[num_motion, max_length, ...]` padded tensor; storage
is ragged via `motion_start_idx + motion_num_frames`.

This is appropriate for EgoDemo-scale training. A much larger future corpus
should use sharding/on-demand caching instead.


## Episode command warm-up / pre-roll

A sampled motion no longer starts with an instantaneous jump from the G1
nominal reset pose to the NPZ target.

Default:

```python
teleop_cmd.warmup_duration_s = 0.8
teleop_cmd.warmup_profile = "smoothstep"
```

The reset sequence is:

```text
reset robot
  ↓
sample one NPZ motion
  ↓
compute ONE fixed episode SE(2) alignment
  ↓
snapshot post-reset simulated task pose
  ↓
0.8 s pre-roll:
    cmd = blend(sim_reset_pose, aligned_npz_start_pose)
    command_time = frozen
  ↓
alpha = 1
  ↓
normal NPZ playback:
    cmd = aligned_npz(command_time)
    command_time += step_dt
```

The source/reference layers are intentionally separate:

```text
_npz_*_cw
    raw current frame sampled from the NPZ

aligned_npz_*
    raw NPZ frame after the fixed episode SE(2) alignment

cmd_*
    target actually consumed by policy/reward
    (pre-roll blend during warm-up, then exactly aligned_npz_*)

sim_*
    actual MuJoCo task-space state
```

For positions/heights, warm-up uses interpolation

```text
cmd = (1 - alpha) * sim_reset + alpha * aligned_npz
```

For wrist orientation it uses shortest-path normalized quaternion
interpolation. Shoulder heading uses the shortest wrapped yaw displacement.

The smoothstep profile is:

```text
u = clamp(warmup_time / warmup_duration_s, 0, 1)
alpha = 3*u^2 - 2*u^3
```

so the command target begins and ends the pre-roll with zero blend velocity.

### Reward behavior

Tracking rewards are **not** ramped down. Physical penalties and tracking
terms are all active from the first step.

Because the first emitted command is the post-reset simulated task pose,
tracking error starts near zero and the target itself moves gradually toward
the selected motion.

### Motion time behavior

The NPZ motion clock is frozen during warm-up.

Before post-motion recovery is considered, an 8.0 s clip with the default
0.8 s pre-roll contributes:

```text
0.8 s warm-up + 8.0 s recorded motion
```

With the current recovery defaults, the complete scheduled command is:

```text
0.8 s warm-up + 8.0 s motion + 1.0 s recovery + 0.4 s neutral hold
```

The first full NPZ target frame is held for one control observation before the
motion clock begins advancing.

With `post_motion_behavior="recover"`, `command_finished` is emitted only after
the recovery blend and neutral hold complete. Random within-clip starts
(`sampling_mode="uniform"`) therefore remain safe as well.

To disable pre-roll and recover the previous behavior:

```python
teleop_cmd.warmup_duration_s = 0.0
```

## Episode metrics, recovery, and exhaustive evaluation

### True trajectory-level training metrics

The command term now accumulates tracking error over the **recorded NPZ motion
phase only**. Warm-up and post-motion recovery are intentionally excluded.
At reset, TensorBoard receives both time-mean and max error for the completed
episode:

```text
Metrics/teleop/episode_mean_left_wrist_pos_error
Metrics/teleop/episode_max_left_wrist_pos_error
Metrics/teleop/episode_mean_right_wrist_pos_error
Metrics/teleop/episode_max_right_wrist_pos_error

Metrics/teleop/episode_mean_left_wrist_ori_error
Metrics/teleop/episode_max_left_wrist_ori_error
Metrics/teleop/episode_mean_right_wrist_ori_error
Metrics/teleop/episode_max_right_wrist_ori_error

Metrics/teleop/episode_mean_shoulder_mid_xy_error
Metrics/teleop/episode_max_shoulder_mid_xy_error
Metrics/teleop/episode_mean_shoulder_heading_error
Metrics/teleop/episode_max_shoulder_heading_error

Metrics/teleop/episode_mean_left_shoulder_height_error
Metrics/teleop/episode_max_left_shoulder_height_error
Metrics/teleop/episode_mean_right_shoulder_height_error
Metrics/teleop/episode_max_right_shoulder_height_error

Metrics/teleop/episode_motion_completion_ratio
Metrics/teleop/episode_motion_completed
Metrics/teleop/episode_recovery_started
Metrics/teleop/episode_recovery_completed
Metrics/teleop/episode_tracking_steps
Metrics/teleop/episode_motion_id
```

For every error quantity there is also an `episode_final_*` scalar. On a
normal recovered episode this is the final neutral-hold error; on an early
fall it is the last pre-reset error snapshot.

The older `cmd_vs_sim_*` metrics are kept as instantaneous debug metrics.

### Return-to-neutral recovery

Training now defaults to:

```python
teleop_cmd.post_motion_behavior = "recover"
teleop_cmd.recovery_duration_s = 1.0
teleop_cmd.recovery_hold_s = 0.4
teleop_cmd.recovery_profile = "smoothstep"
```

The full episode becomes:

```text
post-reset nominal task pose
        │
        ├── 0.8 s warm-up ──► NPZ first frame
        │
        ├── recorded NPZ trajectory
        │
        ├── 1.0 s recovery ─► nominal upright/arm pose
        │                      at FINAL commanded shoulder XY/heading
        └── 0.4 s neutral hold ─► command_finished
```

The nominal recovery **shape** is captured immediately after robot reset, but
it is not forced back to the episode's original world position.  At the end of
the motion, that nominal pose is rigidly yaw/XY transformed into the motion's
**final commanded shoulder-mid XY and heading**.  Thus legitimate locomotion
and turning are preserved, while wrist pose and shoulder heights return to the
nominal upright/arm configuration.

Crucially, this recovery target is computed entirely from the reset nominal
pose plus the **commanded** final shoulder frame. It is **not** constructed
from the robot's actual end state. Therefore ending a motion in a poor or
contorted configuration cannot move the recovery target toward the robot and
erase the recovery error.

This is intended to discourage irreversible tracking shortcuts. Tracking
rewards remain active during recovery, as do upright, joint-limit,
self-collision, action-rate, and other physical terms.  In addition, a small
`recovery_joint_posture` penalty (weight `-0.10`) is active only after the
recorded trajectory ends; it pulls redundant joints toward the G1 default
standing posture without constraining the motion-tracking phase itself.

Available behaviors:

```python
# Old behavior: end as soon as the NPZ reaches its final frame.
teleop_cmd.post_motion_behavior = "terminate"

# Default: return to nominal, hold, then end episode.
teleop_cmd.post_motion_behavior = "recover"

# Continuous training: recover to nominal and then sample another NPZ without
# resetting MuJoCo.  Useful later for long-horizon robustness training.
teleop_cmd.post_motion_behavior = "recover_then_chain"
```

Interactive `play=True` uses `recover_then_chain` with motion 0, so the viewer
repeats warm-up -> motion -> recovery without a physics reset.

### Evaluate every motion once

The package contains an exhaustive benchmark runner:

```bash
python -m src.tasks.teleop.tools.evaluate_all_motions \
  --checkpoint-file logs/rsl_rl/g1_teleop_teacher/RUN/model_10000.pt \
  --command-dir /absolute/path/to/all_npz \
  --batch-size 32 \
  --output-dir eval/model_10000
```

It recursively enumerates the same valid/sorted NPZ library used for training,
assigns a distinct fixed motion id to each vectorized environment, and keeps
batching until every motion has been executed once.

Evaluation disables:

```text
actor observation corruption
pushes
friction randomization
encoder bias randomization
COM randomization
random reset XY/yaw
```

and always uses:

```text
sampling_mode = start
post_motion_behavior = recover
loop = false
```

Outputs:

```text
eval/model_10000/
├── per_motion.csv
└── summary.json
```

`per_motion.csv` contains, for every NPZ:

```text
motion id / relative path / duration
motion completion ratio
motion completed
recovery completed
fell over
tracking steps
mean + max trajectory errors
final recovery/neutral errors
episode reward sum
```

`summary.json` contains equal-motion aggregate statistics plus the worst 20
motions ranked by mean wrist-position error.


## v5: wrist reward transfer + velocity + quiet feet + knee support + history

Actor and critic now use:

```python
history_length = 3
```

and each frame adds:

```text
command_wrist_lin_vel_absolute   6
sim_wrist_lin_vel_absolute       6
```

The existing 24-D absolute pose command remains unchanged.  Because both the
per-frame observation dimension and history changed, v4 checkpoints are not
actor-shape compatible with v5.

Wrist position shaping is now:

```text
wrist_pos_coarse  +3.0, exp(-RMS_error / 0.20)
wrist_pos_fine    +4.0, 1 - tanh(RMS_error / 0.05)
inter_wrist       +0.5, exp(-relative_error / 0.10)
wrist_velocity    -0.02, bilateral squared linear-velocity error
```

Command wrist velocity is finite-differenced from the exact emitted target
(after warm-up/recovery shaping). Alignment/recenter epoch jumps are explicitly
zeroed rather than exposed as target motion.

Physical-quality terms added:

```text
feet_slide                  -0.15
quiet_feet                  -0.50
undesired_ground_contacts   -1.0
termination               -200.0
```

`quiet_feet` is gated and activates only for an already-tracked quasi-static
task: wrist error <4 cm, wrist target speed <5 cm/s, shoulder-mid target speed
<3 cm/s, shoulder heading rate <0.15 rad/s, base XY speed <0.12 m/s, and base
angular speed <0.35 rad/s. Warm-up is excluded.

Kneeling is explicitly allowed. Feet and the collision geoms attached to the
two G1 knee links are excluded from the undesired-ground contact sensor:

```text
left_shin_collision
left_linkage_brace_collision
right_shin_collision
right_linkage_brace_collision
```

The stock G1 MJCF has no separate patella collision geom, so this permits
knee/shin-link support. If later we need "patella allowed, shin forbidden", the
MJCF needs dedicated knee-pad collision geoms.


## v5.1 evaluator performance fix

The exhaustive evaluator previously executed Python conditions on CUDA tensors
on every environment step:

```python
if not torch.any(active_env):
...
if not torch.any(newly_done):
...
```

Those branches force CUDA-to-CPU synchronization and serialize an otherwise
asynchronous GPU rollout.

The v5.1 evaluator keeps done/metric state on GPU and synchronizes only every
`--progress-interval-steps` (default 250) for status output, plus once at the
end of each batch.

It also sorts motions by duration before batching by default.  This reduces
wasted simulation caused by short clips sharing a batch with a very long clip.

Useful options:

```bash
--batch-size 256
--progress-interval-steps 250
--preserve-motion-order
```

The evaluator prints motion-duration statistics and an estimated batching
padding ratio.  A high padding ratio means smaller duration-bucketed batches can
be faster than putting every motion in one batch, even when the larger batch has
more GPU occupancy.


## v6: explicit causal task-error vector + 25-frame history

The teacher now receives an explicit **24-D current target-minus-state error
vector** in addition to the existing absolute command and simulated task state:

```text
left wrist position error (_ew)        3
right wrist position error (_ew)       3
left wrist rotation-vector error       3
right wrist rotation-vector error      3
left wrist linear-velocity error       3
right wrist linear-velocity error      3
shoulder-mid XY error                  2
left/right shoulder-height error       2
shoulder heading [cos,sin] error       2
                                        --
                                        24
```

Wrist orientation uses:

```python
quat_box_minus(target_quat, measured_quat)
```

which gives a shortest/sign-invariant axis-angle error vector.  Heading uses
the difference between target and measured `[cos(yaw), sin(yaw)]` vectors, so
the representation is continuous through +/-pi.

### Causal contract

`task_error_vector` uses only the **current** emitted teleop command and current
robot state. It does not access a future NPZ sample. History likewise contains
only current and past observations.

### Longer history

The default is now:

```python
history_length = 25
```

At 50 Hz this covers about 0.5 s.

Both `make_teleop_env_cfg()` and `unitree_g1_teleop_env_cfg()` accept an
optional `history_length` argument, so 3/10/25-frame ablations do not require
editing the observation implementation.

### Default observation size

v5/v5.1 actor features per frame: 153  
v6 explicit vector error: +24  
v6 actor features per frame: 177

With 25-frame history:

```text
actor  = 177 * 25 = 4425
critic = 188 * 25 = 4700
```

The critic keeps the previous 8 scalar task-error summary in addition to the
new vector error.

This changes network input shape, so v5/v5.1 checkpoints are **not directly
compatible**. Train a fresh v6 policy.
