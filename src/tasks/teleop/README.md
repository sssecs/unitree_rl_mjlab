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
and loops it. You can change:

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

For an 8.0 s clip with the default 0.8 s pre-roll:

```text
0.8 s warm-up + 8.0 s motion
```

The first full NPZ target frame is held for one control observation before the
motion clock begins advancing.

`command_finished` is disabled while `warmup_active` is true, which also makes
random within-clip starts (`sampling_mode="uniform"`) safe.

To disable pre-roll and recover the previous behavior:

```python
teleop_cmd.warmup_duration_s = 0.0
```
