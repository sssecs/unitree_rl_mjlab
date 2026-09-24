# G1 single-knee acquisition curriculum

This patch targets the failure mode where PPO never discovers a single-knee support topology.
It is designed for the current EgoDex+PICO synthesized dataset and the existing kneel motion-library extension.

## What changed

### 1. Robot-specific support reward instead of human knee-height regression

The previous approach reward compared:

```text
G1 knee-link height <-> human knee landmark height
```

with a narrow Gaussian. That is a cross-embodiment mismatch: G1 makes ground contact with the shin/linkage collision geometry while the knee-link center can remain above the human landmark target.

The new support reward uses:

- descriptor dims `15:17` (`knee_pseudo_contact_left/right`) to determine intended support side;
- descriptor knee-ground distance only to smoothly activate the kneeling objective during the human descent;
- G1 knee-link normalized height only as a monotonic dense *approach proxy*;
- actual `left/right_knee_ground_contact` sensors for the final support bonus;
- an asymmetry bonus so a symmetric squat is not the easiest solution;
- a wrong-knee contact penalty.

Default K0 support terms inside the reward are approximately:

```text
1.50 * dense target-knee descent
4.00 * correct shin/knee contact
1.00 * single-knee asymmetry
-1.25 * wrong-knee contact
```

### 2. No large negative torso Huber basin

`kneel_human_style_reward` now uses a bounded positive torso term:

```text
0.5 * (1 + cos(torso_error))
```

The K0 acquisition task uses an even broader positive-only posture reward.

### 3. K0 acquisition task

New task:

```text
Unitree-G1-Teleop-Kneel-Acquire
```

It:

- exposes the target 8-D style descriptor + validity bit to actor/critic;
- starts from `KNEES_BENT_KEYFRAME` to reduce the first exploration barrier;
- strongly prioritizes discovering one-knee support;
- makes wrist tracking weak rather than dominant;
- disables shoulder tracking during K0;
- disables `quiet_feet` during K0;
- relaxes slide/action/joint-limit/fall penalties during acquisition;
- keeps balance reward off.

A second task:

```text
Unitree-G1-Teleop-Kneel-Acquire-Home
```

uses exactly the same observation/reward structure but resets from the normal HOME pose. Use it after the bent-init stage succeeds.

## First run

Train one side only. Do not use `any` for K0.

```bash
bash tools/train_kneel_acquisition.sh /path/to/synth_dataset left bent 2048
```

Then repeat right separately:

```bash
bash tools/train_kneel_acquisition.sh /path/to/synth_dataset right bent 2048
```

## What to watch

The most important metrics are:

```text
contact/left_knee
contact/right_knee
episode/left_knee_contact_fraction
episode/right_knee_contact_fraction
descriptor_knee_contact_target
descriptor_knee_contact_match
descriptor_knee_support_approach
style_robot_pelvis_height
style_human_pelvis_height
```

For a left-knee run, the first positive sign is not wrist tracking. It is:

```text
descriptor_knee_support_approach rises
-> left knee contact begins to appear
-> episode/left_knee_contact_fraction rises
```

while right-knee contact remains low.

If `descriptor_knee_support_approach` improves but actual knee contact stays identically zero, inspect G1 collision/contact geometry before further reward tuning.

If both knees contact together, increase `asymmetry_weight` / `wrong_contact_weight` in `kneel_env_cfg.py`.

## Suggested curriculum

### K0-A: bent initialization

```text
Unitree-G1-Teleop-Kneel-Acquire
```

Goal: discover stable single-knee contact.

### K0-B: HOME initialization

Once the target-knee contact fraction is reliably high, switch to:

```text
Unitree-G1-Teleop-Kneel-Acquire-Home
```

Prefer resuming from the K0-A checkpoint so the observation dimensions remain identical.

### K1: restore task precision

Then continue with:

```text
Unitree-G1-Teleop-Kneel-TargetDescriptor
```

This restores the original wrist/shoulder task weights while retaining the improved support reward and bounded torso reward.

Only after K1 works should KL/KR be mixed or candidate-set autonomous selection be tested.

## Important

This patch assumes the existing current repo already contains:

- `kneel_motion_library.py`
- `kneel_extensions.py`
- `left_knee_ground_contact` / `right_knee_ground_contact` support
- EgoDex+PICO `*.style_candidates.npz` loading

It is intended for the current `codex` line around commit `bb0f66d` (`Adjust reward`).
