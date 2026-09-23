# EgoDex + PICO static-kneel training

This extension is designed for the synthesized dataset produced by the current
EgoDex/PICO pipeline:

```text
PICO stand -> kneel
  -> PICO-to-EgoDex handover
  -> EgoDex manipulation
  -> EgoDex-to-PICO handover
  -> same PICO tail / stand-up
```

The command remains an **absolute fixed-world task**.  The robot is allowed to
change its pelvis, feet and support configuration without moving the wrist
reference frame with its body.

## What changed

The original `Unitree-G1-Teleop` task remains registered and backward
compatible. Two additional tasks are registered:

```text
Unitree-G1-Teleop-Kneel
Unitree-G1-Teleop-Kneel-Baseline
```

The kneel task adds:

- direct loading of `episode.style_candidates.npz`;
- whole-episode selection of the candidate indexed by
  `selected_pico_template_index` in the command metadata;
- optional left/right episode filtering with
  `UNITREE_TELEOP_KNEEL_SIDE`;
- phase-aware wrist task reward scaling from
  `wrist_tracking_weight_scale`;
- static-kneel style reward weight `2.0`;
- a 20% floor on the wrist-error style gate;
- weaker initial wrist-orientation reward (`0.75`);
- explicit left/right knee/shin contact sensors and episode contact fractions;
- knee-aware COM support hull support for later balance experiments;
- no second post-motion recovery in training, because the synthesized clip
  already contains the PICO return-to-stand tail.

The baseline task uses the exact same command trajectories but disables the
human descriptor reward.

## Validate a synthesized dataset

```bash
python -m src.tasks.teleop.tools.inspect_kneel_dataset \
  /absolute/path/to/synth_dataset \
  --show-files
```

## Recommended first experiments

Command-only baseline:

```bash
export UNITREE_TELEOP_COMMAND_DIR=/data/synth
export UNITREE_TELEOP_KNEEL_SIDE=any
python scripts/train.py Unitree-G1-Teleop-Kneel-Baseline \
  --env.scene.num-envs=2048
```

Left-knee style:

```bash
export UNITREE_TELEOP_COMMAND_DIR=/data/synth
export UNITREE_TELEOP_KNEEL_SIDE=left
python scripts/train.py Unitree-G1-Teleop-Kneel \
  --env.scene.num-envs=2048
```

Right-knee style:

```bash
export UNITREE_TELEOP_KNEEL_SIDE=right
python scripts/train.py Unitree-G1-Teleop-Kneel \
  --env.scene.num-envs=2048
```

Equivalent helper:

```bash
bash tools/train_static_kneel.sh /data/synth left descriptor 2048
```

For the first mechanism experiment, keep:

```text
balance_mode = off
push disturbances = off
sampling_mode = start
```

Do **not** mix left/right descriptor supervision into one policy until the two
sides have independently been shown to learn.  The actor does not receive a
human-style label, so identical sparse tasks with contradictory KL/KR reward
branches would otherwise create ambiguous supervision.

## Important metrics

In addition to the existing wrist/shoulder tracking metrics, inspect:

```text
contact/left_knee
contact/right_knee
contact/any_knee

episode/left_knee_contact_fraction
episode/right_knee_contact_fraction
episode/any_knee_contact_fraction
episode/bilateral_knee_contact_fraction

command/phase
command/wrist_task_weight
```

A successful kneel run should not merely lower the pelvis.  The intended
support side should have substantial real knee/shin ground-contact fraction
while wrist tracking remains comparable to the command-only baseline.

## Balance curriculum later

The first static-kneel experiments intentionally keep `balance_mode=off`.
When balance reward is re-enabled, the runtime extension uses the contacting
feet **plus** left/right knee/shin support patches when computing the support
hull.  This avoids the old feet-only balance model incorrectly penalizing a
stable single-knee support configuration.

`reward_gate` also no longer deletes the entire human style reward: it only
attenuates the shape component, with a 25% floor, while torso/knee semantic
style remains active.
