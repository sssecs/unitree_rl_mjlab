# G1 kneeling reference teacher

Task: `Unitree-G1-Kneel-Tracking`.

This isolated task reuses the repository's G1 tracking configuration and PPO
network. Its hypothesis is that soft joint/body tracking with future reference
frames can learn a physically feasible approximation to a kinematic kneeling
sequence. Reference repair is deliberately deferred until runtime evidence
justifies it. This is a smoke-tested environment, not a trained kneeling policy.

## Reference data

All motion data lives in this task's `data/` directory:

- `transition_original.npz`: exact copy of
  `~/g1_transition_optimizer/transition_output_v02/transition.npz`.
  The source contains 54 knots over 6.8 seconds and a 29-DoF G1 pose sequence:
  stand, bend, squat, left-knee hover/touchdown, right-knee hover/touchdown,
  upright double kneel, forward low double kneel. Arms remain neutral.
- `kneel_50hz.npz`: 391 frames at 50 Hz, including a 1-second terminal hold.
  Root position and joints use linear interpolation; root orientation uses
  quaternion SLERP. FK uses the installed mjlab G1 model. Joint velocities use
  MuJoCo position differencing; body linear and world angular velocities use
  centered differences, with one-sided differences at endpoints.
- `source_summary.json` and `source_validation.json`: optimizer diagnostics,
  which validate the kinematic reference, not dynamic execution.

The converted archive includes joint/body names, sample times, original contact
modes, coarse foot/knee support targets, source/model SHA-256 hashes, duration
and hold metadata. Foot targets include forefoot support; they do not prescribe
heel contact. Inactive contact-anchor NaNs in the original archive are not used.
Joint/body ordering and the control frequency are checked at environment startup.

Regenerate locally with the existing environment, without installing packages:

```bash
conda activate unitree_rl_mjlab
python src/tasks/kneel_tracking/convert_motion.py
```

## Teacher design

The teacher follows the future-reference/PPO idea in
[TWIST, Section 3.2](https://arxiv.org/html/2505.02833v1) and its
[official implementation](https://github.com/YanjieZe/TWIST).
It is a task-specific adaptation, not a complete TWIST reproduction.

The actor sees reference frames at offsets 0, 0.1, 0.2, 0.5, 1 and 2 seconds.
Each contains joint positions/velocities, root linear/angular velocities in the
reference root frame, projected gravity, root height and coarse contact targets.
Additional inputs are the current torso orientation error, measured base
velocities, joint positions/velocities, previous action and contact phase.
The critic also receives privileged tracked-body positions and orientations.
Actor/critic widths are 536/662. This is a privileged simulation teacher.

Actions retain the existing default-pose offset and G1 per-joint scaling/PD
actuators. They are not reference-pose residuals. Rewards retain soft relative
body pose tracking, torso orientation, self-collision and action-rate costs,
and add joint position/velocity and local root velocity tracking. Absolute root
XY tracking is removed. Body tracking has broad 0.3 m position tolerance;
reference height remains an approximate target. Contact targets enter the
observations and actual shin-ground contacts are logged; virtual knee markers
are not constrained to exactly touch the floor. Joint soft limits are expanded
from 90% to the model's existing hard limits, which remain enforced.

The existing collision geometry is preserved. Reference poses can penetrate
that geometry into the floor (about 4.5 cm at the shin capsule); this remains a
known reference discrepancy. The policy may deviate from it. Both training and
play reset only to the first standing frame, with no perturbations or random
kneel initialization. Noise, pushes and domain randomization are disabled for
this nominal validation. Knee/shin ground contacts are allowed. Reference-relative
height/orientation termination checks remain, with height tolerances of 0.5 m.
There is no separate forbidden-ground-contact termination in this first version.

Episode length is 8 seconds. Reference time clamps at 7.8 seconds and never
teleports the robot at the clip endpoint. A just-reset command stays at frame
zero for its first command update. Student distillation, deployment, robustness
training and reward tuning are outside this change.

## Local validation

```bash
conda activate unitree_rl_mjlab
python -m src.tasks.kneel_tracking.validate --num-envs 4
python scripts/train.py Unitree-G1-Kneel-Tracking \
  --gpu-ids '[0]' --env.scene.num-envs=32 \
  --agent.max-iterations=20 --agent.logger=tensorboard \
  --agent.run-name=kneel_smoke --enable-nan-guard=True
```

The task's default training budget is also 20 iterations. One GPU means 32
environments per rank and 32 globally for the command above. The motion is loaded
from the task directory automatically; no `--motion-file` argument is needed.
Use `--env.commands.kneel.motion-file=PATH` for another archive of the same schema.

Validated on 2026-10-03 with local RTX 4070 12 GB, the existing
`unitree_rl_mjlab` miniconda environment, mjlab 1.2.0, MuJoCo 3.5.0, Warp 1.12.1,
PyTorch 2.14.0+cu130 and rsl-rl-lib 5.0.1. GPU access required running outside
the execution sandbox; no environment packages were changed.

The lifecycle diagnostic temporarily retains only the time-limit termination
and uses zero policy actions. Four environments traversed all 391 reference
frames and 401 steps with finite rewards/observations, no endpoint reset and a
successful explicit reset to frame zero. This verifies reference progression,
not physical motion reproduction.

The standard PPO run used the normal task terminations and finished all 20
updates, producing checkpoints at iterations 0, 10 and 19. All 45 TensorBoard
scalar series were finite. Final value/surrogate losses were 0.3066/-0.0417;
mean episode length was 58.67 control steps (about 1.17 seconds), well short of
the complete motion. Mean reward was -2.3951 and adaptive learning rate reached
1e-5. These measurements establish the training pipeline runs; they do not
establish successful kneeling or an improvement over another policy. No
large-scale training or parameter sweep was performed.

Logs/checkpoints remain under
`logs/rsl_rl/g1_kneel_tracking/2026-10-03_12-44-29_kneel_smoke_20261003/`
and are excluded from Git. The previous locomotion/teleop tasks are unchanged.
