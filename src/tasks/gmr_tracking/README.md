# Train on one GMR G1 trajectory

Task: `Unitree-G1-GMR-Tracking`. This task trains on exactly one reference file,
using the existing future-frame teacher/PPO implementation. It does not scan a
directory or mix clips. The default training budget is 20 PPO updates.

The bundled default is the shorter of the two current Pico exports:
`~/GMR/outputs/pico/trackingData_20261005_120456_unitree_g1.npz`.
The exact source is copied to `data/pico_original.npz`, its JSON metadata to
`data/source_metadata.json`, and the converted reference to `data/pico_50hz.npz`.
Training and playback do not depend on the external GMR directory.

The source has 885 frames at 60 Hz, covering 14.733 seconds. Conversion maps GMR
root quaternions from xyzw to MuJoCo wxyz, reorders joints by name, resamples at
50 Hz and computes body FK and velocities using the installed mjlab G1 model.
The converted archive has 788 frames through 15.74 seconds, including about one
second of terminal hold. Episode length is reference duration plus 0.2 seconds.
There is no reference repair or additional ground-height adjustment.

This clip has no contact labels. Unlike the kneeling task, current contact-phase
and future contact-target inputs are omitted, giving actor/critic widths of
507/633. Knee/shin contacts are still measured in simulation. All resets use the
first recorded pose; future reference offsets span 0 to 2 seconds. Rewards,
collision geometry, actuators and PPO settings otherwise follow the nominal
kneeling teacher. The task permits physical tracking deviations and holds the
endpoint without teleporting. Random-reference initialization and deployment
distillation are not implemented. A kneeling-task checkpoint is incompatible
with this task's observation widths; train a fresh policy.

## Small local validation

```bash
cd ~/unitree_rl_mjlab
conda activate unitree_rl_mjlab
python -m src.tasks.kneel_tracking.validate \
  --task Unitree-G1-GMR-Tracking --num-envs 4
python scripts/train.py Unitree-G1-GMR-Tracking \
  --gpu-ids '[0]' --env.scene.num-envs=32 \
  --agent.max-iterations=20 --agent.logger=tensorboard \
  --agent.run-name=pico_single_smoke --enable-nan-guard=True
```

This uses one GPU: 32 environments per rank and 32 globally. Logs/checkpoints
are stored in `logs/rsl_rl/g1_gmr_tracking/`.

Validated locally on 2026-10-05 with the existing miniconda environment and RTX
4070. Four environments traversed all 788 reference frames over 798 steps with
finite rewards/observations, no endpoint reset, and successful explicit reset.
The normal task completed 20 PPO updates and saved `model_19.pt`; all 45
TensorBoard scalar series were finite. Final value/surrogate losses were
0.1596/-0.0410 and mean episode length was 39.69 steps. This confirms training
runs, not that the policy reproduces the full recorded motion. No large-scale
training was launched. Original kneeling lifecycle/observation widths and its
converted reference arrays also passed regression checks. The other Pico file
was converted and its selection/episode-length configuration verified.

The smoke run is retained at
`logs/rsl_rl/g1_gmr_tracking/2026-10-05_12-25-43_pico_single_smoke_20261005/`.

## Larger training and checkpoint playback

For the user's RTX 4090, start with 2048 environments per rank and globally on
one GPU. Actual memory use and throughput have not been validated on that card.
Run long training in a local tmux session:

```bash
python scripts/train.py Unitree-G1-GMR-Tracking \
  --gpu-ids '[0]' --env.scene.num-envs=2048 \
  --agent.max-iterations=10000 --agent.save-interval=500 \
  --agent.logger=tensorboard --agent.run-name=pico_single_2048 \
  --enable-nan-guard=True
```

```bash
python scripts/play.py Unitree-G1-GMR-Tracking \
  --checkpoint-file /path/to/model_9999.pt \
  --num-envs 1 --device cuda:0 --viewer native
```

Use `--viewer viser` for a machine without a desktop display.

## Select another single trajectory

Convert the other Pico export (or another GMR G1 NPZ of the same schema):

```bash
python src/tasks/kneel_tracking/convert_motion.py \
  --source ~/GMR/outputs/pico/trackingData_20261005_120431_unitree_g1.npz \
  --output src/tasks/gmr_tracking/data/pico_other_50hz.npz

export UNITREE_GMR_MOTION_FILE="$PWD/src/tasks/gmr_tracking/data/pico_other_50hz.npz"
```

Then use the same training/play commands above. Keep this variable set to the
same converted file for training and playback; it also determines episode
length. Unset it to restore the bundled default. Raw GMR NPZ archives must be
converted first. The converter reads NPZ with `allow_pickle=False`; PKL export
is not required. The converted archive records source/model SHA-256 hashes.

The original kneeling task remains available with its existing observation
layout and default reference.
