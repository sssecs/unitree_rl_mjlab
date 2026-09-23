# G1 kneeling descriptor teacher screen

## Question

Does giving the privileged teacher the current eight-dimensional target posture
descriptor improve single-knee support and style while preserving wrist tracking?
The extra validity bit makes the actor input nine-dimensional. All reward terms
and the wrist command remain unchanged in the primary style-only vs style-actor
comparison.

## Runs

Wave 1 uses seed 4201 on four independent H20 processes:

| GPU | Variant | Difference from preceding variant |
| --- | --- | --- |
| 0 | `task_only` | Command-only control; descriptor reward weight 0 |
| 1 | `style_only` | Descriptor reward weight 2, hidden target |
| 2 | `style_actor` | Same reward; current target descriptor in actor and critic |
| 3 | `style_actor_w3` | Same observation; descriptor reward weight 3 |

Wave 2 repeats the primary comparison at seeds 4202 and 4203: style-only and
style-actor at each seed, one process per GPU. The primary paired comparison
therefore has three independent seeds (4201, 4202, 4203). Task-only and weight
3 are exploratory one-seed screens, not final claims.

Each run uses one GPU, 1,024 environments per rank and 1,024 total, 24 steps
per environment per PPO iteration, and 2,500 iterations. Four concurrent runs
use 4,096 environments in total. No run uses DDP. Each experiment has its own
remote tmux session and provenance bundle.

## Dataset and launch

All 2,742 validated command/style pairs are used in every run: 1,365 left and
1,377 right kneels. The actor-target comparison therefore also tests whether
the current descriptor disambiguates postures across sides.
Source: `/mnt/hdd/humanoid_locomotion/datasets/egodex_pico_kneel_synthesis/EgoDex-PICO-kneel-synth/test`.
Destination: `/home/dev/EgoDex-PICO-kneel-synth` on `unitree-trainer`.
The full dataset is about 1.3 GiB. `tools/remote_kneel_data_sync.py` transfers
it without deletion and verifies checksums. The dataset and repository must
both be writable by the remote SSH user before launch.

Plans: `g1_kneel_descriptor_seed4201_v1.json` and
`g1_kneel_descriptor_compare_v1.json`. The bounded local two-wave scheduler is
`tools/remote_two_wave_sweep.py`; it starts wave 2 only if all wave-1 sessions
finish with exit code 0. Launch it in detached local tmux after code/data sync.
Remote training itself is always launched by `tools/remote_train.sh` in remote
detached tmux sessions and uses the existing `run_train.sh`.

## Readout

Primary: intended-side knee contact fraction during active motion, opposite-side
knee contact fraction, motion completion, fall rate, and wrist position errors.
Analyze left and right episodes separately as well as together.
Secondary: descriptor components, shoulder errors, episode length, reward,
losses, entropy, learning rate, throughput, and finiteness. Compare paired
seeds at equal iteration and wall time. A higher total reward alone is not a
success criterion. The fixed held-out evaluator is required before selecting
a final policy.

## User launch

After the dataset and code are synchronized, run from the workstation:

```bash
./tools/start_kneel_overnight.sh
```

The script starts a detached local tmux coordinator. It first launches a
remote-tmux validation of the target-descriptor task with one GPU, 256
environments, and three PPO iterations. A failed validation prevents the eight
full runs. Each full run uses a separate detached remote tmux session via
`remote_train.sh`. The workstation must remain online until wave 2 has been
launched; once started, remote training survives workstation disconnects.
The coordinator log is `.autotune/g1_kneel_descriptor_overnight_v1.log`.
