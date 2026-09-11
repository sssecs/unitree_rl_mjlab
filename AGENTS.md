# Unitree G1 RL Training Agent Instructions

This repository is used for Unitree G1 locomotion reinforcement-learning development.

The local workstation is used for:

* editing code
* reviewing diffs
* running Codex
* lightweight analysis

The remote Kubernetes training server is used for:

* GPU training
* TensorBoard logs
* checkpoints
* long-running experiments

## Remote host

SSH alias:

`unitree-trainer`

Remote repository:

`/home/dev/unitree_rl_mjlab`

The remote machine has 4 NVIDIA H20 GPUs.

Do not use Kubernetes APIs directly.

Do not use `kubectl`.

All interaction with the training server must go through the provided scripts in `tools/` or read-only SSH commands when necessary.

---

# Required workflow

Always follow this sequence when modifying and testing training behavior:

1. Inspect the current code and existing experiment results.
2. Form a clear hypothesis for the next change.
3. Modify files locally.
4. Run `git diff` and inspect the change.
5. Synchronize the local repository to the remote server.
6. Run a small validation experiment when appropriate.
7. If validation succeeds, launch the full experiment.
8. Monitor training remotely.
9. Read TensorBoard metrics and training output.
10. Compare against the previous baseline.
11. Keep or revert the change.
12. Only then start the next experiment.

Do not change many unrelated variables in one experiment unless explicitly requested.

Prefer controlled experiments where each run tests a specific hypothesis.

---

# Synchronizing code

After modifying local code, synchronize it with:

`./tools/remote_sync.sh`

Always inspect:

`git diff`

before synchronization.

Never use:

`rsync --delete`

Do not directly edit the remote repository unless there is a strong reason.

The local Git repository is the source of truth.

---

# Starting training

All long-running training must run inside a detached remote `tmux` session.

Never run a long training process directly over an attached SSH connection.

Use:

`./tools/remote_train.sh <session-name> <training arguments...>`

Example full training run:

`./tools/remote_train.sh g1_baseline Unitree-G1-Flat --gpu-ids all --env.scene.num-envs=1024 --agent.logger tensorboard`

`--env.scene.num-envs` is per training process/GPU. Therefore four-GPU DDP
with `num-envs=1024` uses 4,096 environments globally; `num-envs=4096` uses
16,384 globally. Always report both per-rank and total environment counts.

Use descriptive session names, for example:

* `g1_baseline`
* `g1_reward_vel_v2`
* `g1_action_penalty_v3`
* `g1_lr_3e4`
* `g1_curriculum_v2`
* `g1_dr_friction_v3`

Do not reuse the same session name for different experiments.

`remote_train.sh` records the Git commit, dirty status/binary diff, hashes of
all synchronized files, an archive of untracked files, exact arguments,
resolved environment and agent YAML, rank seeds, per-rank/global environment
counts, and the actual TensorBoard log directory under
`.autotune/<session>/`. Formal runs should normally start from a clean committed
tree; this provenance bundle is a safeguard, not a replacement for version
control.

---

# Validation before full training

When testing potentially risky code or configuration changes, start with a small validation run.

Recommended validation configuration:

`./tools/remote_train.sh <session-name> Unitree-G1-Flat --gpu-ids "[0]" --env.scene.num-envs=256 --agent.logger tensorboard`

Inspect the output before launching a full 4-GPU experiment.

A validation run should verify:

* environment initialization
* CUDA/Warp initialization
* no Python exceptions
* no NaNs
* PPO begins learning
* rewards are finite
* simulation is progressing

Do not launch a full 4-GPU run if the small validation run is broken.

---

# Full training

For a normal full DDP training run with 4,096 environments globally use:

`./tools/remote_train.sh <session-name> Unitree-G1-Flat --gpu-ids all --env.scene.num-envs=1024 --agent.logger tensorboard`

Before starting a full run, check the remote status:

`./tools/remote_status.sh`

Do not start another full 4-GPU DDP experiment if one is already using all GPUs.

For controlled screening or multi-seed confirmation, four independent one-GPU
runs may be launched as one registered group with `tools/remote_sweep.py`.
Within a group, use distinct GPUs and analyze all siblings together. Same-seed
screening ranks parameter candidates; final claims require independent training
seeds and/or the fixed held-out evaluator.

---

# Monitoring training

Check remote tmux sessions and GPU utilization with:

`./tools/remote_status.sh`

Read recent output from a training session with:

`./tools/remote_log.sh <session-name>`

For more output:

`./tools/remote_log.sh <session-name> 500`

SSH disconnection is not a training failure.

Training runs inside remote `tmux` and should continue if the workstation loses network connectivity.

If SSH reconnects after an interruption:

1. run `./tools/remote_status.sh`
2. inspect the tmux session
3. inspect recent logs
4. only restart training if the previous process actually stopped

---

# Stopping training

Gracefully stop an experiment using:

`./tools/remote_stop.sh <session-name>`

Prefer graceful interruption.

Do not use `kill -9` unless explicitly necessary.

Do not delete checkpoints after stopping an experiment.

---

# Training results

Remote experiment results are stored under:

`/home/dev/unitree_rl_mjlab/logs/rsl_rl/`

For G1 velocity training, expect results under:

`/home/dev/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/`

Use TensorBoard event files as the primary source for comparing experiments.

Important metrics include:

* mean reward
* episode length
* velocity tracking performance
* termination/fall rate
* action acceleration
* policy loss
* value loss
* entropy
* learning rate
* training throughput

Do not judge an experiment only from terminal output.

Prefer quantitative comparison between experiments.

Do not declare a best policy from small single-seed differences. Use a
same-seed four-way group for parameter screening, then confirm the winner with
at least three independent training seeds and the unchanged held-out evaluator.

---

# Hyperparameter tuning

You may tune:

* learning rate
* entropy coefficient
* PPO clipping
* discount factor
* GAE lambda
* batch/minibatch configuration
* number of steps per environment
* reward weights
* curriculum parameters
* command distributions
* domain randomization
* observation noise
* termination conditions
* policy architecture when justified

For reward tuning, pay special attention to:

* linear velocity tracking
* angular velocity tracking
* posture stability
* action smoothness
* joint acceleration
* foot slip
* foot clearance
* contact behavior
* fall rate

Avoid blindly maximizing total reward.

A higher total reward is not automatically a better locomotion policy.

Consider stability, smoothness, tracking quality, gait quality, and robustness.

---

# Experiment discipline

Every experiment should have:

* a descriptive session name
* a specific hypothesis
* the exact configuration change
* a baseline for comparison

When possible, change one conceptual factor at a time.

Examples:

Good:

`Increase linear velocity tracking reward from 1.0 to 1.5 and compare tracking error.`

Bad:

`Change ten reward terms, PPO learning rate, network size, and domain randomization simultaneously.`

If an experiment performs worse, revert the unsuccessful change using Git rather than manually reconstructing the previous configuration.

---

# Git policy

Before changes:

`git status`

After changes:

`git diff`

Do not destroy uncommitted user changes.

Do not use destructive Git commands unless explicitly requested.

Avoid:

* `git reset --hard`
* `git clean -fd`
* force push

unless explicitly authorized.

Prefer small, reviewable changes.

---

# Environment restrictions

The remote CUDA and Python environment is known to work and must be treated as immutable.

Do not run:

* `pip install`
* `pip uninstall`
* `pip upgrade`
* `conda install`
* `conda update`
* `conda remove`

Do not modify:

* `/home/dev/miniconda3`
* `/home/dev/nvidia-550.127`
* CUDA libraries
* NVIDIA libraries
* PyTorch version
* Warp version
* MuJoCo version
* MuJoCo Warp version
* NCCL configuration
* SSH keys
* SSH configuration
* Kubernetes configuration

Training must always be launched through the existing remote:

`/home/dev/unitree_rl_mjlab/run_train.sh`

The launcher contains required CUDA/NVML compatibility configuration.

Do not bypass it.

---

# Safety and resource limits

Do not:

* launch multiple full 4-GPU experiments simultaneously
* create unbounded experiment loops
* consume arbitrary amounts of disk space
* recursively delete training logs
* delete previous checkpoints
* modify system packages
* use sudo
* attempt privilege escalation

If an experiment repeatedly crashes, inspect the error before restarting it.

Do not repeatedly restart the same failing experiment without identifying the cause.

---

# Codex operating principle

Act as an RL training engineer, not as a package/environment administrator.

Focus on:

* locomotion behavior
* reward design
* PPO configuration
* curriculum
* domain randomization
* experiment analysis

Treat the working CUDA/PyTorch/MuJoCo environment as infrastructure that must not be modified.
