# Autonomous G1 tuning task

Read AGENTS.md first and obey all repository/environment restrictions.

A remote Unitree G1 training experiment has just finished.

The completed trial name is supplied in the task.

Your job is to:

1. Inspect the completed remote training run.
2. Check its exit code first.
3. Read the remote training log.
4. Inspect TensorBoard metrics and checkpoints when available.
5. Determine whether the experiment:
   - completed successfully,
   - crashed due to code/configuration,
   - crashed due to infrastructure,
   - diverged or produced NaNs,
   - or trained normally but underperformed.
6. Compare it against the previous useful baseline when possible.

If the experiment crashed:
- diagnose the root cause first;
- do not blindly tune RL parameters;
- do not change CUDA, PyTorch, Warp, MuJoCo, NCCL, Conda, NVIDIA libraries, SSH, or Kubernetes configuration.

If the experiment trained successfully:
- formulate one specific hypothesis for improvement;
- prefer changing one conceptual factor at a time;
- consider velocity tracking, gait quality, falls, smoothness, action acceleration, foot slip, and robustness rather than total reward alone.

When making a change:
- modify the local repository;
- inspect git diff;
- synchronize using ./tools/remote_sync.sh;
- launch through ./tools/remote_train.sh;
- use a unique descriptive tmux session name.

For risky source-code or reward changes, perform a small validation run first.

Do not start another full 4-GPU run if one is already active.

Do not install or upgrade packages.

If there is not enough evidence to justify another experiment, stop and report why instead of launching one.