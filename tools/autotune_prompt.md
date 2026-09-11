# Autonomous G1 tuning task

Read AGENTS.md first and obey all repository/environment restrictions.

Read `experiments/wrist_recovery_training_plan.md` and
`experiments/autotune_journal.md` before interpreting results or choosing the
next stage. The plan defines the fixed held-out protocol and promotion gates.

A remote Unitree G1 training experiment or registered experiment group has just
finished.

The completed trial name is supplied in the task.

This is one completion-triggered Codex turn. Do not poll, sleep, wait for, or
monitor a newly launched run in this turn. The shell watcher will invoke a new
Codex turn only after that run writes its DONE marker. Finish promptly after
either launching the next justified run or recording a stopping decision.

Token budget for this turn: 8,000 tokens. Use targeted log tails, metric
extraction, and focused source reads. Do not re-read the whole repository or
reconstruct project history already recorded in the journal.

This watcher normally runs GPT-5.6 Terra with medium reasoning. It may perform
routine result extraction, failure diagnosis, held-out comparison, and a
single-factor experiment already justified by the training plan. If the next
step requires redesigning the policy architecture, observation interface,
reward system, curriculum stage, or acceptance criteria, do not improvise the
design or launch another run. Record the evidence and recommended decision in
the journal, then stop so the interactive GPT-5.6 Sol session can review it.

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
   For a group, inspect every member and compare siblings together. A
   same-seed parameter screen may rank candidates, but cannot establish a final
   best policy; promote its winner to independent-seed confirmation.
7. For a completed full run, run the fixed held-out evaluator on both the
   candidate and retained baseline (or reuse results produced by the identical
   evaluator version, suite, seeds, and settings). Do not select a checkpoint
   from training TensorBoard metrics alone, and do not change held-out seeds,
   scenarios, or thresholds to favor a candidate.
   Use `tools/remote_evaluate_wrist.sh` with a unique evaluation name for a
   remote checkpoint; it copies the evaluator artifacts back to local results.
8. Update `experiments/autotune_journal.md` with the hypothesis, exact change,
   validation/full-run status, result, baseline comparison, and next decision.

If the experiment crashed:
- diagnose the root cause first;
- do not blindly tune RL parameters;
- do not change CUDA, PyTorch, Warp, MuJoCo, NCCL, Conda, NVIDIA libraries, SSH, or Kubernetes configuration.

If the experiment trained successfully:
- formulate one specific hypothesis for improvement;
- prefer changing one conceptual factor at a time;
- consider velocity tracking, gait quality, falls, smoothness, action acceleration, foot slip, and robustness rather than total reward alone.
- do not claim a 1--3% improvement from one training seed unless the fixed
  held-out comparison is consistent; require independent training seeds when
  the decision remains within likely stochastic variation.

When making a change:
- modify the local repository;
- inspect git diff;
- synchronize using ./tools/remote_sync.sh;
- launch through ./tools/remote_train.sh;
- use a unique descriptive tmux session name.

For risky source-code or reward changes, first run a 1-GPU validation with 256
environments. Only after it initializes, produces finite rewards, and starts PPO
successfully may you launch a full 4-GPU run. Remember that
`--env.scene.num-envs` is per rank: use 1,024 per rank for 4,096 environments
globally, and report both counts. Validation runs do not count toward the
five-full-run budget.

Do not start another full 4-GPU run if one is already active.

Stop instead of launching another experiment when any of these applies:
- all requested acceptance metrics have been met;
- five full training runs have been completed;
- three consecutive full runs fail to improve the key wrist/balance metrics;
- the next step requires a material design decision from the user;
- evidence is insufficient to justify a controlled next experiment.

Never launch a run merely to keep the loop alive. When stopping, write the
reason and the best checkpoint/result to the journal. Also write a concise stop
reason to `.autotune/STOP_REQUESTED`. The shell watcher will mark the completed
trial handled and then exit cleanly instead of remaining resident.

Do not install or upgrade packages.

If there is not enough evidence to justify another experiment, stop and report why instead of launching one.
