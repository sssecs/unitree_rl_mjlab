# RETIRED: watcher no longer invokes Codex

Since 2026-09-15, `autotune_watch.sh` only extracts deterministic TensorBoard
values and emails JSON. This prompt is retained as experiment history and is not
read by the watcher. No completion-triggered model analysis or next-run decision
is performed.

# Historical autonomous G1 tuning task

Read AGENTS.md first and obey all repository/environment restrictions.

LATEST for `g1_persistent_operation_*`: read
`experiments/persistent_operation_next.md` and its materialized JSON. ONE four-way
screen only. Analyze all siblings, report conditional persistent/FD/contact
metrics and censored/absent coverage honestly; email existing summary via watcher,
write STOP_REQUESTED and STOP. Do NOT launch confirmation or any further run.
This overrides all older stage promotion rules for this group.

LATEST less-conservative user revision for continuous-stage completions: read
`experiments/operation_capability_pack_next.md`. The running3cm baseline and its
confirmation were CANCELLED by explicit user request. Do not relaunch them.
Implement+validate the rich
capability pack, freeze its gates, then at most ONE four-way strategy screen;
stop if implementation/validation is incomplete and stop after the pack screen.
Do not use unimplemented flags or claim one-factor causal effects for strategy
comparisons. Multi-seed final claims remain required, not every tiny extension.

For `g1_operation_capability_pack_*` completions, the latest pack plan overrides
all older stage promotion/confirmation instructions below. Analyze this ONE
registered screen, report provisional candidate and limitations, then write
STOP_REQUESTED and exit. NEVER launch another group automatically.

For `g1_continuous_wrist_*` completions, use latest explicit authorization
`experiments/continuous_wrist_next.md`: continuous POSITION Stage6A1 now has
implementation and predeclared gates, but its screen/confirmation budget is now
CANCELLED by the latest user request. Old future-plan
statements saying the generator is unimplemented no longer apply to this subset.
Do not automatically add mobility/vertical/rotation/EgoDex trajectories.

For `g1_bilateral_transport_*` completions, latest user authorization is
`experiments/bilateral_transport_next.md`: follow its two-group speed-envelope
screen/confirmation budget and intersection gates. It overrides the completed
stationary bilateral stop only for this stage; no continuous-motion advance.

Read `experiments/wrist_recovery_training_plan.md` and
`experiments/autotune_journal.md` before interpreting results or choosing the
next stage. The user makes the final policy judgment; automated results are
provisional engineering evidence.

For `g1_ground_leg_smooth_*` completions, read
`experiments/ground_leg_smooth_next.md`: its new two-group budget supersedes the
completed workspace stop only for the isolated leg-jitter screen/confirmation.
Do not launch continuous-command training within that budget.

Latest user authorization: read `experiments/bilateral_ground_next.md` for
`g1_ground_leg_smooth_confirm_v1` and `g1_bilateral_ground_*` completions. It
overrides the old post-leg-confirmation stop ONLY after unchanged confirmation
gates AND the new bilateral smoke pass, authorizing two bounded bilateral groups.
Do not stop solely to seek user confirmation after the qualifying leg group;
the user will be asleep. Do not add speed, continuous motion or load-balance rewards.

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
routine result extraction, failure diagnosis, held-out comparison, and
controlled experiments justified by the staged training plan. The user has
explicitly authorized automatic promotion beyond Stage 3 during this weekend.
Advance to the next defined stage only after its current promotion gate passes;
run validation before a risky change. Do not advance into a stage whose input
data, evaluator, or quantitative gate is still undefined, and do not change an
acceptance criterion to manufacture a pass.

Your job is to:

1. Inspect the completed remote training run.
2. Check its exit code first.
3. Read the remote training log.
4. Inspect TensorBoard metrics and checkpoints when available.
   For wrist/height groups use the remote installed Python to run
   `tools/summarize_wrist_training.py <sessions...>` (read-only). Its
   `conditional` results normalize masked numerators by actual command
   fractions. Never compare raw `_masked` values against conditional gates.
   Check actual final shoulder/wrist targets against configured ranges;
   resolved configuration and finite PPO alone are not command validation.
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
7. Do not run the complete held-out matrix during autonomous exploration. It is
   intentionally reserved for the user's final manual review because a group
   matrix takes roughly two hours. If a plan explicitly sets
   `evaluation.automatic=true`, read the materialized results from the group
   manifest and do not rerun them. Otherwise rank training candidates only
   provisionally from TensorBoard/log diagnostics and preserve every checkpoint.
8. Update `experiments/autotune_journal.md` with the hypothesis, exact change,
   validation/full-run status, result, baseline comparison, and next decision.

If the experiment crashed:
- diagnose the root cause first;
- do not blindly tune RL parameters;
- do not change CUDA, PyTorch, Warp, MuJoCo, NCCL, Conda, NVIDIA libraries, SSH, or Kubernetes configuration.

If the experiment trained successfully:
- formulate one specific hypothesis for improvement;
- if a valid Stage 5A group misses its gate but its recorded three-group budget
  remains, use the training plan's allowed controlled interventions to continue
  within 5A when supported by evidence. A missing prewritten next JSON is not
  a reason to stop. Never advance to 5B or relax gates to manufacture a pass.
- prefer changing one conceptual factor at a time;
- for continuous-wrist training, the latest `experiments/operation_capability_pack_next.md` supersedes the cancelled horizontal Stage6A1 screen; the one-pack completion must stop, not launch confirmation;
- Stage 5A v3 has independently passed. For authorized Stage 5B clutch v3,
  follow its three-group budget and three-mode gates in the training plan;
  do not stop merely for lack of a prewritten follow-up JSON. Normalize moving
  statistics by moving fractions and preserve world-anchored adjustment/balance.
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
- launch independent parameter or seed runs as a registered group. Omit the
  `evaluation` block during exploration unless the user explicitly asks for an
  automatic held-out run.

For risky source-code or reward changes, first run a 1-GPU validation with 256
environments. Only after it initializes, produces finite rewards, and starts PPO
successfully may you launch a full 4-GPU run. Remember that
`--env.scene.num-envs` is per rank: use 1,024 per rank for 4,096 environments
globally, and report both counts. Validation runs do not count toward the
active stage budget.

Do not start another full 4-GPU run if one is already active.

Stop instead of launching another experiment when any of these applies:
- all requested training stages have been explored and no controlled next step
  remains;
- the active stage budget recorded in `experiments/autotune_journal.md` has
  been exhausted;
- two consecutive registered experiment groups fail to improve the targeted
  held-out wrist/balance metrics;
- the next stage lacks its required data or control interface (a missing
  automatic evaluator alone is not a blocker because final evaluation is manual);
- evidence is insufficient to justify a controlled next experiment.

Never launch a run merely to keep the loop alive. When stopping, write the
reason and the best checkpoint/result to the journal. Also write a concise stop
reason to `.autotune/STOP_REQUESTED`. The shell watcher will mark the completed
trial handled and then exit cleanly instead of remaining resident.

Do not install or upgrade packages.

If there is not enough evidence to justify another experiment, stop and report why instead of launching one.
