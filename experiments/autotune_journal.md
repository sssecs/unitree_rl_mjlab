# G1 wrist-recovery autotune journal

## Operating constraints

- Pure reinforcement learning only for this phase.
- Change one conceptual factor per controlled experiment when practical.
- Validate risky changes on one GPU with 256 environments before a full run.
- At most five full 4-GPU runs; validation runs do not count.
- Stop after three consecutive full runs without improvement, after meeting the
  acceptance criteria, or when a material user decision is needed.

## Experiments

### `g1_wrist_recovery_v0_smoke`

- Type: 1-GPU launcher smoke test, 256 environments, 10 iterations.
- Result: failed before Python startup with exit code 126 because the tracked
  `run_train.sh` has no executable bit while `remote_train.sh` invoked it as an
  executable.
- Decision: invoke the required launcher as `bash ./run_train.sh`; no RL change.

### `g1_wrist_recovery_v0_smoke2`

- Type: 1-GPU environment smoke test, 256 environments, 10 iterations.
- Result: CUDA, Warp, and the base environment initialized, then command-manager
  construction failed because `heading_command=False` retained a non-null
  heading range inherited from the velocity task.
- Decision: explicitly set `twist.ranges.heading = None`; no RL change.

### `g1_wrist_recovery_v0_smoke3`

- Hypothesis: explicitly clearing the inherited heading range would allow all
  managers to construct and PPO to begin on the wrist-recovery environment.
- Exact change under test: `twist.ranges.heading = None`; no reward or PPO
  hyperparameter was changed.
- Status: 1-GPU validation with 256 environments and 10 iterations; failed with
  exit code 1 before PPO startup, so no full run was launched.
- Result: CUDA, Warp, MuJoCo, the base environment, and the command, action,
  observation, and termination managers initialized successfully. Reward-manager
  construction then raised `TypeError: object of type 'NoneType' has no len()`:
  `electrical_power_cost` attempted to resolve `joint_names=None`. The run
  produced no TensorBoard event file or checkpoint, so it neither diverged nor
  generated finite training metrics to evaluate.
- Baseline comparison: no useful training baseline exists yet; this run made it
  one manager farther than `g1_wrist_recovery_v0_smoke2` but still did not reach
  PPO. It does not count toward the five-full-run budget.
- Decision: classify as a code/configuration crash. Select all robot joints only
  for the energy term with
  `SceneEntityCfg("robot", joint_names=(r".*",))`, then repeat the same 1-GPU,
  256-environment, 10-iteration validation as
  `g1_wrist_recovery_v0_smoke4`. No RL parameter will be tuned until the task
  initializes and emits finite rollout/training metrics.

### `g1_wrist_recovery_v0_smoke4`

- Hypothesis: selecting all robot joints explicitly for `electrical_power_cost`
  would let reward-manager construction finish and allow finite PPO training to
  begin.
- Exact change under test: the energy reward's asset selector is
  `SceneEntityCfg("robot", joint_names=(r".*",))`; all reward weights, PPO
  settings, commands, curricula, and disturbances remained unchanged.
- Status: successful 1-GPU validation with 256 environments and 10 iterations;
  exit code 0. It initialized CUDA, Warp, MuJoCo, every environment manager, and
  PPO, then completed all 81,920 simulation steps at about 5,800 steps/s.
- Result: training was finite with no exceptions or NaNs. TensorBoard contains
  10 samples for every key scalar. At iteration 9, mean reward was 0.459, mean
  episode length was 66.42 steps, value loss was 0.493, surrogate loss was
  -0.0274, policy standard deviation was 0.201, mean wrist-position error was
  0.710 m, peak wrist-position error was 0.754 m, wrist-rotation error was 0.798
  rad, inter-wrist error was 0.168 m, mean action acceleration was 0.403, and
  foot stagger was 0.161 m. Bad-orientation terminations were 4.44 per episode
  batch and base-height terminations were 0.188. The worsening errors relative
  to iteration 0 occurred during only ten startup iterations and are not treated
  as convergence or divergence. TensorBoard was written under
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_08-38-32`; checkpoints
  `model_0.pt` and `model_9.pt` are present.
- Baseline comparison: no useful trained baseline exists. Unlike smoke3, this
  run passed the previously failing reward-manager boundary and demonstrated
  finite rollout collection and PPO updates. It is validation only and does not
  count toward the five-full-run budget.
- Decision: the source/configuration fix is validated. Launched the unchanged
  first full baseline as `g1_wrist_recovery_v0_full1` in detached remote tmux
  on all four GPUs with 4,096 environments per rank (16,384 globally) and
  5,000 iterations. The hypothesis is that the staged
  30,000-step quiet-hold warmup and 60,000-step reach/payload/push ramp give PPO
  enough curriculum to learn wrist holding and balance recovery. Do not tune a
  reward or PPO parameter before establishing this baseline.

### `g1_wrist_recovery_v0_full1`

- Hypothesis: the staged 30,000-control-step quiet-hold warmup followed by the
  60,000-step reach, payload, and push ramp would let the initial PPO setup learn
  accurate bimanual wrist holding while retaining balance recovery.
- Exact change under test: none relative to the successful `v0_smoke4`
  validation. This was the first full baseline, using the initial wrist-recovery
  reward configuration and PPO settings.
- Status: successful full 4-GPU run with 4,096 environments per rank, 16,384
  environments globally, and 5,000
  iterations; exit code 0. All workers completed normally in 3:15:44 after
  2,621,440,000 environment steps. This is full run 1 of the five-run budget.
- Result: no traceback, code/configuration failure, infrastructure failure, NaN,
  or non-finite TensorBoard scalar was found. The run directory is
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_08-43-32`; it contains the
  TensorBoard event file and checkpoints every 100 iterations plus
  `model_4999.pt`. Over the final 100 iterations, mean reward was 136.819 and
  mean episode length was 599.88/600. Mean wrist-position error was 0.00222 m,
  peak wrist error 0.01707 m, wrist-rotation error 0.05721 rad, inter-wrist error
  0.00255 m, and foot stagger 0.16119 m. Mean action acceleration was 0.50665;
  foot-slide reward contribution was -0.00277 and action-rate contribution was
  -0.20098. Base-height and bad-orientation terminations averaged 0.00344 and
  0.000625 per logged episode batch. All of these scalars were finite. The
  remaining conspicuous balance/tracking weakness was zero-command yaw-rate
  error at 0.43369 versus horizontal-velocity error at 0.10635.
- Baseline comparison: this is the first useful full baseline. Relative to the
  10-iteration smoke validation, it progressed from startup behavior (0.710 m
  wrist-position error, 66.42-step episodes) to stable full-length episodes and
  millimeter-scale mean wrist error under the fully ramped task. The smoke run
  is not a trained comparator, so no causal improvement claim is made.
- Best checkpoint/result: retain `model_4999.pt` as the baseline checkpoint. Its
  final values (0.00199 m mean wrist-position error, 0.01678 m peak error,
  0.05537 rad rotation error, 0.00235 m inter-wrist error, and no final-batch
  base-height or bad-orientation termination) are representative of the stable
  final-100 window rather than an isolated spike.
- Decision: continue because only one full run has completed and one controlled
  weakness is supported by the metrics. Test the hypothesis that strengthening
  the zero-yaw stabilization signal will reduce yaw-rate error without degrading
  wrist tracking, survival, recovery stepping, or action smoothness. Change only
  `base_yaw_rate` reward weight from -0.1 to -0.5. Because this is a reward
  change, `g1_wrist_recovery_yawpen_v1_val` was launched in detached remote tmux
  on one GPU with 256 environments for 10 iterations. Its result is pending;
  launch a second full run only in the next completion-triggered turn if this
  validation initializes and starts finite PPO.

### `g1_wrist_recovery_yawpen_v1_val`

- Hypothesis: increasing the zero-command base-yaw-rate penalty from -0.1 to
  -0.5 will give the policy a stronger yaw-stabilization signal without
  breaking environment initialization, PPO learning, wrist tracking, balance,
  or action smoothness.
- Exact change under test: only the `base_yaw_rate` reward weight changed from
  -0.1 to -0.5. Commands, curricula, disturbances, all other reward weights,
  and PPO settings remained unchanged.
- Status: successful 1-GPU validation with 256 environments and 10 iterations;
  exit code 0. CUDA, Warp, MuJoCo, all environment managers, rollout
  collection, and PPO initialized and completed normally. This validation does
  not count toward the five-full-run budget.
- Result: all 81,920 simulation steps completed at 6,120 steps/s on the final
  iteration. The run produced a TensorBoard event file with 10 samples for
  every key scalar and checkpoints `model_0.pt` and `model_9.pt` under
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_12-07-19`. No traceback,
  exception, NaN, or non-finite scalar was found. At iteration 9, mean reward
  was 0.0474, mean episode length 66.36 steps, surrogate loss -0.02638, mean
  wrist-position error 0.7144 m, peak wrist error 0.7534 m, wrist-rotation
  error 0.8167 rad, inter-wrist error 0.1777 m, mean action acceleration
  0.4000, horizontal-velocity error 0.0938, yaw-rate error 0.1000, and foot
  stagger 0.1400 m. Base-height and bad-orientation terminations were 0.3125
  and 4.0625 per logged episode batch. The low reward and short episodes are
  normal startup behavior in this deliberately short validation, not evidence
  of divergence or converged performance.
- Baseline comparison: compared with the original successful 10-iteration
  `v0_smoke4` validation, final yaw-rate error was effectively unchanged
  (0.1000 versus 0.1009), as were mean episode length (66.36 versus 66.42),
  wrist-position error (0.7144 versus 0.7099), and action acceleration (0.4000
  versus 0.4027). Total reward is lower (0.0474 versus 0.4592) because the
  tested penalty has a larger negative scale; that expected accounting change
  is not used as a quality comparison. Ten startup iterations are sufficient
  for safety validation but not for testing the yaw-stabilization hypothesis.
- Decision: the reward change is safe enough for a controlled full comparison,
  and remote status showed no tmux sessions or GPU processes. Launch full run 2
  of at most five as `g1_wrist_recovery_yawpen_v1_full2` on all four GPUs with
  4,096 environments per rank (16,384 globally) and 5,000 iterations. Compare
  its final-window yaw-rate
  error, wrist errors, terminations, foot stagger/slide, and action acceleration
  directly against `g1_wrist_recovery_v0_full1`; do not infer improvement from
  total reward because its scale changed.

### `g1_wrist_recovery_yawpen_v1_full2`

- Hypothesis: increasing the zero-command base-yaw-rate penalty from -0.1 to
  -0.5 would reduce yaw drift without degrading wrist tracking, survival,
  recovery stepping, foot behavior, or action smoothness.
- Exact change under test: only `base_yaw_rate` reward weight changed from -0.1
  to -0.5 relative to `g1_wrist_recovery_v0_full1`. Commands, curricula,
  disturbances, all other reward weights, and PPO settings were unchanged. The
  preceding `g1_wrist_recovery_yawpen_v1_val` validation passed.
- Status: successful full 4-GPU run with 4,096 environments per rank, 16,384
  environments globally, and 5,000
  iterations; exit code 0. All workers completed normally in 3:16:41 after
  2,621,440,000 environment steps. This is full run 2 of the five-run budget.
- Result: training was finite and stable. The run produced 5,000 samples for
  each of 50 TensorBoard scalars with no non-finite values, plus 51 checkpoints
  through `model_4999.pt`, under
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_12-12-08`. Over the final
  100 iterations, mean reward was 136.616, mean episode length was 599.978/600,
  value loss was 0.01307, surrogate loss was -0.00321, and policy standard
  deviation was 0.22155. Mean wrist-position error was 0.00217 m, peak wrist
  error 0.01693 m, wrist-rotation error 0.05732 rad, inter-wrist error 0.00261
  m, horizontal-velocity error 0.10824, yaw-rate error 0.36214, foot stagger
  0.14617 m, and mean action acceleration 0.49371. Action-rate and foot-slide
  reward contributions were -0.19718 and -0.00262. Base-height and
  bad-orientation terminations averaged 0.000625 and 0 per logged episode batch.
  A SIGTERM traceback emitted only during torchrun cleanup after “All workers
  completed successfully” and “Workers exited without errors”; it did not
  affect the exit code, metrics, event file, or final checkpoint and is not
  classified as a code, training, or infrastructure failure.
- Baseline comparison: versus the final-100 window of
  `g1_wrist_recovery_v0_full1`, yaw-rate error improved 16.5% (0.43369 to
  0.36214), action acceleration improved 2.6% (0.50665 to 0.49371), foot
  stagger improved 9.3% (0.16119 to 0.14617), and foot-slide cost magnitude
  improved 5.3% (0.00277 to 0.00262). Mean and peak wrist-position errors
  improved 2.4% and 0.8%; wrist-rotation error was effectively unchanged
  (+0.2%), while inter-wrist error and horizontal-velocity error regressed
  slightly (+2.4% and +1.8%). Episode length was unchanged at effectively the
  600-step maximum, and rare orientation/height terminations did not worsen.
  Total reward is not treated as a quality comparison because its scale changed.
- Best checkpoint/result: retain this run's `model_4999.pt` as the current best
  result because it materially improves the targeted yaw metric and also
  improves smoothness and foot behavior without a meaningful wrist or survival
  tradeoff.
- Decision: continue; two full runs have completed, the latest improved the key
  balance metric, and yaw error remains the largest conspicuous weakness. Test
  whether a smaller second strengthening step preserves those gains while
  reducing yaw further by changing only `base_yaw_rate` from -0.5 to -1.0.
  Because this is a reward change, launch `g1_wrist_recovery_yawpen_v2_val` as
  a 1-GPU, 256-environment, 10-iteration validation. Only a later
  completion-triggered turn may launch full run 3 after verifying finite PPO.

### `g1_wrist_recovery_yawpen_v2_val`

- Hypothesis: increasing the zero-command base-yaw-rate penalty from -0.5 to
  -1.0 will provide a further yaw-stabilization signal without breaking
  environment initialization, finite PPO learning, wrist tracking, balance, or
  action smoothness.
- Exact change under test: only `base_yaw_rate` reward weight changed from -0.5
  to -1.0 relative to `g1_wrist_recovery_yawpen_v1_full2`. Commands,
  curricula, disturbances, all other reward weights, and PPO settings remained
  unchanged.
- Status: successful 1-GPU validation with 256 environments and 10 iterations;
  exit code 0. CUDA, Warp, MuJoCo, all environment managers, rollout
  collection, and PPO initialized and completed normally. This validation does
  not count toward the five-full-run budget.
- Result: all 81,920 simulation steps completed, reaching 5,972 steps/s on the
  final iteration. The run produced 10 samples for each of 50 TensorBoard
  scalars, all finite, and checkpoints `model_0.pt` and `model_9.pt` under
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_15-38-00`. No traceback,
  exception, NaN, or non-finite scalar was found. At iteration 9, mean reward
  was -0.2355, mean episode length 66.12 steps, value loss 0.4381, surrogate
  loss -0.02845, mean wrist-position error 0.7106 m, peak wrist error 0.7518 m,
  wrist-rotation error 0.8050 rad, inter-wrist error 0.1613 m, mean action
  acceleration 0.4010, horizontal-velocity error 0.0931, yaw-rate error 0.1013,
  and foot stagger 0.1435 m. Base-height and bad-orientation terminations were
  0.0625 and 4.53125 per logged episode batch.
- Baseline comparison: compared with `g1_wrist_recovery_yawpen_v1_val` at
  iteration 9, startup behavior was effectively unchanged: yaw-rate error was
  0.1013 versus 0.1000, episode length 66.12 versus 66.36, wrist-position error
  0.7106 versus 0.7144, action acceleration 0.4010 versus 0.4000, and foot
  stagger 0.1435 versus 0.1400 m. The lower total reward (-0.2355 versus
  0.0474) is the expected accounting effect of the stronger negative reward
  weight and is not treated as a quality regression. Ten startup iterations
  validate safety and finite optimization, not the yaw hypothesis itself.
- Decision: the controlled reward change passed validation. Remote status
  showed no tmux sessions, no GPU processes, and all four GPUs idle. Launched
  full run 3 of at most five as `g1_wrist_recovery_yawpen_v2_full3` with 4,096
  environments and 5,000 iterations on all four GPUs. Compare its final-100
  yaw-rate error, wrist errors, terminations, foot stagger/slide, horizontal
  velocity, and action acceleration directly against
  `g1_wrist_recovery_yawpen_v1_full2`; do not compare total reward across the
  changed reward scales.

### `g1_wrist_recovery_yawpen_v2_full3`

- Hypothesis: increasing the zero-command base-yaw-rate penalty from -0.5 to
  -1.0 would further reduce yaw drift without materially degrading wrist
  tracking, survival, recovery stepping, foot behavior, or action smoothness.
- Exact change under test: only `base_yaw_rate` reward weight changed from -0.5
  to -1.0 relative to `g1_wrist_recovery_yawpen_v1_full2`. Commands,
  curricula, disturbances, all other reward weights, and PPO settings were
  unchanged. The preceding `g1_wrist_recovery_yawpen_v2_val` validation passed.
- Status: successful full 4-GPU run with 4,096 environments per rank, 16,384
  environments globally, and 5,000
  iterations; exit code 0. All workers completed normally in 3:14:47 after
  2,621,440,000 environment steps. This is full run 3 of the five-run budget.
- Result: training was finite and stable. The run produced 5,000 samples for
  each of 50 TensorBoard scalars with no non-finite values, plus 51 checkpoints
  through `model_4999.pt`, under
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_15-42-56`. Over the final
  100 iterations, mean reward was 135.786, mean episode length was 599.953/600,
  value loss was 0.01508, surrogate loss was -0.00320, and policy standard
  deviation was 0.22974. Mean wrist-position error was 0.00237 m, peak wrist
  error 0.01661 m, wrist-rotation error 0.06116 rad, inter-wrist error 0.00279
  m, horizontal-velocity error 0.11007, yaw-rate error 0.33685, foot stagger
  0.14380 m, and mean action acceleration 0.50870. Action-rate and foot-slide
  reward contributions were -0.21485 and -0.00300. Base-height and
  bad-orientation terminations averaged 0 and 0.00156 per logged episode batch.
  The log ends with all workers completing successfully and contains no
  traceback, training failure, or infrastructure failure.
- Baseline comparison: versus the final-100 window of
  `g1_wrist_recovery_yawpen_v1_full2`, yaw-rate error improved another 7.0%
  (0.36214 to 0.33685) and foot stagger improved 1.6% (0.14617 to 0.14380 m).
  Peak wrist error improved 1.9% (0.01693 to 0.01661 m). However, mean action
  acceleration regressed 3.0% (0.49371 to 0.50870), foot-slide cost magnitude
  regressed 14.5% (0.00262 to 0.00300), mean wrist-position error regressed
  9.1% (0.00217 to 0.00237 m), wrist-rotation error regressed 6.7% (0.05732 to
  0.06116 rad), and inter-wrist error regressed 7.0% (0.00261 to 0.00279 m).
  Horizontal-velocity error regressed 1.7%; episode length remained effectively
  maximal, and rare termination rates did not materially change. Total reward
  is not used for comparison because the reward scale changed.
- Best checkpoint/result: retain
  `g1_wrist_recovery_yawpen_v1_full2`'s `model_4999.pt` as the best balanced
  checkpoint; this run's `model_4999.pt` is the best yaw-specialized result but
  pays measurable smoothness, foot-slide, and wrist-tracking costs.
- Decision: continue because three full runs have completed, the latest still
  improved the targeted yaw metric, and the two tested weights bracket a clear
  tradeoff. Test the single midpoint `base_yaw_rate=-0.75`: the hypothesis is
  that it will retain most of the -1.0 yaw gain while recovering the superior
  action smoothness, foot sliding, and wrist tracking seen at -0.5. Because
  this is a reward change, remote status was checked and showed no tmux
  sessions, no GPU processes, and all four GPUs idle. Launched
  `g1_wrist_recovery_yawpen_v3_mid075_val` as a 1-GPU, 256-environment,
  10-iteration validation; its result is pending. Only a later
  completion-triggered turn may launch full run 4 after verifying finite PPO.

### `g1_wrist_recovery_yawpen_v3_mid075_val`

- Hypothesis: setting the zero-command `base_yaw_rate` penalty to the midpoint
  -0.75 will be safe for PPO and, over a full run, may retain most of the yaw
  improvement at -1.0 while recovering the better wrist tracking, action
  smoothness, and foot sliding observed at -0.5.
- Exact change under test: only `base_yaw_rate` changed from -1.0 to -0.75
  relative to `g1_wrist_recovery_yawpen_v2_full3` (and from -0.5 to -0.75
  relative to the current best balanced `g1_wrist_recovery_yawpen_v1_full2`).
  Commands, curricula, disturbances, all other reward weights, and PPO settings
  remained unchanged.
- Status: successful 1-GPU validation with 256 environments and 10 iterations;
  exit code 0. CUDA, Warp, MuJoCo, all environment managers, rollout
  collection, and PPO initialized and completed normally. This validation does
  not count toward the five-full-run budget.
- Result: all 81,920 simulation steps completed, reaching 5,852 steps/s on the
  final iteration. The run produced 10 samples for each of 50 TensorBoard
  scalars, all finite, and checkpoints `model_0.pt` and `model_9.pt` under
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_19-12-35`. No traceback,
  exception, code/configuration failure, infrastructure failure, NaN, or
  non-finite scalar was found. At iteration 9, mean reward was -0.16735, mean
  episode length 65.62 steps, value loss 0.47185, surrogate loss -0.02647,
  mean wrist-position error 0.70748 m, peak wrist error 0.74759 m,
  wrist-rotation error 0.78834 rad, inter-wrist error 0.16458 m, mean action
  acceleration 0.40169, horizontal-velocity error 0.09218, yaw-rate error
  0.09760, and foot stagger 0.14984 m. Base-height and bad-orientation
  terminations were 0.125 and 3.90625 per logged episode batch.
- Baseline comparison: startup behavior remained in the same range as the
  preceding `g1_wrist_recovery_yawpen_v1_val` (-0.5) and
  `g1_wrist_recovery_yawpen_v2_val` (-1.0) validations. At iteration 9,
  yaw-rate error was 0.09760 versus 0.1000 and 0.1013, episode length 65.62
  versus 66.36 and 66.12, wrist-position error 0.70748 versus 0.7144 and
  0.7106, and action acceleration 0.40169 versus 0.4000 and 0.4010. These
  small startup differences do not establish policy-quality improvement. The
  changed total reward scale is not used for comparison, and ten iterations
  validate safety rather than the midpoint tradeoff hypothesis.
- Decision: the controlled midpoint reward passed validation. Remote status
  showed no tmux sessions, no GPU processes, and all four GPUs idle. Launch
  full run 4 of at most five as `g1_wrist_recovery_yawpen_v3_mid075_full4`
  with 4,096 environments per rank (16,384 globally) and 5,000 iterations on
  all four GPUs. Compare its
  final-100 yaw-rate error, wrist errors, terminations, foot stagger/slide,
  horizontal velocity, and action acceleration directly against both
  `g1_wrist_recovery_yawpen_v1_full2` and
  `g1_wrist_recovery_yawpen_v2_full3`; do not compare total reward across the
  changed reward scales.

### `g1_wrist_recovery_yawpen_v3_mid075_full4`

- Hypothesis: setting the zero-command `base_yaw_rate` penalty to the midpoint
  -0.75 would retain most of the yaw improvement at -1.0 while recovering the
  better wrist tracking, action smoothness, and foot sliding observed at -0.5.
- Exact change under test: only `base_yaw_rate` changed from -1.0 to -0.75
  relative to `g1_wrist_recovery_yawpen_v2_full3` (and from -0.5 to -0.75
  relative to the best balanced `g1_wrist_recovery_yawpen_v1_full2`). Commands,
  curricula, disturbances, all other reward weights, and PPO settings were
  unchanged. The preceding `g1_wrist_recovery_yawpen_v3_mid075_val`
  validation passed.
- Status: successful full 4-GPU run with 4,096 environments per rank, 16,384
  environments globally, and 5,000
  iterations; exit code 0 and DONE marker present. All workers completed
  normally after 2,621,440,000 environment steps; the training loop elapsed
  3:15:24. This is full run 4 of the five-run budget.
- Result: training was finite and stable. The run produced 5,000 samples for
  each of 50 TensorBoard scalars with no non-finite values, plus 51 checkpoints
  through `model_4999.pt`, under
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_19-18-39`. Over the final
  100 iterations, mean reward was 135.981, mean episode length was
  599.842/600, value loss was 0.01705, surrogate loss was -0.00299, and policy
  standard deviation was 0.22822. Mean wrist-position error was 0.00273 m,
  peak wrist error 0.01786 m, wrist-rotation error 0.06086 rad, inter-wrist
  error 0.00301 m, horizontal-velocity error 0.10971, yaw-rate error 0.34722,
  foot stagger 0.17139 m, and mean action acceleration 0.50134. Action-rate
  and foot-slide reward contributions were -0.20490 and -0.00262. Base-height
  and bad-orientation terminations averaged 0.00375 and 0.00219 per logged
  episode batch. The final log reports that all workers completed successfully.
  Its SIGTERM traceback occurred only during torchrun cleanup after successful
  completion and did not affect the exit code, event file, metrics, or final
  checkpoint; it is not a code, training, or infrastructure failure.
- Baseline comparison: versus the final-100 window of the best balanced -0.5
  run, `g1_wrist_recovery_yawpen_v1_full2`, yaw-rate error improved 4.1%
  (0.36214 to 0.34722), but mean wrist-position error regressed 25.9% (0.00217
  to 0.00273 m), peak wrist error regressed 5.5%, wrist-rotation error regressed
  6.2%, inter-wrist error regressed 15.2%, foot stagger regressed 17.3%, action
  acceleration regressed 1.5%, and horizontal-velocity error regressed 1.4%.
  Foot-slide cost was effectively unchanged. Versus the -1.0 yaw-specialized
  `g1_wrist_recovery_yawpen_v2_full3`, yaw-rate error was 3.1% worse and foot
  stagger was 19.2% worse; action acceleration and foot sliding improved, but
  most wrist-position metrics remained worse. The same pattern is present in
  the final-500 windows, so it is not an isolated end-of-run fluctuation. Total
  reward is not used across the different reward scales.
- Best checkpoint/result: retain
  `g1_wrist_recovery_yawpen_v1_full2`'s
  `logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_12-12-08/model_4999.pt`
  as the best balanced checkpoint. It has the strongest combined wrist
  tracking, smoothness, foot behavior, survival, and materially improved yaw
  error. Retain `g1_wrist_recovery_yawpen_v2_full3/model_4999.pt` only as the
  yaw-specialized alternative. The local `base_yaw_rate` configuration was
  reverted from the unsuccessful midpoint -0.75 to the best balanced -0.5.
- Decision: stop without launching another experiment. Four full runs have
  completed. The midpoint hypothesis failed, the response of wrist and foot
  metrics to yaw-penalty strength is non-monotonic, and no numeric acceptance
  threshold or supported single-factor next change identifies how to spend the
  final full-run slot without making a material design tradeoff. A fifth run
  merely to keep the loop active would not be justified. Resume only after a
  user decision prioritizes additional yaw suppression versus wrist/foot
  quality, or supplies an acceptance target that supports a controlled next
  experiment.

### Held-out evaluator baseline (`yawpen_v1_full2`)

- Added a deterministic evaluator with six fixed scenarios, 64 parallel
  environments, and seeds 1103, 2207, and 3301 (1,152 episodes total). It
  explicitly controls wrist targets, wrist payloads, and root-velocity pushes,
  disables automatic training disturbances, and emits per-episode CSV plus
  aggregate JSON and a manifest.
- Static hold, payload, and push each achieved 100% success with no falls.
  Static mean wrist position/orientation error was 1.29 mm/0.0121 rad. Under
  push, mean position error was 1.22 mm, mean peak was 3.58 mm, and peak foot
  displacement averaged 3.93 cm.
- Symmetric reach achieved 53.1% success with 46.4% falls; asymmetric reach
  achieved 38.5% success with 37.5% falls; combined reach/load/push achieved
  40.1% success with 58.3% falls. These failures were hidden by the excellent
  aggregate training metrics.
- Decision: retain the checkpoint, but classify it as passing Stages 0--2 only.
  The next training hypothesis should expand Stage 3 reach and asymmetric
  support-posture training rather than spending the remaining experiment on a
  further yaw-reward sweep. Do not tune against the exact held-out tuples.

### Engineering audit correction

- The four historical full runs used `env.scene.num_envs=4096` under four DDP
  ranks. That means 4,096 environments per rank and 16,384 globally, consistent
  with `5000 * 32 * 16384 = 2,621,440,000` environment steps. Earlier journal
  wording that called this simply “4,096 environments” was corrected.
- All four historical full policies came from one base training seed (with
  rank-local seeds offset by local rank). Their 1--3% differences are not
  statistically sufficient to establish a universal best configuration. The
  `yawpen_v1_full2` checkpoint remains the retained single-run baseline because
  of its held-out balance, but its “best” label is provisional pending
  independent-seed confirmation.
- Future runs use explicit per-rank/global counts, source/config provenance,
  grouped four-way screening, and independent-seed confirmation before a final
  policy claim.
- Infrastructure validation passed on 2026-09-11. A one-GPU run with 64
  environments and seed 907 completed two finite PPO iterations and wrote the
  commit, dirty diff, exact arguments, file hashes, resolved YAML, TensorBoard
  path, and run metadata. A four-GPU DDP run with 16 environments per rank and
  seed 911 completed one iteration; metadata correctly recorded world size 4,
  64 global environments, rank seeds 911--914, and the log reported exactly
  `64 * 32 = 2,048` environment steps.
- The first full-source snapshot was 170 MB, so the retained design now stores
  the Git commit, binary dirty diff, hashes for every synchronized file, and a
  compressed archive of untracked files only (9--12 KB in validation). The
  oversized smoke-test archive was removed; its configs, hashes, logs, and
  checkpoints remain.
- Manifest-based tar synchronization was integration-tested by uploading and
  then deleting one temporary managed file. The second sync pruned exactly that
  path. A subsequent read-only audit found no extra files under remote `src`,
  `scripts`, `tools`, or `experiments`.

### Stage 3 screen planned (`g1_stage3_screen_v1`)

- Hypothesis: the retained policy fails held-out reach because training only
  presents bilateral forward translations. Continuously sampled independent
  wrist x/y/z offsets and modest rotations should teach proactive whole-body
  support for asymmetric targets; increasing the fraction of reach episodes
  may improve this further, provided static/load/push skills do not regress.
- Change: add an `asymmetric_probability` command parameter. Asymmetric reach
  episodes independently sample each wrist from x 0.12--0.28 m, y
  -0.10--0.10 m, z -0.06--0.06 m and random-axis rotation -0.25--0.25 rad.
  Sampling is continuous and does not replay the fixed held-out tuples.
- Controlled screen: four independent one-GPU runs cross reach probability
  0.50/0.80 and asymmetric probability 0.00/0.50. All use seed 1201, 4,096
  environments per run, and 5,000 iterations. The shared seed ranks the four
  configurations but does not support a final best-policy claim.
- Active Stage 3 budget: this four-run screen plus, only if promoted by the
  unchanged held-out suite, one confirmation group with at least three new
  training seeds. The earlier five-run cap applied to the completed yaw-reward
  pilot and is no longer the stopping rule for this new stage.
- Validation: `g1_stage3_asym_smoke_v1` completed two finite PPO iterations
  with 256 environments but, correctly, sampled no reaches during the 30k-step
  curriculum warmup. `g1_stage3_asym_active_smoke_v2` therefore set warmup to
  zero and ramp to one for five iterations. It exited 0, remained finite, and
  reported asymmetric fractions rising to 0.435 (the configured expectation is
  `0.80 * 0.50 = 0.40`), proving the new target branch was exercised.
- Launch: all four runs started in detached remote tmux sessions on 2026-09-11
  at approximately 03:50 CST from clean commit `86b9df6`, one run per H20.
  A post-launch check found all four sessions alive, each GPU at 60--73%
  utilization, finite rewards, and PPO iterations advancing. As expected, the
  asymmetric metric remains zero before the 30k-step curriculum warmup ends.
- Status: four-run screen active; watcher group `g1_stage3_screen_v1` will wait
  for all siblings before one medium-reasoning analysis turn.

### Stage 3 screen result (`g1_stage3_screen_v1`)

- Hypothesis and exact change: this fixed-seed, four-way screen tested reach
  probability `0.50` versus `0.80` and asymmetric-target probability `0.00`
  versus `0.50`; all other training, reward, curriculum, and PPO settings were
  identical. Each sibling used seed 1201, one H20, 4,096 environments (both
  per-rank and global), and 5,000 iterations (655,360,000 environment steps).
- Status: all four runs exited 0 and completed their final iteration normally:
  `reach050_asym000` (03:14:49), `reach080_asym000` (03:13:28),
  `reach050_asym050` (03:12:57), and `reach080_asym050` (03:11:13). Their
  preserved logs end at iteration 4999 with 600-step episodes and zero final
  base-height/bad-orientation terminations; no crash, CUDA/Warp infrastructure
  error, or non-finite training behavior was observed in the final logs.
- Training diagnostics (final 100, ranking-only): `reach050_asym000` was the
  least costly setting (wrist mean/peak/rotation errors 2.76/17.49 mm/0.0667
  rad, action acceleration 0.6103, foot stagger 0.1623 m, yaw error 0.4691).
  Raising reach to 0.80 without asymmetry worsened these to
  3.09/18.02 mm/0.0696 rad, 0.6285, 0.1645 m, and 0.4900. Adding 0.50
  asymmetry was costlier: reach050 had 3.71/18.95 mm/0.0883 rad, 0.6495,
  0.1771 m, and 0.5104; reach080 had 3.19/18.25 mm/0.0831 rad, 0.6416,
  0.1703 m, and 0.4969. The sampled asymmetric fractions were respectively
  0, 0, 0.255, and 0.402, consistent with the intended branch. These are not
  acceptance evidence and do not select a policy.
- Held-out evaluation blocker: the required nominal fixed suite was invoked on
  `reach050_asym000` via `tools/remote_evaluate_wrist.sh`. It first exposed a
  launcher quoting defect that sent an empty optional argument; the local
  wrapper was minimally fixed to omit optional arguments when none are given,
  reviewed with `git diff`, and synchronized. A rerun then initialized the
  immutable remote evaluator environment on cuda:0 but returned without
  `summary.json`, `episodes.csv`, or copied artifacts. At the immediate status
  check there was no tmux session and no listed GPU process, despite GPU 0
  reporting 99% utilization. This is an evaluator-launch/infrastructure issue,
  not evidence about any policy. The baseline nominal result remains the
  identical evaluator-v1 run at
  `results/wrist_recovery_eval/2026-09-11_10-56-40_model_4999_nominal`, but no
  comparable candidate or robust result exists.
- Decision: stop. The Stage 3 promotion gate explicitly requires useful fixed
  held-out improvement without regression of static, payload, and push; this
  cannot be assessed, and the training-only diagnostics already provide no
  defensible winner. Do not launch the independent-seed confirmation group or
  another training run. First repair and validate the remote evaluator process
  lifecycle, then rerun the unchanged nominal and robust suites for all four
  candidates and the retained baseline before considering promotion.

### Stage 3 evaluator incident correction

- The evaluator had not failed: its artifacts were written one timestamped
  directory below the requested output root. The watcher checked only the root
  and therefore misclassified a successful evaluation as missing. The existing
  `reach050_asym000` nominal result was recovered intact.
- The recovered result has 100% success and zero falls for static, payload,
  push, symmetric reach, and combined. Asymmetric reach has zero falls but only
  19.8% success, so the complete four-candidate nominal/robust matrix is still
  required before promotion.
- The remote evaluation wrapper now starts each evaluator in detached tmux,
  records log/exit/DONE state, tolerates reconnects, discovers nested artifacts,
  and supports idempotent recovery. An 8-environment detached evaluator smoke
  test completed with exit 0 and copied all expected artifacts.
- The shell watcher will materialize the full fixed evaluation matrix before
  invoking Codex. This removes long evaluator waits from the model turn and
  prevents training-only selection. Weekend automation may promote beyond
  Stage 3 within the bounded gates in the training plan.
- Stop notifications now support SMTP through the ignored
  `.autotune/notify.env`; notification remains disabled until local credentials
  and a recipient are configured.

### Stage 3 screen final result and confirmation promotion (`g1_stage3_screen_v1`)

- Status: all four fixed-seed screen siblings exited 0 after 5,000 PPO
  iterations (655,360,000 environment steps each) on one H20 with 4,096
  environments per rank/global. Their preserved logs completed normally at
  iteration 4999; finite TensorBoard metrics and final checkpoints are present,
  with no code/configuration error, infrastructure failure, NaN, or divergence.
- Fixed held-out comparison: the two no-asymmetry candidates did not promote:
  asymmetric reach success was only 19.8%/17.2% nominal and 18.8%/18.8% robust
  for reach probabilities 0.50/0.80, respectively. Both asymmetric-target
  candidates passed every nominal gate and the bounded robust gate: static,
  payload, push, symmetric reach, asymmetric reach, and combined were each
  100% successful with 0% falls in both suites. This materially improves the
  retained `yawpen_v1_full2` baseline, whose nominal/robust reach successes
  were 50.5/46.9% symmetric, 39.1/25.5% asymmetric, and 44.3/38.0% combined,
  with 40--61% falls.
- Ranking decision: select `reach050_asym050`, not as a final policy claim but
  as the confirmation configuration. It had lower nominal/robust mean wrist
  position errors than `reach080_asym050`: symmetric 0.89/0.96 versus
  1.07/1.12 mm, asymmetric 1.53/1.67 versus 1.68/1.70 mm, and combined
  0.84/0.92 versus 1.01/1.06 mm. Its static/load/push metrics remain quiet,
  and the lower reach mixture is therefore the conservative controlled choice.
- Decision: the unchanged Stage 3 promotion gate passes for the screen winner.
  Launch the sole permitted independent-seed confirmation group
  `g1_stage3_confirm_v1`: seeds 101, 211, 307, and 401, one H20 each, 4,096
  environments per run (both per-rank and global), 5,000 iterations, with only
  reach probability 0.50 and asymmetric-target probability 0.50 retained from
  the winning screen. The registered plan fixes nominal/robust evaluation,
  `model_4999.pt`, and the retained baseline. Aggregate held-out results across
  seeds; reject any seed exceeding a relevant fall limit by over five points.

### Stage 3 independent-seed confirmation result (`g1_stage3_confirm_v1`)

- Hypothesis and exact change: confirm the promoted `reach_probability=0.50`,
  `asymmetric_probability=0.50` configuration from the fixed-seed screen.
  Relative to retained `yawpen_v1_full2`, this changes only the wrist-target
  mixture; the existing randomized reset payloads and interval timed pushes,
  reward, curriculum, PPO settings, and fixed evaluator remain unchanged.
- Status: all four independent one-H20 siblings (`seed101`, `seed211`,
  `seed307`, and `seed401`) exited 0, wrote DONE markers, TensorBoard event
  files, and `model_4999.pt` checkpoints after 5,000 iterations. Each used
  4,096 environments (one rank, hence both per-rank and global) and completed
  in 3:11--3:16. Their final logs show finite PPO diagnostics, full 600-step
  episodes, and no traceback, NaN, code/configuration failure, or
  infrastructure failure. Final logged mean wrist error was 2.7--5.0 mm,
  action acceleration 0.629--0.703, and asymmetric sample fraction
  0.203--0.287. Remote status after completion showed no tmux session or GPU
  process.
- Fixed held-out nominal matrix (192 episodes/scenario/seed): aggregate
  success was 100% for static hold, payload, push, symmetric reach, and
  combined; asymmetric reach was 96.22% +/- 7.55 percentage points across
  seeds. All fall rates were 0%. Mean wrist position error was 0.068--0.205 cm
  by scenario, well below the Stage 3 4 cm final-position requirement. The
  lowest individual asymmetric-reach success was 84.90% (seed401), but this is
  not a rejection condition: the independent-seed rule applies gates to the
  aggregate and rejects only an individual fall rate exceeding its limit by
  more than five points; every individual fall rate was 0%.
- Fixed held-out robust matrix: aggregate success was 100% for static hold,
  payload, push, symmetric reach, and combined; asymmetric reach was
  93.88% +/- 9.38 percentage points. Again all fall rates were 0%, with all
  earlier-stage robust conditions above 85% success and both reach conditions
  above 70% success. This exceeds the bounded-robust limits without tuning
  the fixed seeds, scenarios, or thresholds.
- Baseline comparison: retained `yawpen_v1_full2` produced nominal/robust
  success of 50.52/48.96% symmetric reach, 39.06/29.69% asymmetric reach, and
  44.79/44.79% combined, with 39.58--54.17% falls. The confirmation group
  removes these reach/combined falls and raises every relevant success rate.
  This is consistent across four independent training seeds, unlike a
  single-seed training-metric claim.
- Decision: Stage 3 promotion is confirmed. No new Stage 4 run is launched:
  the confirmed training distribution already combines the required Stage 4
  axes (asymmetric reach plus reset payload and timed interval pushes) and the
  existing fixed combined evaluator is already passed at 100% nominal and
  robust success with 0% falls. There is no separately defined Stage 4 input
  factor or controlled screen whose result would distinguish a meaningful next
  change. Stage 5 must not begin because its moving-command held-out scenarios
  and quantitative gates are explicitly undefined in the training plan.
  Retain the four confirmation `model_4999.pt` checkpoints as the validated
  policy set; do not select a single seed from the held-out matrix for a new
  claim. Stop the autonomous loop pending a defined Stage 5 evaluator/gate or
  a user-specified next-stage training interface.

### Stage 5A shoulder-height and low-reach screen

- User-selected interface: extend the existing planar twist and bilateral wrist
  pose commands with a masked shoulder-line height target. Shoulder height is
  measured at the two fixed shoulder-pitch joint anchors, not at arm-dependent
  link centers. No torso-pitch, knee-angle, or reference-motion command is
  supplied, so forward waist flexion, knee flexion, and mixtures remain valid.
- Safety shaping: backward torso lean receives a one-sided penalty beyond about
  5 degrees. During initial learning, an active height task terminates only
  after its smooth target trajectory is halfway complete and backward lean
  exceeds 25 degrees. A shoulder-height-difference penalty has a 3 cm dead zone
  to reject severe lateral exploitation without suppressing ordinary balance.
  The fixed base-height and waist/hip-deviation costs are gated or reduced only
  during height tasks; prior standing behavior retains the original costs.
- Automatic evaluation policy: complete nominal/robust group matrices are no
  longer run during exploration because they take roughly two hours. Training
  metrics rank candidates provisionally, every checkpoint is preserved, and
  the user performs the final visual and held-out judgment. The watcher runs a
  held-out matrix only for a plan explicitly setting `evaluation.automatic`.
- Validation `g1_stage5_height_active_smoke_v1` exited 2 before simulation: the
  two-value Tyro options were initially passed in the wrong CLI representation.
  The plan now uses Python tuple syntax, as required by `mjlab.TYRO_FLAGS`.
- Validation `g1_stage5_height_active_smoke_v2` exited 0 and exercised 100%
  deep height tasks, but the immediate 15-degree termination left mean episodes
  near 15 steps and therefore provided inadequate learning horizon. This was a
  training-design failure, not a CUDA or numerical failure.
- Validation `g1_stage5_height_active_smoke_v3` exited 0 after five finite PPO
  iterations with 256 environments. The delayed 25-degree hard constraint had
  zero backward-lean terminations; mean episode length reached roughly 67--77
  steps under the deliberately harsh 100% deep-task setting. Logged torso
  forward bend was 0.37--0.57 while backward lean fell near zero in the final
  iterations, confirming the intended branch and sign convention. The formal
  runs retain the 30k-step warmup and 60k-step ramp.
- Registered screen `g1_stage5_height_screen_v1` crosses height-task frequency
  0.25/0.50 with moderate versus deep wrist/shoulder targets. All four siblings
  use seed 1501, one H20, 4,096 environments (per-rank and global), 5,000 PPO
  iterations, and the confirmed Stage 3 reach/asymmetry mixture. This screen
  tests stationary height/ground-reaching control; planar movement remains zero
  and will be introduced as the next separate factor after provisional review.

### Stage 5A shoulder-height and low-reach screen result (`g1_stage5_height_screen_v1`)

- Hypothesis and exact change: this fixed-seed four-way screen crossed
  height-task frequency (0.25/0.50) with moderate
  shoulder/wrist ranges (0.82--1.02 m / 0.35--0.55 m) versus deep ranges
  (0.68--0.98 m / 0.12--0.35 m). It retained the confirmed Stage 3
  reach/asymmetry mixture, zero planar twist, rewards, and PPO settings. Each
  sibling used seed 1501, one H20, 4,096 environments (both per-rank and
  global), and 5,000 PPO iterations (655,360,000 environment steps).
- Status: all four siblings exited 0 and completed normally at iteration 4999
  in 3:16--3:20. Each preserved `train.log`, a TensorBoard event file, and
  checkpoints through `model_4999.pt`. Final logs and final-100 TensorBoard
  windows were finite; no traceback, CUDA/Warp or infrastructure error, NaN,
  or divergence occurred. Remote status after completion showed no tmux
  session, GPU process, or active full DDP job.
- Provisional final-100 diagnostics: the two 25% cases were substantially
  better than 50% exposure. `freq025_deep` had height error 0.2847 m,
  mean/peak wrist error 0.1906/0.2110 m, action acceleration 0.7332,
  yaw error 0.6180, action-rate cost -0.4070, and foot-slide cost -0.00589;
  `freq025_moderate` had 0.2879 m, 0.1938/0.2135 m, 0.7743, 0.6472,
  -0.4493, and -0.00663. At 50%, moderate/deep height error rose to
  0.5437/0.5421 m, wrist error to 0.3840/0.3851 m, and action acceleration
  to 0.8422/0.8199. Episodes remained 596.6--598.4 of 600 and backward-lean
  terminations remained at or below 0.22% of logged episode batches. Deep
  25% had somewhat higher foot stagger (0.1666 m versus 0.1533 m), so this
  ranks a training candidate rather than establishing a final policy claim.
- Baseline comparison and decision: this plan deliberately omits automatic
  held-out evaluation, and Stage 5 has no fixed quantitative acceptance gate;
  the results are engineering diagnostics only, with all checkpoints retained
  for the user's final visual/held-out judgment. The resolved configurations
  confirm that the deep ranges were applied. Provisionally retain
  `freq025_deep/model_4999.pt` as the stationary-height candidate because it
  improves the relevant tracking and smoothness diagnostics at the same 25%
  exposure. The next defined single factor is gradual planar twist. Launch
  registered fixed-seed screen `g1_stage5_twist_screen_v1`, retaining exactly
  the 25%-deep mixture while crossing non-standing command fraction (25%/50%)
  and one low/medium planar-twist amplitude. No evaluation block is included;
  do not interpret the screen as a final policy selection.

### Stage 5B gradual planar-twist screen result (`g1_stage5_twist_screen_v1`)

- Hypothesis and exact change: starting from the provisional Stage 5A `freq025_deep`
  mixture, this fixed-seed screen introduced only planar twist.  It crossed non-standing
  command exposure of 25%/50% (`rel-standing-envs=0.75/0.50`) with low
  x/y/yaw ranges of +/-0.10/+/-0.05/+/-0.10 and medium ranges of
  +/-0.20/+/-0.10/+/-0.20.  The 25%-deep height mixture, 0.50 reach probability,
  0.50 asymmetric probability, rewards, curriculum, PPO settings, and seed 1601
  were otherwise unchanged.  Each sibling used one H20, 4,096 environments
  (both per-rank and global), and 5,000 iterations (655,360,000 environment steps).
- Status: every sibling recorded exit code 0 and DONE: `active25_low` at
  20:54:46 UTC, `active50_low` at 20:51:33, `active25_medium` at 20:51:29,
  and `active50_medium` at 20:50:22.  All completed iteration 4999 normally,
  with preserved TensorBoard event files and 11 MiB `model_4999.pt` checkpoints
  in respectively `2026-09-11_17-27-24`, `17-27-30`, `17-27-37`, and
  `17-27-43`.  Log tails and final-100 TensorBoard windows contain no traceback,
  CUDA/Warp or infrastructure error, NaN, divergence, or non-finite scalar.
  Remote status after completion had no tmux session, GPU process, or active full
  DDP job.
- Provisional final-100 diagnostics (training evidence only):
  `active25_low` / `active50_low` / `active25_medium` / `active50_medium` had
  wrist mean error 0.1885/0.1922/0.1965/0.1840 m, shoulder-height error
  0.2809/0.2858/0.2862/0.2718 m, action acceleration
  0.7468/0.7615/0.7399/0.7294, and x-y velocity error
  0.1527/0.1626/0.1635/0.1856.  Yaw error was 0.6304/0.6446/0.6223/0.6225;
  foot stagger was 0.1490/0.1528/0.1712/0.1524 m; and action-rate cost was
  -0.4236/-0.4345/-0.4140/-0.4024.  Mean episodes remained 598.3/598.3/598.3/598.8
  of 600.  Backward-lean termination was zero except `active25_medium` (0.0097)
  and `active50_medium` (0.00094) per logged episode batch; the corresponding
  base-height plus bad-orientation counts were 0.0309/0.0350/0.0372/0.0253.
- Baseline comparison and decision: compared with the stationary `freq025_deep`
  candidate (height/wrist/action-acceleration 0.2847/0.1906/0.7332), low-amplitude
  25%-active twist retains essentially the same height and wrist tracking while
  adding the best x-y tracking and avoiding any backward-lean termination.  It is
  therefore the conservative provisional Stage 5B candidate:
  `/home/dev/unitree_rl_mjlab/logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-11_17-27-24/model_4999.pt`.
  This is not a final policy claim: the plan intentionally has no automatic
  held-out evaluation or numerical Stage 5 promotion gate, and all sibling
  checkpoints are retained for the user's visual/fixed-suite review.  Stop
  autonomous training here.  No further defined Stage 5 controlled factor or
  acceptance criterion exists, and Stage 6 requires the absent mapped-trajectory
  split and interface contract; launching another run would not be justified.

### Stage 5A v1 post-analysis and repair decision

- The provisional selections above are overturned by absolute-task analysis.
  At 25% height exposure, final-100 wrist error was about 0.19 m; at 50% it was
  about 0.385 m. This near-linear scaling with height exposure, while Stage 3
  non-height tracking was millimeter-scale, indicates that the height-active
  cases were being sacrificed rather than solved. All candidates finished with
  shoulder height near 1.065--1.070 m and only about 6--9 degrees of forward
  torso bend. The temporary lower posture around iteration 2,000 disappeared by
  iteration 3,000, consistent with convergence to the easier standing solution.
- Root cause: shoulder height and absolute wrist height were sampled
  independently. The XML link offsets give a 0.4104 m shoulder-pitch-anchor to
  wrist-yaw-anchor path-length upper bound. Monte Carlo integration of the
  configured independent uniform ranges shows that the vertical gap alone
  exceeded 0.410 m for 75.5% of moderate and 95.3% of deep samples; forward
  extension and joint/collision limits make the true infeasible fraction higher.
  The exponential shoulder term and fine wrist term then supplied almost no
  far-field gradient, while alive and easy-task rewards remained available.
- Stage 5B additionally used the velocity command's 0.1 norm deadband. Only
  about 28.0% of sampled low-range commands survive it, so `active25_low` had
  nonzero commands in only about 7.0% of all environments. It was the easiest
  candidate, not evidence of learned locomotion. Stage 5B remains paused.
- Repair hypothesis: translate each height-active wrist target by the same
  vertical displacement commanded for the shoulders, add only a small residual
  wrist-height offset, and add height-active Huber losses for wrist and shoulder
  errors. This should remove contradictory commands and prevent large-error
  reward saturation without prescribing whether the robot bends its waist,
  knees, or both. First validate with stationary commands, then screen moderate
  versus deep shoulder lowering at 35%/50% exposure.

### Stage 5A feasible-target validation (`g1_stage5a_feasible_smoke_v2`)

- Single-H20 validation used 256 environments, 120 PPO iterations, 100% active
  moderate height commands, and no reach asymmetry. It exited 0 with finite
  rewards, losses, observations, and the new conditional diagnostics; no Python,
  CUDA, Warp, configuration, or NaN failure occurred.
- The resolved configuration contains the coupled wrist offset range
  +/-0.02 m, shoulder range 0.78--0.90 m, Huber wrist/shoulder weights -8/-6,
  and zero planar command. The logged target shoulder--wrist vertical gap stayed
  at 0.348--0.349 m, below the 0.410 m link-path upper bound, confirming that the
  previous independent-height contradiction is removed.
- This deliberately abrupt random-policy stress test is not a performance
  evaluation: mean episodes increased only from roughly 60 to 82 steps and
  still contained bad-orientation/backward-lean terminations. The formal screen
  therefore retains the normal 30k-step warmup and 60k-step severity/probability
  ramp so standing is learned before height commands become frequent. The smoke
  authorizes the formal run only at the software/numerical level.

### Stage 5A feasible-target repair screen result (`g1_stage5a_feasible_screen_v2`)

- Hypothesis and exact change: this fixed-seed, four-way screen tested the
  repaired coupled height targets and height-active Huber losses at height-task
  exposure 0.35/0.50 and shoulder-height ranges 0.78--0.98 m (moderate) versus
  0.62--0.90 m (deep).  All siblings retained zero planar twist, reach and
  asymmetric-target probabilities of 0.50, wrist residual-height range
  +/-0.02 m, low-reach extension 0.06--0.14 m, the normal warmup/ramp, rewards,
  PPO settings, and seed 1701.  Each used one H20 with 4,096 environments
  (per-rank and global) for 5,000 PPO iterations (655,360,000 environment
  steps).  The plan intentionally has no automatic held-out evaluation.
- Status: all four siblings exited 0, wrote DONE markers, TensorBoard event
  files, and final `model_4999.pt` checkpoints.  Their final logs complete at
  iteration 4999 after 3:19--3:23.  TensorBoard contains 5,000 finite samples
  for every inspected scalar; log tails contain no traceback, CUDA/Warp or
  infrastructure failure, NaN, or divergence.  Remote status after completion
  had no tmux sessions and no GPU processes.
- Final-100 provisional diagnostics, ordered as 35%-moderate / 50%-moderate /
  35%-deep / 50%-deep: height-active wrist error was
  8.73/14.77/9.38/13.24 cm; height-active shoulder error was
  7.61/11.49/6.54/7.71 cm; non-height wrist error remained
  0.33/0.25/0.35/0.26 cm; and mean episode length was
  432.0/361.5/432.8/357.3 of 600.  Backward-lean termination was zero in every
  final window and mean backward lean was at most 0.00129 rad, so the repaired
  target and one-sided safety constraint behaved as intended.  However,
  bad-orientation terminations were 0.084/5.471/3.235/5.548 per logged episode
  batch, respectively.  Increasing exposure from 35% to 50% therefore worsened
  height wrist tracking and episode survival materially in both depth ranges.
- Baseline comparison: unlike the rejected v1 screen, the coupled targets keep
  the shoulder--wrist command feasible and preserve millimetre-scale
  non-height tracking rather than sacrificing the old task.  This is a useful
  engineering repair, but not a Stage 5A promotion: the predeclared stationary
  gate requires height-active wrist error below 4 cm, height-active shoulder
  error below 5 cm, nearly full episode length, no systematic backward lean,
  and acceptable non-height retention.  No sibling meets the first three
  requirements.  `active035_moderate/model_4999.pt`
  (`2026-09-11_23-28-57`) is the closest provisional wrist/retention candidate,
  while `active035_deep/model_4999.pt` has the lower shoulder error; neither is
  retained as a promoted policy and all four checkpoints are preserved.
- Decision: stop autonomous training.  Stage 5B twist remains blocked by the
  explicit stationary-height gate, and no next stationary curriculum factor,
  stage budget, or acceptance criterion is defined that would justify another
  registered group without changing the agreed task design.  Do not relax the
  4/5-cm or episode-length criteria to manufacture a pass.  Resume only after
  the user supplies a controlled next Stage 5A intervention or accepts a
  revised, predeclared curriculum/gate.

### Stage 5A v3 engineering correction (supersedes v2 interpretation)

- Read-only CPU reproduction confirmed shoulder and wrist residual samples
  remained exactly zero because advanced-index `.uniform_()` changed a copy.
  Symmetric reach extension had the same bug (zero or stale). V2 therefore
  commanded final shoulder height zero at full curriculum; moderate/deep ranges
  were not actually applied. The v2 screen cannot rank depth configurations.
- V2 raw masked errors were also mistaken for conditional errors. Corrected
  final-100 wrist errors for 35%-moderate / 50%-moderate / 35%-deep / 50%-deep
  are 24.92/29.59/26.76/26.43 cm; shoulder errors are
  21.72/23.02/18.66/15.38 cm; non-height wrist errors are about 0.5 cm.
  Backward-lean metrics are projections/sines, not radians. Existing checkpoints
  and historical records are preserved, but v2 promotion claims are withdrawn.
- V3 hypothesis: correct explicit sampling writeback and masked-metric
  normalization, with no reward/PPO change. Add CPU regression checks of deployed
  sampling and target initialization, plus final-target diagnostics. Re-run the
  same four-way exposure/depth screen and seed1701 under a unique v3 group.
- Authorized new Stage 5A budget: three registered groups including v3 and
  confirmation. Consumed: 0/3 before launch. V1/v2 bugged runs do not count.
  Follow-ups may change only depth curriculum, ramp duration, or the common
  Huber-weight scale, based on failure evidence. An unmet gate or missing next
  JSON alone must not stop justified in-budget 5A improvement. No automatic
  complete evaluator; preserve all models and email the existing round summary.

### Stage 5A v3 validation and first budgeted group

- CPU regression passed on the immutable remote environment. Actual moderate
  shoulder samples span 0.7801--0.9798 m and initialized wrist heights span
  0.4155--0.6492 m; deep samples span 0.6212--0.8996 m with wrists
  0.2571--0.5651 m. Subset isolation, fresh symmetric reach, curriculum
  interpolation, and masked-metric normalization passed. Full-pose dynamics
  feasibility is still a training question, not established by this test.
- `g1_stage5a_sampling_smoke_v3` used one H20, 256 environments (per-rank/global),
  20 PPO iterations, and 100% moderate height tasks without the normal warmup.
  It exited 0. Real simulation logged final shoulder target 0.8452 m and wrist
  target 0.4961 m, NOT zero, and finite PPO/reward/command diagnostics. Short
  random-policy episodes still fell; this is software validation only. The full
  group restores the normal warmup/ramp and the original v2 reward weights.
- New summary tool reproduced v2 35%-moderate conditional wrist error
  0.249186 m and shoulder error 0.217205 m from its TensorBoard, confirming
  normalization and historical-name compatibility.
- Authorized next launch: `g1_stage5a_sampling_screen_v3`, first of THREE
  budgeted groups (1/3 on launch). Four independent H20 runs, same seed1701,
  4,096 environments per rank/run; 16,384 concurrent across the group, not DDP.
  Each run is 5,000 PPO iterations. No automatic complete held-out evaluator.
  Watcher must analyze the four siblings once, email the existing summary, and
  use the remaining two groups for justified controlled 5A improvement or seed
  confirmation, rather than stopping merely because this first group misses
  the gate. Stage 5B remains paused.

### Stage 5A corrected-sampling screen result (`g1_stage5a_sampling_screen_v3`)

- Hypothesis and exact change: this first budgeted Stage 5A group repeated the
  35%/50% height-task exposure by moderate/deep shoulder-target-depth screen
  after fixing indexed sampling writeback.  It retained the coupled wrist
  targets, height-active Huber losses, zero planar twist, normal warmup/ramp,
  PPO/rewards, 0.50 reach and asymmetric probabilities, residual wrist-height
  range +/-0.02 m, low-reach extension 0.06--0.14 m, and fixed seed 1701.
  Every sibling used one H20 and 4,096 environments per run (4,096 global per
  run; 16,384 concurrent across the four independent runs), for 5,000 PPO
  iterations.  The plan contains no automatic held-out evaluation.
- Status: all four named siblings exited 0 and completed normally at iteration
  4999.  Their TensorBoard event files contain 5,000 finite samples for every
  inspected scalar and final `model_4999.pt` checkpoints are preserved.  Exit
  codes, log tails, and final-100 windows show no traceback, numerical issue,
  CUDA/Warp/infrastructure failure, NaN, or divergence.  Remote status after
  completion showed no tmux session, GPU process, or active full DDP job.
- Command validation and final-100 conditional diagnostics: actual height
  command fractions were 0.3480/0.5001/0.3512/0.4960 for
  35%-moderate/50%-moderate/35%-deep/50%-deep.  Corresponding actual final
  shoulder/wrist targets were 0.8795/0.5307, 0.8799/0.5313,
  0.7590/0.4104, and 0.7603/0.4117 m, respectively, consistent with the
  configured moderate (0.78--0.98 m) and deep (0.62--0.90 m) curricula and
  the approximately 0.349 m coupled vertical gap.  Conditional height-active
  wrist errors were 0.475/0.369/0.463/0.405 cm; shoulder errors were
  0.562/0.396/0.524/0.403 cm; and non-height wrist errors were
  0.418/0.427/0.438/0.559 cm.  These are normalized conditional values, not
  raw `_masked` population numerators.
- Gate and provisional ranking: all siblings satisfy the stationary 5A
  training diagnostic gate: both height errors are below 4/5 cm, final-100
  episode length is 599.22/599.45/598.83/598.97 of 600 (above 590), non-height
  wrist error is below 2 cm, and backward-lean projection is only
  0.00045/0.00059/0.00109/0.00075 with zero or 0.00031 termination count per
  logged batch.  `active050_moderate` is the provisional fixed-seed winner:
  it has the lowest height-active wrist and shoulder errors, nearly full
  episodes, the lowest action acceleration (0.7411), and lower foot stagger
  (0.1278) than the other candidates.  This is training evidence only and not
  a final-policy or held-out claim.
- Decision: promote the unchanged 50%-active moderate curriculum to
  independent-seed confirmation rather than tune a passing configuration.
  `g1_stage5a_sampling_confirm_v3` is the second of at most three authorized
  5A registered groups (2/3 on launch): four one-GPU runs with seeds
  101/211/307/401, 4,096 environments per run, and 5,000 iterations.  It
  omits an evaluation block; all checkpoints remain preserved and Stage 5B
  stays paused until confirmation is reviewed.

### Stage 5A independent-seed confirmation result (`g1_stage5a_sampling_confirm_v3`)

- Hypothesis and exact change: this second of at most three authorized Stage 5A
  groups held the corrected-sampling, 50%-height-active moderate curriculum
  unchanged (coupled shoulder range 0.78--0.98 m, wrist residual +/-0.02 m,
  reach/asymmetric probability 0.50, zero planar twist, normal warmup/ramp,
  rewards, and PPO) and tested independent seeds 101/211/307/401.  Each
  sibling used one H20 with 4,096 environments per run (4,096 global per run;
  16,384 concurrent across the group) for 5,000 PPO iterations.  The group
  has no automatic held-out evaluation.
- Status: all supplied exit codes are 0; every run reached iteration 4999,
  preserved a `model_4999.pt`, and has 5,000 finite samples for all inspected
  TensorBoard scalars.  Preserved train-log tails contain no traceback, NaN,
  divergence, CUDA/Warp, or infrastructure error.  Post-completion remote
  status has no tmux session, GPU process, or active full-DDP job.
- Command validation and aggregate final-100 conditional diagnostics: actual
  height fractions are 0.5031/0.5024/0.4999/0.4970, and actual active final
  shoulder/wrist targets are 0.8792/0.5305, 0.8807/0.5320, 0.8812/0.5328, and
  0.8804/0.5318 m.  Thus targets match the configured moderate range and keep
  the coupled vertical gap at 0.3484--0.3487 m.  Conditional height-active
  wrist error is 0.367/0.443/0.416/0.475 cm (mean 0.425, sample SD 0.047);
  shoulder error is 0.357/0.454/0.427/0.464 cm (mean 0.425, SD 0.047); and
  non-height wrist error is 0.384/0.480/0.491/0.590 cm (mean 0.486, SD 0.085).
  These are normalized conditional values, never raw `_masked` numerators.
  Final-100 episode length is 599.46/599.06/599.18/598.57 (mean 599.07), and
  backward-lean projection is 0.00111/0.00080/0.00071/0.00065 with at most
  0.00125 backward-lean terminations per logged batch.  Action acceleration
  is 0.735/0.822/0.797/0.859 and foot stagger 0.159/0.132/0.138/0.146 m;
  these are retained as provisional training-quality diagnostics only.
- Gate and decision: every independent seed passes the fixed stationary 5A
  gate (height wrist <4 cm, shoulder <5 cm, episode length >=590, non-height
  wrist <2 cm, and no systematic backward-lean failure).  The confirmation
  establishes stationary lowering across seeds, but does not make a final
  policy claim or replace the user's visual/fixed-held-out review.  Promote
  to the next defined conceptual factor, gradual planar twist: launch the
  registered fixed-seed `g1_stage5b_twist_screen_v2` screen, retaining the
  confirmed 50%-moderate stationary mixture while crossing only twist exposure
  (25%/50%) and low/medium amplitude.  No evaluation block is included and all
  confirmation checkpoints remain preserved.

### Stage 5B gradual planar-twist screen v2 result (`g1_stage5b_twist_screen_v2`)

- Hypothesis and exact change: following the independently confirmed stationary
  Stage 5A 50%-active moderate curriculum, this fixed-seed screen changed only
  planar-twist exposure (25% or 50%) and command amplitude (low: x/y/yaw
  +/-0.10/+/-0.05/+/-0.10; medium: +/-0.20/+/-0.10/+/-0.20). All runs kept
  the 50% coupled moderate shoulder-height mixture, wrist/reach settings,
  rewards, PPO, warmup, and seed 1801. Each sibling used one H20 with 4,096
  environments per run (4,096 global per run; 16,384 concurrent across the
  four independent runs) for 5,000 PPO iterations. The plan has no automatic
  held-out evaluation.
- Status: all four supplied exit codes are 0 and all runs reached iteration
  4999. Each remote TensorBoard stream has 5,000 finite samples for inspected
  scalars, its final 100-window has no non-finite tag, and all final
  `model_4999.pt` checkpoints (11 MiB) are preserved. Direct train-log tails
  contain no traceback, NaN, divergence, CUDA/Warp, configuration, or
  infrastructure failure. Post-completion remote status has no tmux session,
  GPU process, or active full-DDP job.
- Command validation and final-100 conditional diagnostics, ordered as
  25%-low / 50%-low / 25%-medium / 50%-medium: actual height fractions were
  0.5024/0.4983/0.4967/0.4987; active final shoulder/wrist targets were
  0.8806/0.5321, 0.8808/0.5324, 0.8795/0.5307, and 0.8794/0.5308 m. These
  match the moderate configured range and retain the coupled 0.3484--0.3488 m
  vertical gap. Conditional height-active wrist errors were
  0.488/0.392/0.570/0.508 cm; shoulder errors 0.468/0.430/0.536/0.524 cm; and
  non-height wrist errors 0.639/0.445/0.681/0.705 cm. These values are
  normalized conditional metrics, not raw `_masked` population numerators.
  Mean final-100 episode lengths were 598.41/599.12/597.73/598.22 of 600, and
  backward-lean termination was zero in every sibling; backward-lean
  projections were only 0.00054/0.00076/0.00096/0.00056.
- Provisional comparison and stopping decision: `active50_low` provides the
  best combined training diagnostics: lowest x-y/yaw velocity error
  (0.1640/0.6337), lowest height-active wrist error, lowest action acceleration
  (0.7807), and the highest episode length. Medium amplitude is consistently
  worse on velocity tracking (x-y 0.1951/0.1993, yaw 0.7946/0.7001), wrist
  retention, and action acceleration (0.9189/0.8269); it also has higher
  bad-orientation termination counts. `active50_low` is consequently the
  conservative provisional Stage 5B checkpoint:
  `/home/dev/unitree_rl_mjlab/logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-12_11-12-34/model_4999.pt`.
  This is not a final policy claim: no complete held-out matrix was run, and
  the user retains final visual/fixed-suite judgment. Stop autonomous
  training: Stage 5B defines no further controlled factor or quantitative
  promotion criterion, and Stage 6 requires the absent mapped-trajectory
  split/interface contract. Launching another run would not be evidence-based.

### Stage 5B explicit clutch repair v3 (user-authorized)

- Diagnosis: 5B v2 retained world-fixed wrists while requesting sustained
  locomotion, and applied quiet-body penalties unconditionally. Tiny wrist
  errors do not establish mobile wrist control. Treat its candidate only as
  a preserved diagnostic baseline, not as successful Stage 5B promotion.
- User authorized three episode types: transport (clutch=1), bounded anchored
  adjustment (clutch=0, small twist), and autonomous balance (clutch=0, zero
  twist). The terminal mixture is 25%/25%/50%. Transport world wrist poses are
  updated by commanded-reference translation/yaw, not actual robot motion;
  anchored wrists remain fixed in world. Shoulder height remains world-z.
  Adjustment starts at 3 s, lasts at most 2 s, and has commanded cumulative
  path/yaw budgets 0.08 m/0.12 rad. These do not restrict actual recovery steps.
- Code commit `34ba05c` adds the binary clutch observation (534 rather than
  531 history observations), motion-request gating for quiet-body/feet rewards,
  and accumulated moving-only wrist/height/velocity diagnostics. Old command
  defaults retain their original dimensions; new policies start from scratch.
  Existing `Metrics/twist/error_vel_*` normalization depends on the long
  resampling interval and must NOT be compared with earlier runs. Use the new
  normalized moving-only metrics and commanded magnitudes instead.
- Both CPU checks passed on the immutable remote training Python, including
  noncontiguous reset subsets, explicit sample assignment, independent
  transport translation/yaw, anchored world-pose retention, bounded command
  stopping, and moving statistics surviving terminal post-adjustment stops.
- `g1_stage5b_clutch_smoke_v3` used one GPU, 256 environments, 100 iterations,
  immediate full command difficulty: exit 0, finite PPO/TB, all modes sampled.
  Random-initialization falls were frequent; this is runtime validation only,
  not task mastery. Final-version gradual-course validation is
  `g1_stage5b_clutch_smoke_v3_ramp` (one GPU, 256 environments, 200 iterations,
  no warmup/6k ramp): exit 0, all 200 inspected TB samples finite. Final-20
  balance/transport/adjust fractions were 0.5232/0.2504/0.2264; episode length
  was only 90.65/600 and adjustment moving coverage only 0.0000789 of samples.
  This confirms runtime/telemetry, NOT mastery or reliable adjustment statistics.
  Normal full-run warmup remains essential. These smoke runs do not consume
  the registered budget.
- Reviewed next group: `experiments/stage5b_clutch_screen_v3.json`; four
  independent one-H20 runs, 4,096 environments globally per run and 16,384
  concurrently, seed 1802, 5,000 iterations, normal 30k warmup/60k ramp.
  Hold the entire three-mode/height distribution fixed and vary only a common
  velocity-tracking reward multiplier 1/2/4/8. Gradual validation completed;
  authorize the reviewed full launch, consuming group 1/3; at most one
  justified single-factor follow-up and one independent-seed confirmation.
  No automatic full evaluator. Email only the existing analysis summaries.

### Stage 5B clutch-v3 tracking-scale screen result (`g1_stage5b_clutch_screen_v3`)

- Hypothesis and exact change: the first of at most three authorized clutch-v3
  groups held the complete three-mode (50% balance / 25% transport / 25%
  adjustment), world-z height, coupled moderate shoulder/wrist, clutch, command
  ramp, PPO, and seed-1802 settings fixed. It screened only common linear/yaw
  tracking-reward scales 1/2/4/8 (linear weights 1/2/4/8 and yaw weights
  0.5/1/2/4). Each sibling used one H20 and 4,096 environments per run (4,096
  global per run; 16,384 concurrent across the four independent runs) for 5,000
  iterations. The plan contained no automatic held-out evaluation.
- Status: all four named siblings exited 0, reached iteration 4999, preserved
  final checkpoints, and logged 5,000 finite TensorBoard samples for every
  inspected scalar. Remote log tails had no traceback, NaN/divergence, or
  configuration/CUDA/Warp/infrastructure failure; post-run remote status had no
  tmux session or GPU process. All three mode fractions and nonzero moving
  transport/adjustment fractions were present, so their conditional diagnostics
  are available. Actual final height-active shoulder/wrist targets were
  0.8799/0.5312, 0.8802/0.5317, 0.8795/0.5308, and 0.8793/0.5308 m for
  scales 1/2/4/8, matching the configured 0.78--0.98 m moderate curriculum and
  coupled approximately 0.349 m gap.
- Final-100 normalized conditional diagnostics, ordered by scale 1/2/4/8:
  height-active wrist/shoulder errors were 0.423/0.444, 0.416/0.382,
  0.531/0.487, and 0.710/0.716 cm; balance wrist errors were
  0.490/0.461/0.505/0.697 cm; mean episode length was
  598.87/599.08/598.89/599.07 of 600; and backward-lean projection was
  0.00050/0.00092/0.00020/0.00033. Thus all stationary retention and per-mode
  wrist (<4 cm) gates pass. Moving transport xy/yaw errors improved
  0.116/0.493, 0.094/0.434, 0.086/0.367, and 0.080/0.289 m/s/rad/s; adjustment
  values similarly improved to 0.073/0.275 at scale 8. However, every sibling
  misses the fixed moving gate (xy <0.05, yaw <0.10, and each below half the
  respective moving commanded magnitude). The scale-8 candidate is the best
  provisional training diagnostic but is not a promoted policy or held-out
  claim.
- Baseline comparison and decision: the monotonic velocity improvement across
  the only screened factor supports exactly one evidence-based follow-up within
  the remaining 5B budget; low-scale values cannot meet the fixed gate, while
  wrist retention remains well inside its bound. The reviewed follow-up plan
  `experiments/stage5b_clutch_tracking_followup_v3.json` holds every task and
  PPO setting fixed and screens only higher common scales 12/16/24. A 256-env,
  one-GPU validation at the highest scale must first establish finite runtime
  and PPO before this registered group may launch. This would consume group 2/3;
  the sole remaining group is reserved for independent-seed confirmation only
  if an unchanged winner passes the fixed training gate.
- Validation launch: `g1_stage5b_clutch_tracking_validate_v3` is a one-H20,
  256-environment (per-rank/global) 200-iteration runtime/PPO validation of the
  highest proposed scale (linear/yaw weights 24/12), with the same three-mode
  distribution and all other settings unchanged. It does not consume the 5B
  registered-group budget. Its completion must be inspected in a new watcher
  turn; no full group is launched in this completion-triggered turn.

### Stage 5B clutch-v3 high-scale validation result (`g1_stage5b_clutch_tracking_validate_v3`)

- Hypothesis and exact change: runtime/PPO validation of the highest proposed
  common tracking scale only (linear/yaw 24/12), with the reviewed three-mode
  clutch distribution and all other Stage 5B v3 settings unchanged.  One H20,
  256 environments per rank/global, seed 1802, and 200 PPO iterations; this is
  a validation, not a registered-group budget entry.
- Status: exit code 0; the run completed all 200 iterations.  Its train-log
  tail has no traceback or infrastructure/configuration failure, and all 200
  inspected TensorBoard scalars are finite.  PPO progressed (final mean reward
  24.41, surrogate loss -0.0199, value loss 54.10), and the remote host is
  idle after completion.  This passes the required finite runtime/PPO gate.
- Diagnostics: the normal 30k-step warmup left this short run entirely in
  balance mode in the final-100 window (balance fraction 1.0; transport,
  adjustment, height, and moving fractions 0).  Consequently the normalized
  moving and height conditional metrics are unavailable, not zero or a gate
  pass.  The short random-policy episode length (76.57/600) and balance wrist
  error (0.600 m) are expected validation-only diagnostics and are not compared
  with full-run learning gates.  No NaN or non-finite tag was present.
- Decision: the risky high-scale change is runtime-safe and the prior screen's
  monotonic tracking evidence still supports the single allowed follow-up.
  Launch the reviewed registered `g1_stage5b_clutch_tracking_followup_v3`
  screen at scales 12/16/24.  It consumes Stage 5B group 2/3: three independent
  one-GPU runs with 4,096 environments per run (4,096 global each; 12,288
  concurrent total), fixed seed 1802, and 5,000 iterations.  It has no
  automatic held-out evaluator; preserve every checkpoint and reserve group
  3/3 solely for independent-seed confirmation if an unchanged candidate
  passes the fixed three-mode training diagnostics.

### Stage 5B clutch-v3 high-scale follow-up result (`g1_stage5b_clutch_tracking_followup_v3`)

- Hypothesis and exact change: the second of three authorized groups held the
  entire clutch-v3 task, three-mode fractions, command ranges/ramp, stationary
  height settings, PPO configuration, and fixed seed 1802 unchanged. It tested
  only common linear/yaw velocity-tracking reward scales 12/16/24 (linear
  weights 12/16/24; yaw weights 6/8/12). The three independent one-H20 runs
  used 4,096 environments each (4,096 per-rank/global per run; 12,288
  concurrently) for 5,000 iterations. No automatic held-out evaluator was run.
- Status: supplied exit codes are 0 for `track12`, `track16`, and `track24`;
  each has a DONE marker, completed a 5,000-sample finite TensorBoard history,
  and preserved `model_4999.pt`. Read-only remote status after completion showed
  no tmux sessions or GPU processes. TensorBoard summaries found no non-finite
  tag, and the completed-run logs/provenance contain no reported code,
  configuration, CUDA/Warp, or infrastructure failure. Checkpoints are
  respectively under `2026-09-12_20-17-07`, `20-17-13`, and `20-17-19` in
  `logs/rsl_rl/g1_wrist_recovery_teacher/`.
- Command and stationary diagnostics: actual final height-active shoulder/wrist
  targets were 0.8804/0.5318, 0.8810/0.5325, and 0.8797/0.5313 m, consistent
  with the configured 0.78--0.98 m shoulder range and 0.3485--0.3486 m coupled
  gap. Final-100 normalized conditional height wrist/shoulder errors were
  0.894/1.052, 1.000/1.074, and 1.741/1.089 cm; balance wrist errors were
  0.860/1.192/1.980 cm; and mean episode length was 598.60/598.11/596.65 of
  600 for scales 12/16/24. Thus stationary and per-mode wrist retention remain
  inside the fixed bounds, but this does not establish moving-task success.
- Moving diagnostics and baseline comparison: normalized transport moving
  xy/yaw errors were 0.0666/0.2308, 0.0724/0.2547, and 0.0708/0.2274 m/s and
  rad/s; adjustment moving errors were 0.0642/0.2190, 0.0657/0.2313, and
  0.0657/0.2098. Corresponding transport commanded magnitudes were about
  0.059/0.050 and adjustment magnitudes about 0.0295/0.0297. Scale 12 improves
  over the preceding scale-8 provisional transport result (0.080/0.289), but
  every sibling misses the fixed xy <0.05 and yaw <0.10 gates and is also above
  half its respective moving commanded magnitude. Raising the scale from 12 to
  16/24 worsens xy tracking and wrist/foot/action diagnostics, so there is no
  monotonic evidence for another increase. These are normalized conditional
  values, not raw `_masked` numerators, and remain training diagnostics only.
- Decision: stop autonomous Stage 5B work. This group consumes 2/3; the only
  remaining group is explicitly reserved for independent-seed confirmation of
  an unchanged candidate that passes the fixed three-mode gate. No sibling
  passes, so confirmation is not authorized, and the plan permits no further
  controlled intervention or gate relaxation. Preserve all checkpoints;
  `track12/model_4999.pt` is the least-bad provisional moving candidate from
  this group, not a promoted or held-out-validated policy. Stage 6 remains
  blocked on the mapped trajectory data/interface.
