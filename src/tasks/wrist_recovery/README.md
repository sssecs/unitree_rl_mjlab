# G1 pure-RL wrist recovery

This task trains one 29-DoF PPO policy to hold two world-frame Cartesian wrist
targets while the rest of the body balances and takes corrective steps. It uses no
motion prior, imitation objective, CVAE, VLA, or future reference frames.

Task ID: `Unitree-G1-Wrist-Recovery-Teacher`.

The first 30,000 control steps train quiet wrist holding. Over the next 60,000
steps, forward-reaching commands, persistent wrist payloads, and root-velocity
pushes ramp to full strength. The teacher actor and critic both receive exact state,
wrist errors, foot state, applied wrist wrench, and the sampled push impulse.

The first experiment tests one hypothesis carried over from the Isaac Lab pilot:
penalizing foot motion only while the hold task and base are already stable should
preserve quiet standing without suppressing necessary recovery steps.

## Local playback

Fetch the best balanced checkpoint from the training server:

```bash
./tools/remote_fetch_best_wrist_model.sh
```

Run one local environment with the automatically selected viewer:

```bash
./tools/play_best_wrist_model.sh
```

Use `--viewer native` for a desktop window or `--viewer viser` for the browser
viewer. Extra arguments are forwarded to `scripts/play.py`; for example, record
200 frames with `./tools/play_best_wrist_model.sh --video True`.
