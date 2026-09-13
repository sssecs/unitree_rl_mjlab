"""Short nominal, no-push diagnostic; not the frozen held-out evaluator."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.wrist_recovery.play_profile import apply_stage5b_clutch_profile
from mjlab.utils.lab_api.math import quat_apply


@dataclass
class Config:
  checkpoint_file: str
  output_dir: str = "results/wrist_clutch_diagnostic"
  num_envs: int = 12
  seed: int = 9187
  horizon_s: float = 12.
  device: str = "cuda:0"


def main():
  import src.tasks  # noqa: F401
  c = tyro.cli(Config)
  if c.num_envs < 6 or c.horizon_s <= 5:
    raise ValueError("Use >=6 envs and >5s to cover adjustment and its stopping")
  task = "Unitree-G1-Wrist-Recovery-Teacher"
  cfg = load_env_cfg(task, play=False)
  apply_stage5b_clutch_profile(cfg)
  cfg.scene.num_envs = c.num_envs
  cfg.seed = c.seed
  cfg.episode_length_s = c.horizon_s + .04
  for name in ("push_robot", "hand_payload", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(name, None)
  for group in cfg.observations.values():
    group.enable_corruption = False
  raw = ManagerBasedRlEnv(cfg=cfg, device=c.device)
  agent = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent), device=c.device)
  runner.load(c.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=c.device)
  policy = runner.get_inference_policy(device=c.device)
  raw.reset(seed=c.seed)
  twist = raw.command_manager.get_term("twist")
  wrists = raw.command_manager.get_term("wrists")
  index = torch.arange(c.num_envs, device=c.device) % 6
  names = ("balance", "transport_forward", "transport_backward", "transport_yaw", "adjust_forward", "adjust_yaw")
  twist.mode.copy_(torch.tensor([0,1,1,1,2,2], device=c.device)[index])
  commands = torch.tensor([[0,0,0],[.1,0,0],[-.1,0,0],[0,0,.1],[.04,0,0],[0,0,.06]], device=c.device)
  twist.sampled_twist.copy_(commands[index])
  twist._update_command()
  wrists._update_command()
  for _ in range(3):
    raw.obs_buf = raw.observation_manager.compute(update_history=True)
  obs = env.get_observations()
  origin = wrists.robot.data.root_link_pos_w.clone()
  axis = torch.zeros(c.num_envs,3,device=c.device); axis[:,0]=1
  forward = quat_apply(wrists.robot.data.root_link_quat_w,axis)[:,:2]
  forward /= torch.linalg.vector_norm(forward,dim=-1,keepdim=True).clamp_min(1e-6)
  active = torch.ones(c.num_envs, dtype=torch.bool, device=c.device)
  feet0 = wrists.robot.data.body_link_pos_w[:, wrists.foot_body_ids].clone()
  last_landing = feet0.clone()
  air = torch.zeros(c.num_envs,2, device=c.device)
  trace, landings = [], []
  with torch.no_grad():
    for step in range(round(c.horizon_s/raw.step_dt)):
      requested = twist.command.clone()
      obs, _, done, _ = env.step(policy(obs))
      active &= ~done.bool()
      data = wrists.robot.data
      contact = raw.scene["feet_ground_contact"].data.found.reshape(c.num_envs, -1)[:, :2] > 0
      feet = data.body_link_pos_w[:, wrists.foot_body_ids]
      # Only count touchdown after >=100ms of flight; reject contact chatter.
      touchdown = contact & (air >= .1) & active[:,None]
      for i,j in touchdown.nonzero().tolist():
        landings.append({"env":i,"case":names[int(index[i])],"time_s":step*raw.step_dt,
          "foot":j,"same_foot_displacement_m":float(torch.linalg.vector_norm(feet[i,j,:2]-last_landing[i,j,:2]))})
      last_landing[touchdown] = feet[touchdown]
      air = torch.where(contact, 0., air + raw.step_dt)
      error = torch.linalg.vector_norm(wrists.desired_wrist_pos_w-wrists.robot_wrist_pos_w,dim=-1).mean(-1)
      slip = (torch.linalg.vector_norm(data.body_link_lin_vel_w[:,wrists.foot_body_ids,:2],dim=-1)*contact).mean(-1)
      for i in active.nonzero().flatten().tolist():
        trace.append({"env":i,"case":names[int(index[i])],"time_s":(step+1)*raw.step_dt,
          "cmd_x":float(requested[i,0]),"cmd_y":float(requested[i,1]),"cmd_yaw":float(requested[i,2]),
          "actual_x":float(data.root_link_lin_vel_b[i,0]),"actual_y":float(data.root_link_lin_vel_b[i,1]),
          "actual_yaw":float(data.root_link_ang_vel_b[i,2]),
          "world_dx":float(data.root_link_pos_w[i,0]-origin[i,0]),
          "world_dy":float(data.root_link_pos_w[i,1]-origin[i,1]),"wrist_error_m":float(error[i])})
        trace[-1]["initial_heading_progress_m"] = float(((data.root_link_pos_w[i,:2]-origin[i,:2])*forward[i]).sum())
        trace[-1]["contact_foot_speed_mps"] = float(slip[i])
  summary = {}
  for name in names:
    rows = [r for r in trace if r["case"]==name]
    moving = [r for r in rows if abs(r["cmd_x"])+abs(r["cmd_yaw"])>1e-5]
    def mean(items,key):
      return sum(r[key] for r in items)/len(items) if items else None
    steps = [r for r in landings if r["case"]==name]
    last = {r["env"]:r for r in rows}
    members = (index == names.index(name))
    summary[name] = {"falls":int((~active & members).sum()),"envs":int(members.sum()),
      "moving_samples":len(moving),"wrist_mean_m":mean(rows,"wrist_error_m"),
      "wrist_peak_m":max((r["wrist_error_m"] for r in rows),default=None),
      "moving_cmd_x":mean(moving,"cmd_x"),"moving_actual_x":mean(moving,"actual_x"),
      "moving_cmd_yaw":mean(moving,"cmd_yaw"),"moving_actual_yaw":mean(moving,"actual_yaw"),
      "mean_abs_yaw":sum(abs(r["actual_yaw"]) for r in rows)/len(rows) if rows else None,
      "final_world_dx":mean(list(last.values()),"world_dx"),
      "final_initial_heading_progress_m":mean(list(last.values()),"initial_heading_progress_m"),
      "contact_foot_speed_mps":mean(rows,"contact_foot_speed_mps"),
      "touchdowns":len(steps),"same_foot_stride_m":mean(steps,"same_foot_displacement_m")}
  raw.close()
  out = Path(c.output_dir)
  out.mkdir(parents=True,exist_ok=False)
  for filename,rows in (("episodes.csv",trace),("touchdowns.csv",landings)):
    with (out/filename).open("w") as f:
      if rows:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
  (out/"summary.json").write_text(json.dumps(summary,indent=2))
  (out/"manifest.json").write_text(json.dumps(asdict(c),indent=2))
  print(json.dumps(summary,indent=2))


if __name__ == "__main__":
  main()
