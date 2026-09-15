"""Bounded read-only task diagnostics; no rewards, actions or observations."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import mujoco
import torch


def audit_collision_model(model):
  names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '' for i in range(model.nbody)]
  arms = {i for i,n in enumerate(names) if any(x in n for x in ('shoulder_', 'elbow_', 'wrist_'))}
  legs = {i for i,n in enumerate(names) if any(x in n for x in ('hip_', 'knee_', 'ankle_'))}
  arm_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in arms and (model.geom_contype[i] or model.geom_conaffinity[i])]
  leg_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in legs and (model.geom_contype[i] or model.geom_conaffinity[i])]
  assert arm_geoms and leg_geoms, 'Missing active arm/leg collision geometries'
  rng = np.random.default_rng(2401)
  data = mujoco.MjData(model)
  probes = 0
  for _ in range(128):
    data.qpos[:] = model.qpos0
    for j in range(model.njnt):
      typ, adr = model.jnt_type[j], model.jnt_qposadr[j]
      if typ == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[adr+2] = 1.
      elif typ == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_bodyid[j] in arms | legs:
        lo, hi = model.jnt_range[j]
        data.qpos[adr] = rng.uniform(max(lo,-1.5), min(hi,1.5))
    mujoco.mj_forward(model, data)
    for c in data.contact:
      a,b = model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
      if (a in arms and b in legs) or (b in arms and a in legs):
        if c.dist <= 0: probes += 1
  assert probes > 0, 'No physical arm-leg contacts in synthetic CPU audit'
  return dict(active_arm_geoms=len(arm_geoms), active_leg_geoms=len(leg_geoms),
              synthetic_contact_count=probes, synthetic_poses=128,
              scope='CPU forward of actual compiled model; probe poses are NOT training references')


class OperationDiagnostics:
  def __init__(self, command):
    n, device = command.num_envs, command.device
    self.stats = torch.zeros(n, 10, device=device)
    self.previous_pos = torch.zeros(n,2,3,device=device)
    self.valid = torch.zeros(n,dtype=torch.bool,device=device)
    self.previous_contact = self.valid.clone()
    self.recovering = self.valid.clone()
    self.age = torch.zeros(n,device=device)
    self.streak = self.age.clone()
    for name in ('contact_fraction','contact_wrist_error_masked','release_events_masked',
                 'recoveries_masked','recovery_time_masked','fd_reported_discrepancy',
                 'moving_fraction','fd_velocity_error_masked','reported_velocity_error_masked'):
      command.metrics['diag_'+name] = torch.zeros(n,device=device)
    run_dir = os.environ.get('AUTOTUNE_RUN_DIR')
    self.path = Path(run_dir)/'wrist_trace.csv' if run_dir else None
    audit = audit_collision_model(command._env.sim.mj_model)
    print('[INFO] Arm-leg collision audit:', json.dumps(audit), flush=True)
    if run_dir:
      (Path(run_dir)/'arm_leg_collision_audit.json').write_text(json.dumps(audit,indent=2))

  def reset(self, ids):
    self.stats[ids] = 0
    self.valid[ids] = False
    self.previous_contact[ids] = False
    self.recovering[ids] = False
    self.age[ids] = self.streak[ids] = 0

  def update(self, command):
    actual_pos = command.robot_wrist_pos_w
    actual_vel = command.robot.data.body_link_lin_vel_w[:,command.wrist_body_ids]
    fd_vel = (actual_pos-self.previous_pos)/command._env.step_dt
    valid = self.valid & ~command.needs_initialization
    force = command._env.scene['arm_leg_contact'].data.force_history
    assert force is not None
    peak_force = torch.linalg.vector_norm(force,dim=-1).amax(dim=(1,2))
    contact = (peak_force > 1.) & valid
    error = torch.linalg.vector_norm(actual_pos-command.desired_wrist_pos_w,dim=-1).max(-1).values
    released = self.previous_contact & ~contact & valid
    self.recovering |= released
    self.age = torch.where(released,0.,self.age)
    self.recovering &= ~contact & valid
    self.age += self.recovering*command._env.step_dt
    self.streak = torch.where(self.recovering & (error<.03),self.streak+command._env.step_dt,0.)
    recovered = self.recovering & (self.streak>=.20)
    s = self.stats
    s[:,0] += valid
    s[:,1] += contact
    s[:,2] += contact*error
    s[:,3] += released
    s[:,4] += recovered
    s[:,5] += recovered*self.age
    s[:,6] += valid*torch.linalg.vector_norm(fd_vel-actual_vel,dim=-1).mean(-1)
    moving = valid & command.continuous_active & (command.elapsed >= command.cfg.reach_delay_s+command.cfg.reach_duration_s) & (torch.linalg.vector_norm(command.target_lin_vel_w,dim=-1).mean(-1)>.001)
    s[:,7] += moving
    s[:,8] += moving*torch.linalg.vector_norm(fd_vel-command.target_lin_vel_w,dim=-1).mean(-1)
    s[:,9] += moving*torch.linalg.vector_norm(actual_vel-command.target_lin_vel_w,dim=-1).mean(-1)
    total = s[:,0].clamp_min(1)
    for column,name in ((1,'contact_fraction'),(2,'contact_wrist_error_masked'),(3,'release_events_masked'),
                        (4,'recoveries_masked'),(5,'recovery_time_masked'),(6,'fd_reported_discrepancy'),
                        (7,'moving_fraction'),(8,'fd_velocity_error_masked'),(9,'reported_velocity_error_masked')):
      command.metrics['diag_'+name] = s[:,column]/total
    self.recovering &= ~recovered
    step = command._env.common_step_counter
    if self.path and step % command.cfg.diagnostics_trace_period_steps < command.cfg.diagnostics_trace_window_steps:
      columns = torch.cat((actual_pos.flatten(1),command.desired_wrist_pos_w.flatten(1),actual_vel.flatten(1),fd_vel.flatten(1),command.target_lin_vel_w.flatten(1),peak_force[:,None],command.elapsed[:,None],command.persistent_active[:,None].float(),valid[:,None].float(),command.robot.data.joint_pos[:,command.leg_joint_ids],command.robot.data.joint_vel[:,command.leg_joint_ids],command._env.command_manager.get_term('twist').command),dim=-1)[:2].detach().cpu().tolist()
      header = not self.path.exists()
      with self.path.open('a',newline='') as f:
        writer=csv.writer(f)
        if header: writer.writerow(['step','env']+[f'{kind}_{hand}_{axis}' for kind in ('actual_pos','target_pos','reported_vel','fd_vel','target_vel') for hand in ('left','right') for axis in ('x','y','z')]+['arm_leg_force','elapsed','persistent','fd_valid']+[f'leg_{kind}_{i}' for kind in ('pos','vel') for i in range(len(command.leg_joint_ids))]+['base_cmd_x','base_cmd_y','base_cmd_yaw'])
        for i,row in enumerate(columns): writer.writerow([step,i,*row])
    self.previous_pos.copy_(actual_pos)
    self.valid.fill_(True)
    self.previous_contact.copy_(contact)
