"""CPU regression of reset, finite differences and contact-associated recovery."""
import runpy
from pathlib import Path
from types import SimpleNamespace as NS
import torch

cls=runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/tasks/wrist_recovery/mdp/operation_diagnostics.py'))['OperationDiagnostics']
d=cls.__new__(cls)
d.stats=torch.zeros(2,10); d.previous_pos=torch.zeros(2,2,3)
d.valid=torch.zeros(2,dtype=torch.bool); d.previous_contact=d.valid.clone(); d.recovering=d.valid.clone()
d.age=torch.zeros(2); d.streak=d.age.clone(); d.path=None
force=torch.zeros(2,1,4,3)
c=NS(robot_wrist_pos_w=torch.zeros(2,2,3), desired_wrist_pos_w=torch.zeros(2,2,3),
     robot=NS(data=NS(body_link_lin_vel_w=torch.zeros(2,2,3))), wrist_body_ids=[0,1],
     needs_initialization=torch.zeros(2,dtype=torch.bool), continuous_active=torch.ones(2,dtype=torch.bool),
     elapsed=torch.full((2,),4.), target_lin_vel_w=torch.zeros(2,2,3), metrics={},
     cfg=NS(reach_delay_s=1.,reach_duration_s=2.),
     _env=NS(step_dt=.02,common_step_counter=1,scene={'arm_leg_contact':NS(data=NS(force_history=force))}))
d.update(c)
c.robot_wrist_pos_w[:]=.002; c.robot.data.body_link_lin_vel_w[:]=.1
d.update(c)
assert d.stats[:,6].abs().max()<1e-6
c.robot_wrist_pos_w[:]=.1; force[0,0,0,0]=5
d.update(c); assert d.stats[0,1]==1 and d.stats[1,1]==0
force.zero_(); c.desired_wrist_pos_w[:]=c.robot_wrist_pos_w
for _ in range(12): d.update(c)
assert d.stats[0,3]==d.stats[0,4]==1 and d.stats[0,5]>=.20
d.reset(torch.tensor([0])); c.robot_wrist_pos_w[0]=100
d.update(c)
assert d.stats[0,6]==0 and d.stats[0,0]==0
print('PASS FD alignment, contact association, stable recovery and teleport/reset isolation')
