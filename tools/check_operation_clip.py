"""CPU regression for procedural XYZ/quaternion manipulation clips."""
import runpy
from pathlib import Path
from types import SimpleNamespace
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude, quat_mul, quat_apply

module = runpy.run_path(str(Path(__file__).resolve().parents[1]/"src/tasks/wrist_recovery/mdp/operation_clip.py"))
torch.manual_seed(2301)
n = 32
cfg = SimpleNamespace(low_reach_extension_range=(.02,.18), height_lateral_offset_range=(-.10,.10), extension_range=(.02,.30), lateral_offset_range=(-.10,.10))
offset = torch.zeros(n,2,3)
offset[...,0] = .10
root = torch.zeros(n,4); root[:,0] = 1
goal = root[:,None,:].expand(-1,2,-1)
delta, quats, duration = module['sample_clip'](offset,root,goal,torch.ones(n,dtype=torch.bool),torch.ones(n),cfg)
delta, quats, duration = delta.double(), quats.double(), duration.double()
assert torch.all(delta[...,2]>=0) and torch.all(delta[...,2]<=.12)
assert torch.all(torch.linalg.vector_norm(delta[...,:2],dim=-1)<=.060001)
assert torch.all((offset[:,None,:,0]+delta[...,0]>=.01999)&(offset[:,None,:,0]+delta[...,0]<=.18001))
assert torch.allclose(delta[:,0],torch.zeros_like(delta[:,0]))
assert (duration[:,:,0]-duration[:,:,1]).abs().max()>.01
positions, orientations = [], []
dt = .005
for t in torch.arange(0,float(duration.sum(1).max())+.1,dt):
  pos, quat, stage = module['evaluate_clip'](delta,quats,duration,torch.full((n,),t))
  assert torch.isfinite(pos).all() and torch.isfinite(quat).all()
  assert torch.allclose(quat.square().sum(-1),torch.ones(n,2,dtype=quat.dtype),atol=1e-5)
  positions.append(pos); orientations.append(quat)
p, q = torch.stack(positions), torch.stack(orientations)
v = (p[1:]-p[:-1])/dt
a = (v[1:]-v[:-1])/dt
conjugate = q[:-1].clone(); conjugate[...,1:] *= -1
relative = quat_mul(q[1:],conjugate)
vector = relative[...,1:]
norm = torch.linalg.vector_norm(vector,dim=-1)
omega = vector/norm.clamp_min(1e-12)[...,None]*(2*torch.atan2(norm,relative[...,0]))[...,None]/dt
assert torch.linalg.vector_norm(omega,dim=-1).max()<.601
assert torch.linalg.vector_norm((omega[1:]-omega[:-1])/dt,dim=-1).max()<1.51
assert torch.linalg.vector_norm(v,dim=-1).max()<.121
assert torch.linalg.vector_norm(a,dim=-1).max()<.305
z=p[...,2]; shoulder=z[...,0]*z[...,1]/(z.sum(-1)+.02)
assert torch.all(shoulder<=z.min(-1).values+1e-6)
assert torch.allclose(p[-1],delta[:,-1],atol=1e-6)
assert quat_error_magnitude(q[-1],quats[:,-1]).max()<.001
assert quat_error_magnitude(q,goal.double()[None].expand_as(q)).max()>.1
# Independent reference rotation composes the whole clip, not measured body pose.
rotation = torch.tensor([.9238795325,0.,0.,.3826834324],dtype=torch.float64).expand_as(quats)
rotated = module['evaluate_clip'](quat_apply(rotation,delta),quat_mul(rotation,quats),duration,torch.full((n,),4.))
original = module['evaluate_clip'](delta,quats,duration,torch.full((n,),4.))
assert torch.allclose(rotated[0],quat_apply(rotation[:,0],original[0]),atol=1e-8)
assert quat_error_magnitude(rotated[1],quat_mul(rotation[:,0],original[1])).max()<1e-5
print('PASS XYZ bounds, asynchronous segments, quaternion/end continuity, shoulder feasibility and translation derivatives',float(torch.linalg.vector_norm(v,dim=-1).max()),float(torch.linalg.vector_norm(a,dim=-1).max()))

# Successor endpoints and non-completed hand must not jump or be overwritten.
done=torch.zeros(n,2,dtype=torch.bool); done[:,0]=True
old_delta,old_quats,old_duration=delta.clone(),quats.clone(),duration.clone()
module['continue_clip'](delta,quats,duration,done,offset.double(),root.double(),goal.double(),torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.float64),cfg)
assert torch.allclose(delta[:,0,0],old_delta[:,-1,0])
assert torch.allclose(quats[:,0,0],old_quats[:,-1,0])
assert torch.equal(delta[:,:,1],old_delta[:,:,1]) and torch.equal(quats[:,:,1],old_quats[:,:,1])
assert torch.equal(duration[:,:,1],old_duration[:,:,1])
start=module['evaluate_clip'](delta,quats,duration,torch.zeros(n,2))
assert torch.allclose(start[0][:,0],old_delta[:,-1,0])
after=module['evaluate_clip'](delta,quats,duration,torch.full((n,2),.0001))
assert torch.linalg.vector_norm((after[0][:,0]-start[0][:,0])/.0001,dim=-1).max()<1e-5
assert torch.linalg.vector_norm(delta[...,:2],dim=-1).max()<=.060001
assert delta[...,2].min()>=0 and delta[...,2].max()<=.120001
print('PASS successor continuity, zero boundary velocity, independent hand replacement and retained workspace')
