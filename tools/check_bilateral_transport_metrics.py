#!/usr/bin/env python3
"""CPU check of the actual intersection telemetry block, without simulation."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from summarize_wrist_training import conditional_metrics

source = Path(__file__).resolve().parents[1] / "src/tasks/wrist_recovery/mdp/commands.py"
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BimanualWristCommand")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_update_metrics")
start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "self.cfg.clutch_enabled")
block = method.body[start].body
first = next(i for i, n in enumerate(block) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "speed")
last = next(i for i, n in enumerate(block) if isinstance(n, ast.For) and ast.unparse(n.target) == "(index, name)")
self = NS(bilateral_ground_active=torch.tensor([True, True, True, False, True, True]),
          phase=torch.tensor([1., 1., .5, 1., 1., 1.]),
          bilateral_transport_statistics=torch.zeros(6, 6),
          metrics={"shoulder_height_error": torch.ones(6)*.02},
          robot=NS(data=NS(root_link_lin_vel_b=torch.tensor([[.06, 0.], [0., 0.]]*3))))
twist = NS(mode=torch.tensor([1, 1, 1, 1, 2, 1]),
           command=torch.tensor([[.1, 0., 0.]]*5+[[0., 0., 0.]]))
scope = dict(torch=torch, self=self, twist=twist, denominator=torch.ones(6)*10,
             pos_error=torch.tensor([[.01, .03]]*6), xy_error=torch.ones(6)*.04)
exec(compile(ast.Module(body=block[first:last], type_ignores=[]), str(source), "exec"), scope)
assert torch.equal(self.bilateral_transport_statistics[:, 0], torch.tensor([1., 1., 0., 0., 0., 0.]))
assert torch.allclose(self.bilateral_transport_statistics[:2, 4], torch.tensor([.06, 0.]))
means = {"Metrics/wrists/height_command_fraction": .5}
means.update({"Metrics/wrists/"+k: float(v.mean()) for k,v in self.metrics.items()})
result = conditional_metrics(means)
assert abs(result["bilateral_transport_moving_wrist_error"]-.03) < 1e-6
assert abs(result["bilateral_transport_moving_projected_speed"]-.03) < 1e-6
assert result["bilateral_transport_moving_projected_speed"] < .4*result["bilateral_transport_moving_command_xy"]
print("PASS joint intersection, reach-phase exclusion, stopped-body projection and normalization")
