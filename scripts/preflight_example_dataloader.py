"""
Minimal example for the optional preflight-ml GitHub Action.

This repository itself does not train PyTorch models; this file exists only to
demonstrate how the preflight action is wired into CI for ML training projects.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(128, 3, 224, 224)
y = torch.randint(0, 10, (128,))

dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)

