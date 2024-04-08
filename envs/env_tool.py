import os
import random
import numpy as np
import torch


def check(value, device):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0) if isinstance(value, np.ndarray) else value
    return output

def check_action_env(action):
    """Check if action that input env is a numpy array, if not, convert it to a numpy array."""
    output = []
    for i in range(len(action)):
        output.append(action[i].detach().cpu().numpy())
    return output