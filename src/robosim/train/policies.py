"""Policy architectures (idea.md table), all small enough to run on a Pi 5 later.

P1 MLP is SB3's default MLP policy with an explicit small net_arch. P2/P3/P4 are wired
for the experiment suite (M5): P2 is a 1D-CNN feature extractor over the lidar slice of
the obs vector concatenated with proprio/goal; P3 uses sb3-contrib's RecurrentPPO
(LSTM policy) with the plain flat obs; P4 is a frame-stack wrapper applied to the env
rather than a feature extractor, so it plugs into vanilla PPO's MlpPolicy.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Obs layout offsets — kept in sync with envs/obs.py's canonical layout by construction:
# [lidar (n_beams) | goal (3) | proprio (5) | prev_action (action_dim)]
GOAL_PROPRIO_DIM = 3 + 5


class LidarCNNExtractor(BaseFeaturesExtractor):
    """P2: 1D-CNN over lidar beams, concatenated with an MLP over the remaining (non-lidar)
    obs terms (goal/proprio/prev-action)."""

    def __init__(self, observation_space: gym.Space, n_lidar_beams: int, cnn_channels: int = 8):
        tail_dim = observation_space.shape[0] - n_lidar_beams
        features_dim = 32 + 16
        super().__init__(observation_space, features_dim=features_dim)
        self.n_lidar_beams = n_lidar_beams
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(cnn_channels * 2, 32),
            nn.ReLU(),
        )
        self.tail_mlp = nn.Sequential(nn.Linear(tail_dim, 16), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lidar = observations[:, : self.n_lidar_beams].unsqueeze(1)
        tail = observations[:, self.n_lidar_beams :]
        return torch.cat([self.cnn(lidar), self.tail_mlp(tail)], dim=1)


def policy_kwargs_for(policy: str, n_lidar_beams: int = 16) -> dict:
    policy = policy.lower()
    if policy == "p1":
        return {"net_arch": [64, 64]}
    if policy == "p2":
        return {
            "features_extractor_class": LidarCNNExtractor,
            "features_extractor_kwargs": {"n_lidar_beams": n_lidar_beams},
            "net_arch": [64],
        }
    if policy == "p3":
        # Used only with sb3_contrib.RecurrentPPO's "MlpLstmPolicy"; net_arch stays small.
        return {"net_arch": [64], "lstm_hidden_size": 64}
    if policy == "p4":
        # Frame-stacking is an env wrapper (see train/train.py), not a feature extractor.
        return {"net_arch": [64, 64]}
    raise ValueError(f"unknown policy '{policy}'; choose from p1, p2, p3, p4")


def sb3_policy_name_for(policy: str) -> str:
    return "MlpLstmPolicy" if policy.lower() == "p3" else "MlpPolicy"
