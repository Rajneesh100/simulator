"""Behavior cloning from recorded trajectories (idea.md M5: "BC-init -> PPO fine-tune").

Loads `.npz` files produced by `robosim teleop --record` (arrays `obs`, `actions`),
supervises a small MLP with the same shape as P1's `net_arch=[64,64]` to reproduce the
recorded actions, and saves a state dict whose keys line up with SB3's ActorCriticPolicy
(`mlp_extractor.policy_net.*`, `action_net.*`) so `robosim train --bc-init <path>` can
warm-start the actor pathway with `load_state_dict(..., strict=False)` (value_net/log_std
stay randomly initialized — that's expected, PPO fine-tunes those from scratch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BCPolicyNet(nn.Module):
    """Mirrors SB3 ActorCriticPolicy's default MLP exactly: net_arch=[64, 64] with Tanh
    activations (SB3's default `activation_fn` — NOT ReLU). Matching shapes alone isn't
    enough for the loaded weights to compute the same function; the nonlinearity must
    match too."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.action_net = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.policy_net(obs))

    def sb3_state_dict(self) -> dict:
        sd = {}
        for k, v in self.policy_net.state_dict().items():
            sd[f"mlp_extractor.policy_net.{k}"] = v
        for k, v in self.action_net.state_dict().items():
            sd[f"action_net.{k}"] = v
        return sd


def load_recordings(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    obs_chunks, act_chunks = [], []
    for p in paths:
        data = np.load(p)
        obs_chunks.append(data["obs"])
        act_chunks.append(data["actions"])
    return np.concatenate(obs_chunks), np.concatenate(act_chunks)


def train_bc(
    record_paths: list[str],
    out_path: str,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> Path:
    torch.manual_seed(seed)
    obs, actions = load_recordings(record_paths)
    obs_dim, action_dim = obs.shape[1], actions.shape[1]

    net = BCPolicyNet(obs_dim, action_dim)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    obs_t = torch.from_numpy(obs.astype(np.float32))
    act_t = torch.from_numpy(actions.astype(np.float32))
    n = obs_t.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = net(obs_t[idx])
            loss = nn.functional.mse_loss(pred, act_t[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(idx)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"[bc] epoch {epoch}: mse={total_loss / n:.5f}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.sb3_state_dict(), out)
    print(f"[bc] saved BC weights to {out} (obs_dim={obs_dim}, action_dim={action_dim})")
    return out


def generate_scripted_demo(
    out_path: str,
    scene: str = "flat",
    robot: str = "diffdrive",
    task: str = "goal_reach",
    episodes: int = 20,
    max_steps: int = 300,
    seed: int = 0,
) -> Path:
    """Scripted proportional 'expert' that drives straight at the goal — a stand-in for
    manual `teleop --record` sessions so the BC pipeline can be exercised and tested
    without a human driving interactively (see CLAUDE.md decision log)."""
    from robosim.envs.base_env import RoboSimEnv

    env = RoboSimEnv(scene=scene, robot=robot, task=task, max_episode_steps=max_steps)
    all_obs, all_actions = [], []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            # obs layout: [lidar(n) | dist, sin_b, cos_b | wheel_l, wheel_r, pitch, roll, yaw_rate | prev_action]
            sin_b, cos_b = obs[env.lidar.n_beams + 1], obs[env.lidar.n_beams + 2]
            bearing = np.arctan2(sin_b, cos_b)
            forward = float(np.clip(np.cos(bearing), 0.0, 1.0))
            turn = float(np.clip(2.0 * bearing / np.pi, -1.0, 1.0))
            action = np.array([forward, turn], dtype=np.float32)
            all_obs.append(obs.copy())
            all_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), obs=np.stack(all_obs), actions=np.stack(all_actions))
    print(f"[bc] wrote {len(all_obs)} scripted-demo steps to {out.with_suffix('.npz')}")
    return out.with_suffix(".npz")


def expand_policy_state_dict(sd: dict, extra_obs: int, extra_act: int,
                             fire_bias: float = -1.0) -> dict:
    """Grow an SB3 MlpPolicy state dict to a wider obs/action space without changing
    what it computes on the old inputs: new input columns get zero weight (the combat
    block is ignored until learned), new action rows get zero weight, bias `fire_bias`
    for the fire channel (stays holstered) and 0 for the aim channels, and log_std 0
    (unit exploration) for the new rows. Used to warm-start rally6 from rally5.

    The new obs columns must be APPENDED to the old layout (they are: prev_action[2:]
    and the combat block both come after every column the old policy read)."""
    import torch

    out = dict(sd)
    for key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        if key in out and extra_obs > 0:
            w = out[key]
            out[key] = torch.cat([w, torch.zeros(w.shape[0], extra_obs, dtype=w.dtype)], dim=1)
    if extra_act > 0:
        w = out["action_net.weight"]
        out["action_net.weight"] = torch.cat(
            [w, torch.zeros(extra_act, w.shape[1], dtype=w.dtype)], dim=0)
        b = out["action_net.bias"]
        new_b = torch.zeros(extra_act, dtype=b.dtype)
        new_b[0] = fire_bias
        out["action_net.bias"] = torch.cat([b, new_b], dim=0)
        if "log_std" in out:
            ls = out["log_std"]
            out["log_std"] = torch.cat([ls, torch.zeros(extra_act, dtype=ls.dtype)], dim=0)
    return out
