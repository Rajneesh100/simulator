"""Export a trained (non-recurrent) policy to ONNX — the deployment path to the real
robot (onnxruntime on a Pi). Includes an int8-quantized inference sanity check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _DeterministicPolicyWrapper(nn.Module):
    """Wraps an SB3 ActorCriticPolicy to output only the deterministic (mean) action."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        mean_actions = self.policy.action_net(latent_pi)
        low = torch.as_tensor(self.policy.action_space.low)
        high = torch.as_tensor(self.policy.action_space.high)
        return torch.clamp(mean_actions, low, high)


def export_onnx(checkpoint: str, out_path: str, obs_dim: int) -> Path:
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint, device="cpu")
    wrapper = _DeterministicPolicyWrapper(model.policy).eval()

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_obs,
        str(out),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    _sanity_check(model, wrapper, out, obs_dim)
    _int8_quantize_and_check(out, obs_dim)
    return out


def _sanity_check(model, wrapper, onnx_path: Path, obs_dim: int, n_samples: int = 8) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, size=(n_samples, obs_dim)).astype(np.float32)

    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(obs)).numpy()
    onnx_out = sess.run(["action"], {"obs": obs})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX export mismatch: max diff {max_diff} exceeds tolerance")
    print(f"[export] ONNX sanity check OK (max diff vs torch: {max_diff:.2e})")


def _int8_quantize_and_check(onnx_path: Path, obs_dim: int) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quant_path = onnx_path.with_name(onnx_path.stem + ".int8.onnx")
    quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(quant_path))
    rng = np.random.default_rng(1)
    obs = rng.uniform(-1, 1, size=(4, obs_dim)).astype(np.float32)
    out = sess.run(["action"], {"obs": obs})[0]
    assert np.all(np.isfinite(out)), "int8-quantized model produced non-finite output"
    print(f"[export] int8-quantized model OK: {quant_path}")
    return quant_path
