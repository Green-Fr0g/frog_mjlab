import os
import inspect

import torch

try:
  import wandb
except ImportError:  # pragma: no cover - optional logger dependency
  wandb = None

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from frog_rl.runners import OnPolicyRunner


def _drop_none_values(value):
  if isinstance(value, dict):
    return {
      key: _drop_none_values(item)
      for key, item in value.items()
      if item is not None
    }
  if isinstance(value, list):
    return [_drop_none_values(item) for item in value]
  return value


_MLP_MODEL_KEYS = {
  "class_name",
  "hidden_dims",
  "activation",
  "obs_normalization",
  "distribution_cfg",
}


def _normalize_train_cfg(train_cfg: dict) -> dict:
  cfg = _drop_none_values(train_cfg)
  for key in ("actor", "critic", "student", "teacher"):
    model_cfg = cfg.get(key)
    if isinstance(model_cfg, dict):
      cfg[key] = {
        item_key: item_value
        for item_key, item_value in model_cfg.items()
        if item_key in _MLP_MODEL_KEYS
      }
  return cfg


def _onnx_export_kwargs_single_file() -> dict:
  """Build kwargs that request single-file ONNX export across torch versions."""
  try:
    params = inspect.signature(torch.onnx.export).parameters
  except (TypeError, ValueError):
    return {}

  if "external_data" in params:
    return {"external_data": False}
  if "use_external_data_format" in params:
    return {"use_external_data_format": False}
  return {}


def _inline_external_onnx_data(onnx_path: str) -> None:
  """Merge external tensor data back into a single ONNX file if needed."""
  data_path = f"{onnx_path}.data"
  if not os.path.exists(data_path):
    return

  try:
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(model, onnx_path, save_as_external_data=False)
    if os.path.exists(data_path):
      os.remove(data_path)
    print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
  except Exception as exc:
    print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")


class AMPOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    super().__init__(env, _normalize_train_cfg(train_cfg), log_dir, device)

  def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
    """Export the actor network to ONNX using the local ActorCritic model.
    
    The exported model includes the obs normalizer (if empirical_normalization
    is enabled) so that the ONNX model expects raw observations directly.
    """
    policy = self.alg.get_policy()
    wrapper = policy.as_onnx(verbose=False)
    wrapper.to("cpu")
    wrapper.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      wrapper,
      wrapper.get_dummy_inputs(),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
      **_onnx_export_kwargs_single_file(),
    )
    _inline_external_onnx_data(os.path.join(path, filename))
    # move policy back to training device
    policy.to(self.device)

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self._export_policy_to_onnx(policy_path, filename)
    logger_type = getattr(self.logger, "logger_type", None)
    run_name: str = "local"
    if logger_type in ("wandb", "WandbLogWriter") and wandb is not None and wandb.run:
      run_name = wandb.run.name
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    _inline_external_onnx_data(onnx_path)
    if logger_type in ("wandb", "WandbLogWriter") and wandb is not None:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
