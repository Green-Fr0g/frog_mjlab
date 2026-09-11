"""Export deployment parameters to ``deploy.yaml`` for real-robot deployment.

Mechanism ported from ``unitree_rl_lab`` (via ``frog_lab``): the yaml carries
every environment-side numeric parameter needed to run the bare exported
policy (``policy.onnx``) on hardware — observation scaling/clipping/history,
action scaling/offset/clipping, control rate, joint mapping and actuator
gains — so that preprocessing/postprocessing on the robot matches training
exactly.

Adapted to the mjlab manager-based API:
- policy observation group is ``"actor"`` (mjlab naming);
- action terms are keyed by actuators; ``target_names`` gives the joints each
  action dimension drives;
- actuator gains are read from the entity's actuator configs (PD-style
  ``stiffness``/``damping`` scalars per actuator);
- no SDK joint remapping: ``joint_ids_map`` lists the actual joint names in
  entity order (the i-th action dimension drives the i-th entry).
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from typing import Any

import yaml


def format_value(x):
  """Round floats to 3 significant digits for a compact yaml file."""
  if isinstance(x, float):
    return float(f"{x:.3g}")
  elif isinstance(x, list):
    return [format_value(i) for i in x]
  elif isinstance(x, dict):
    return {k: format_value(v) for k, v in x.items()}
  else:
    return x


def _tensor_to_list(value: Any) -> Any:
  """Convert a torch tensor to a nested list; pass anything else through."""
  if hasattr(value, "detach"):
    return value.detach().cpu().numpy().tolist()
  return value


def _to_yaml(value: Any) -> Any:
  """Convert a cfg value into a yaml-serializable equivalent."""
  if value is None or isinstance(value, (str, bool, int, float)):
    return value
  if isinstance(value, dict):
    return {str(k): _to_yaml(v) for k, v in value.items()}
  if isinstance(value, (list, tuple, set)):
    return [_to_yaml(v) for v in value]
  if isinstance(value, slice):
    return f"slice({value.start}, {value.stop}, {value.step})"
  if is_dataclass(value) and not isinstance(value, type):
    return _to_yaml({f.name: getattr(value, f.name) for f in fields(value)})
  converted = _tensor_to_list(value)
  if converted is not value:
    return _to_yaml(converted)
  return repr(value)  # last resort: keep a readable string.


def _expand_per_dim(value: Any, action_dim: int) -> list[float]:
  """Expand a scalar/per-dim scale or offset to a per-dimension list."""
  if value is None:
    return [0.0] * action_dim
  if isinstance(value, (int, float)):
    return [float(value)] * action_dim
  flat = _tensor_to_list(value)
  if isinstance(flat, list) and flat and isinstance(flat[0], list):
    flat = flat[0]  # (num_envs, dim) -> take the first env.
  return [float(v) for v in flat]


def export_deploy_cfg(env, log_dir, obs_group: str = "actor") -> str:
  """Export the environment-side parameters of ``env`` to ``log_dir/params/deploy.yaml``.

  Only the ``obs_group`` observation group (default ``"actor"``, the policy
  group) is exported; critic/privileged groups are not needed on the deploy
  side. Call this once before training starts, on rank 0 only.

  Args:
    env: A (non-wrapped) :class:`mjlab.envs.ManagerBasedRlEnv` instance.
    log_dir: Training log directory; the file is written to ``params/deploy.yaml``.
    obs_group: Name of the policy observation group to export.

  Returns:
    Path of the written ``deploy.yaml``.
  """
  asset = env.scene["robot"]
  joint_names = list(asset.joint_names)
  joint_index = {name: i for i, name in enumerate(joint_names)}

  cfg = {}  # noqa: SIM904

  # --- joint mapping ---
  # No SDK remapping: actual joint names in entity order. The i-th action
  # dimension drives the i-th entry of this list.
  cfg["joint_ids_map"] = joint_names

  # --- control rate ---
  cfg["step_dt"] = float(env.step_dt)

  # --- actuator gains and default joint state (in joint_ids_map order) ---
  stiffness = [0.0] * len(joint_names)
  damping = [0.0] * len(joint_names)
  for actuator in asset.actuators:
    gain_kp = getattr(actuator.cfg, "stiffness", None)
    gain_kd = getattr(actuator.cfg, "damping", None)
    if gain_kp is None and gain_kd is None:
      continue
    if not isinstance(gain_kp, (int, float)) or not isinstance(gain_kd, (int, float)):
      continue  # non-scalar gains (learned/external actuators) are not supported.
    for target_name in actuator.target_names:
      index = joint_index.get(target_name)
      if index is None:
        continue  # non-joint transmission targets are skipped.
      stiffness[index] = float(gain_kp)
      damping[index] = float(gain_kd)
  cfg["stiffness"] = stiffness
  cfg["damping"] = damping
  cfg["default_joint_pos"] = _tensor_to_list(asset.data.default_joint_pos[0])
  cfg["encoder_bias"] = _tensor_to_list(asset.data.encoder_bias[0])

  # --- commands ---
  cfg["commands"] = {}
  for term_name, term_cfg in env.cfg.commands.items():
    ranges = getattr(term_cfg, "ranges", None)
    if ranges is None or not hasattr(ranges, "lin_vel_x"):
      continue  # only velocity commands are meaningful on the deploy side.
    entry = {
      "ranges": {
        "lin_vel_x": list(ranges.lin_vel_x),
        "lin_vel_y": list(ranges.lin_vel_y),
        "ang_vel_z": list(ranges.ang_vel_z),
      }
    }
    if getattr(ranges, "heading", None) is not None:
      entry["ranges"]["heading"] = list(ranges.heading)
    cfg["commands"][term_name] = entry

  # --- actions ---
  cfg["actions"] = {}
  for term_name in env.action_manager.active_terms:
    term = env.action_manager.get_term(term_name)
    action_dim = term.action_dim
    clip = getattr(term, "_clip", None) if term.cfg.clip is not None else None
    transmission = getattr(term.cfg, "transmission_type", None)
    cfg["actions"][term_name] = {
      "transmission": transmission.name if transmission is not None else "JOINT",
      "actuator_names": list(term.cfg.actuator_names),
      "joint_names": list(term.target_names),
      "joint_ids": _tensor_to_list(term.target_ids),
      "use_default_offset": bool(getattr(term.cfg, "use_default_offset", False)),
      "scale": _expand_per_dim(term.scale, action_dim),
      "offset": _expand_per_dim(term.offset, action_dim),
      "clip": _tensor_to_list(clip[0]) if clip is not None else None,
    }

  # --- observations ---
  obs_manager = env.observation_manager
  if obs_group not in obs_manager.active_terms:
    raise KeyError(
      f"Observation group '{obs_group}' not found; "
      f"available groups: {list(obs_manager.active_terms)}"
    )
  cfg["observations"] = {}
  term_names = obs_manager.active_terms[obs_group]
  term_cfgs = obs_manager._group_obs_term_cfgs[obs_group]
  term_dims = obs_manager.group_obs_term_dim[obs_group]
  for term_name, term_cfg, term_dim in zip(term_names, term_cfgs, term_dims):
    obs_dim = term_dim[-1] if isinstance(term_dim, tuple) and len(term_dim) > 0 else None
    entry = {}
    for field_name in (
      "clip",
      "history_length",
      "delay_min_lag",
      "delay_max_lag",
      "delay_per_env",
      "delay_hold_prob",
      "delay_update_period",
      "delay_per_env_phase",
      "flatten_history_dim",
    ):
      entry[field_name] = _to_yaml(getattr(term_cfg, field_name, None))
    scale = _to_yaml(getattr(term_cfg, "scale", None))
    if scale is None:
      scale = [1.0] * obs_dim if obs_dim is not None else 1.0
    elif isinstance(scale, (int, float)):
      scale = [float(scale)] * obs_dim if obs_dim is not None else float(scale)
    entry["scale"] = scale
    if entry["history_length"] == 0:
      entry["history_length"] = 1
    entry["params"] = {str(k): _to_yaml(v) for k, v in term_cfg.params.items()}
    cfg["observations"][term_name] = entry

  # --- save config file ---
  filename = os.path.join(str(log_dir), "params", "deploy.yaml")
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  cfg = format_value(cfg)
  with open(filename, "w") as f:
    yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
  return filename
