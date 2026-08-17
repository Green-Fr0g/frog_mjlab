import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from frog_mjlab.tasks.amp.utils.runner import FrogMjlabOnPolicyRunner


class AMPOnPolicyRunner(FrogMjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    logger_type = self.logger.logger_type
    run_name: str = (
      wandb.run.name if logger_type in ("wandb", "WandbLogWriter") and wandb.run else "local"
    )
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if logger_type in ("wandb", "WandbLogWriter"):
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
