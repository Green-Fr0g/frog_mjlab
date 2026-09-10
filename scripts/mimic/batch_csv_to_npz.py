"""Batch convert motion CSV files to NPZ using ``csv_to_npz.py``.

This wrapper mirrors ``frog_lab/scripts/mimic/batch_csv_to_npz_frog.py``: it
discovers conversion jobs from yaml configs and invokes the single-file
converter once per CSV. Two input modes are supported:

1. ``--config``: one yaml. Prefer ``csv_paths`` for multiple files; ``csv_path``
   also accepts a string or a list for compatibility.
2. ``--config_dir``: a directory of yaml files; each file may itself contain
   one or many CSV paths.

Example:
    python scripts/mimic/batch_csv_to_npz.py \\
        --config motion_data/config/g1.yaml \\
        --output_dir /tmp/motions

    python scripts/mimic/batch_csv_to_npz.py \\
        --config_dir motion_data/config \\
        --output_dir /tmp/motions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ConversionJob:
  config_path: Path
  csv_path: Path
  output_path: Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
  parser = argparse.ArgumentParser(
    description="Batch convert CSV motion files to NPZ via scripts/mimic/csv_to_npz.py."
  )
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument(
    "--config",
    type=Path,
    help="Single motion config yaml. csv_path / csv_paths may be a string or a list.",
  )
  source.add_argument(
    "--config_dir",
    type=Path,
    help="Directory containing motion config yaml files.",
  )
  parser.add_argument(
    "--output_dir",
    type=Path,
    required=True,
    help="Directory where NPZ files will be written.",
  )
  parser.add_argument("--output_fps", type=float, default=50.0, help="Output NPZ frame rate.")
  parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional 1-based inclusive frame range passed to the single-file converter.",
  )
  parser.add_argument(
    "--pattern",
    type=str,
    default="*.yaml",
    help="Glob pattern used to discover yaml files under --config_dir.",
  )
  parser.add_argument(
    "--recursive",
    action="store_true",
    help="Search --config_dir recursively.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate NPZ files even when the output already exists.",
  )
  parser.add_argument(
    "--dry_run",
    action="store_true",
    help="Print planned commands without running conversion.",
  )
  parser.add_argument(
    "--python",
    type=str,
    default=sys.executable,
    help="Python executable used to run csv_to_npz.py.",
  )
  return parser.parse_known_args()


def _load_yaml(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise ValueError(f"Invalid yaml structure in: {path}")
  return data


def _resolve_path(base_dir: Path, maybe_path: str) -> Path:
  path = Path(maybe_path)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def _normalize_csv_entries(raw: Any, field_name: str) -> list[str]:
  """Accept a string or a list of strings for csv_path / csv_paths."""
  if isinstance(raw, str):
    if not raw.strip():
      raise ValueError(f"'{field_name}' is empty")
    return [raw]
  if isinstance(raw, list):
    if not raw:
      raise ValueError(f"'{field_name}' list is empty")
    entries: list[str] = []
    for item in raw:
      if not isinstance(item, str):
        raise TypeError(f"'{field_name}' entries must be strings, got: {type(item).__name__}")
      if not item.strip():
        raise ValueError(f"'{field_name}' contains an empty path entry")
      entries.append(item)
    return entries
  raise TypeError(f"'{field_name}' must be a string or list of strings, got: {type(raw).__name__}")


def _extract_csv_entries(motion_cfg: dict[str, Any]) -> list[str]:
  has_csv_path = "csv_path" in motion_cfg
  has_csv_paths = "csv_paths" in motion_cfg
  if not has_csv_path and not has_csv_paths:
    raise KeyError("motion_data must contain 'csv_path' or 'csv_paths'")

  entries: list[str] = []
  if has_csv_paths:
    entries.extend(_normalize_csv_entries(motion_cfg["csv_paths"], "csv_paths"))
  if has_csv_path:
    # Compatible with configs that put multiple paths under csv_path by mistake.
    entries.extend(_normalize_csv_entries(motion_cfg["csv_path"], "csv_path"))

  # Keep order while dropping exact duplicates.
  deduped: list[str] = []
  seen: set[str] = set()
  for entry in entries:
    if entry in seen:
      continue
    seen.add(entry)
    deduped.append(entry)
  return deduped


def discover_config_files(config_dir: Path, pattern: str, recursive: bool) -> list[Path]:
  if not config_dir.is_dir():
    raise FileNotFoundError(f"Config directory does not exist: {config_dir}")
  iterator = config_dir.rglob(pattern) if recursive else config_dir.glob(pattern)
  files = sorted(path for path in iterator if path.is_file())
  # Also accept .yml when the default *.yaml pattern is used.
  if pattern == "*.yaml":
    yml_iterator = config_dir.rglob("*.yml") if recursive else config_dir.glob("*.yml")
    files.extend(path for path in yml_iterator if path.is_file())
    files = sorted(set(files))
  return files


def _unique_output_path(output_dir: Path, csv_path: Path, used_names: set[str]) -> Path:
  stem = csv_path.stem
  candidate = f"{stem}.npz"
  if candidate not in used_names:
    used_names.add(candidate)
    return output_dir / candidate

  parent_stem = csv_path.parent.name or "motion"
  candidate = f"{parent_stem}__{stem}.npz"
  if candidate not in used_names:
    used_names.add(candidate)
    return output_dir / candidate

  index = 2
  while True:
    candidate = f"{parent_stem}__{stem}_{index}.npz"
    if candidate not in used_names:
      used_names.add(candidate)
      return output_dir / candidate
    index += 1


def build_jobs_from_config(
  config_path: Path, output_dir: Path, used_names: set[str]
) -> list[ConversionJob]:
  config = _load_yaml(config_path)
  if "motion_data" not in config:
    raise KeyError(f"Missing 'motion_data' in config: {config_path}")
  motion_cfg = config["motion_data"]
  if not isinstance(motion_cfg, dict):
    raise TypeError(f"'motion_data' must be a mapping in: {config_path}")

  jobs: list[ConversionJob] = []
  for entry in _extract_csv_entries(motion_cfg):
    csv_path = _resolve_path(config_path.parent, entry)
    output_path = _unique_output_path(output_dir, csv_path, used_names)
    jobs.append(ConversionJob(config_path=config_path, csv_path=csv_path, output_path=output_path))
  return jobs


def collect_jobs(args: argparse.Namespace) -> list[ConversionJob]:
  output_dir = args.output_dir.expanduser().resolve()
  used_names: set[str] = set()
  jobs: list[ConversionJob] = []

  if args.config is not None:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
      raise FileNotFoundError(f"Config file does not exist: {config_path}")
    jobs.extend(build_jobs_from_config(config_path, output_dir, used_names))
  else:
    config_dir = args.config_dir.expanduser().resolve()
    config_files = discover_config_files(config_dir, args.pattern, args.recursive)
    if not config_files:
      raise FileNotFoundError(f"No yaml files matched '{args.pattern}' in {config_dir}")
    for config_path in config_files:
      jobs.extend(build_jobs_from_config(config_path, output_dir, used_names))

  return jobs


def build_command(
  python_executable: str,
  converter_path: Path,
  job: ConversionJob,
  output_fps: float,
  frame_range: list[int] | None,
  passthrough_args: list[str],
) -> list[str]:
  command = [
    python_executable,
    str(converter_path),
    "--config",
    str(job.config_path),
    "--csv_path",
    str(job.csv_path),
    "--output_name",
    str(job.output_path),
    "--output_fps",
    str(output_fps),
  ]
  if frame_range is not None:
    command.extend(["--frame_range", str(frame_range[0]), str(frame_range[1])])
  command.extend(passthrough_args)
  return command


def main() -> None:
  args, passthrough_args = parse_args()

  script_dir = Path(__file__).resolve().parent
  converter_path = script_dir / "csv_to_npz.py"
  if not converter_path.is_file():
    raise FileNotFoundError(f"Single-file converter not found: {converter_path}")

  jobs = collect_jobs(args)
  if not jobs:
    raise RuntimeError("No conversion jobs found.")

  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] Found {len(jobs)} conversion job(s).")
  print(f"[INFO] Output directory: {output_dir}")
  if passthrough_args:
    print(f"[INFO] Passing extra args to csv_to_npz.py: {' '.join(passthrough_args)}")

  converted = 0
  skipped = 0
  failed = 0

  for job in jobs:
    if job.output_path.exists() and not args.overwrite:
      print(f"[SKIP] {job.output_path} already exists.")
      skipped += 1
      continue

    command = build_command(
      python_executable=args.python,
      converter_path=converter_path,
      job=job,
      output_fps=args.output_fps,
      frame_range=args.frame_range,
      passthrough_args=passthrough_args,
    )
    print(f"[RUN] {job.csv_path} -> {job.output_path}")
    print(f"      {' '.join(command)}")

    if args.dry_run:
      if not job.csv_path.is_file():
        print(f"[WARN] CSV not found for config {job.config_path}: {job.csv_path}")
      converted += 1
      continue

    if not job.csv_path.is_file():
      print(f"[FAIL] CSV not found for config {job.config_path}: {job.csv_path}")
      failed += 1
      continue

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
      print(f"[FAIL] Conversion failed with exit code {result.returncode}: {job.csv_path}")
      failed += 1
      continue
    converted += 1

  print(f"[DONE] converted={converted}, skipped={skipped}, failed={failed}, total={len(jobs)}")
  if failed > 0:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
