from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import datasets
from huggingface_hub import HfApi


@dataclass(frozen = True)
class ExportConfig:
    repo_id: str
    env_id: str
    policy: str
    horizon: int
    master_seed: int
    obs_dim: int = 5
    act_dim: int = 2
    git_commit: str = ""
    private: bool = False
    schema_version: str = "v1"


def build_steps_features(obs_dim: int, act_dim: int) -> datasets.Features:
    return datasets.Features(
        {
            "episode_id": datasets.Value("int64"),
            "t": datasets.Value("int32"),
            "obs_state": datasets.Sequence(datasets.Value("float32"), length = obs_dim),
            "action": datasets.Sequence(datasets.Value("float32"), length = act_dim),
            "reward": datasets.Value("float32"),
            "terminated": datasets.Value("bool"),
            "truncated": datasets.Value("bool"),
            "done": datasets.Value("bool"),
            "seed": datasets.Value("int64"),
            "is_success": datasets.Value("bool"),
            "coverage": datasets.Value("float32"),
            "n_contacts": datasets.Value("int32"),
        }
    )


def build_episodes_features() -> datasets.Features:
    return datasets.Features(
        {
            "episode_id": datasets.Value("int64"),
            "seed": datasets.Value("int64"),
            "length": datasets.Value("int32"),
            "return": datasets.Value("float32"),
            "terminated": datasets.Value("bool"),
            "truncated": datasets.Value("bool"),
            "env_id": datasets.Value("string"),
            "policy": datasets.Value("string"),
            "horizon": datasets.Value("int32"),
        }
    )


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent = 2, sort_keys = True)


def export_save_and_push(
    *,
    steps_dict: Dict[str, Any],
    episodes_rows: List[Dict[str, Any]],
    cfg: ExportConfig,
    out_root: str = "artifacts/datasets",
) -> Dict[str, str]:
    """
    Crea DatasetDict con {steps, episodes}, guarda local, y sube al Hub:
      - dataset (parquet)
      - schema.json
      - runs/<run_id>/run_metadata.json
    """
    run_id = str(uuid.uuid4())
    created_at_utc = _dt.datetime.utcnow().replace(microsecond = 0).isoformat() + "Z"

    # Datasets
    steps_ds = datasets.Dataset.from_dict(
        steps_dict, features = build_steps_features(cfg.obs_dim, cfg.act_dim)
    )
    episodes_ds = datasets.Dataset.from_list(
        episodes_rows, features = build_episodes_features()
    )
    dd = datasets.DatasetDict({"steps": steps_ds, "episodes": episodes_ds})

    # Metadata
    schema_json = {
        "schema_version": cfg.schema_version,
        "tables": ["steps", "episodes"],
        "obs_dim": cfg.obs_dim,
        "action_dim": cfg.act_dim,
        "done_definition": "done = terminated OR truncated",
        "columns": {
            "steps": list(steps_ds.features.keys()),
            "episodes": list(episodes_ds.features.keys()),
        },
    }

    run_metadata = {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "repo_id": cfg.repo_id,
        "git_commit": cfg.git_commit,
        "env_id": cfg.env_id,
        "policy": cfg.policy,
        "horizon": int(cfg.horizon),
        "master_seed": int(cfg.master_seed),
        "num_steps": int(len(steps_ds)),
        "num_episodes": int(len(episodes_ds)),
    }

    # Save local
    out_dir = os.path.join(out_root, f"pusht_{cfg.schema_version}_{run_id}")
    os.makedirs(out_dir, exist_ok = True)
    dd.save_to_disk(os.path.join(out_dir, "dataset"))
    write_json(os.path.join(out_dir, "schema.json"), schema_json)
    write_json(os.path.join(out_dir, "run_metadata.json"), run_metadata)

    # Push dataset
    dd.push_to_hub(
        cfg.repo_id,
        private = cfg.private,
        commit_message=(
            f"{cfg.schema_version}: {cfg.policy} rollouts "
            f"run_id={run_id} episodes={len(episodes_ds)} horizon={cfg.horizon} "
            f"master_seed={cfg.master_seed}"
        ),
    )

    # Upload metadata files (extra)
    api = HfApi()
    api.upload_file(
        path_or_fileobj = os.path.join(out_dir, "schema.json"),
        path_in_repo="schema.json",
        repo_id = cfg.repo_id,
        repo_type="dataset",
        commit_message = f"Add schema.json ({cfg.schema_version})",
    )
    api.upload_file(
        path_or_fileobj = os.path.join(out_dir, "run_metadata.json"),
        path_in_repo = f"runs/{run_id}/run_metadata.json",
        repo_id = cfg.repo_id,
        repo_type="dataset",
        commit_message = f"Add run metadata run_id={run_id}",
    )

    return {"run_id": run_id, "out_dir": out_dir}
