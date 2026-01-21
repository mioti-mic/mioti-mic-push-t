from __future__ import annotations

import argparse
import json
from typing import Any, Dict
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _jsonable(x):
    # Evita petar con tensores/ndarrays: solo imprime tipos y formas básicas.
    try:
        shape = getattr(x, "shape", None)
        if shape is not None:
            return {"type": type(x).__name__, "shape": list(shape)}
        return {"type": type(x).__name__}
    except Exception:
        return {"type": type(x).__name__}


def main():
    p = argparse.ArgumentParser(description = "Sanity check: LeRobotDataset can be loaded and indexed.")
    p.add_argument("--repo-id", type = str, default = "lerobot/pusht")
    p.add_argument("--index", type = int, default = 0)
    args = p.parse_args()

    ds = LeRobotDataset(args.repo_id)
    sample = ds[args.index]

    # No asumimos keys exactas: inspección defensiva.
    summary: Dict[str, Any] = {
        "repo_id": args.repo_id,
        "index": args.index,
        "sample_keys": sorted(list(sample.keys())) if hasattr(sample, "keys") else None,
        "fields": {},
    }

    if hasattr(sample, "items"):
        for k, v in sample.items():
            summary["fields"][k] = _jsonable(v)

    print("LEROBOT DATASET: OK")
    print(json.dumps(summary, indent = 2, sort_keys = True))



if __name__ == "__main__":
    raise SystemExit(main())
