from __future__ import annotations

import argparse
import json
from pathlib import Path

from pusht.env_check import run_smoke_test, to_dict


def main() -> int:
    p = argparse.ArgumentParser(description = "PushT smoke test (Gymnasium + gym-pusht).")
    p.add_argument("--env-id", default = "gym_pusht/PushT-v0")
    p.add_argument("--seed", type = int, default = 0)
    p.add_argument("--horizon", type = int, default = 300)
    p.add_argument("--render", action = "store_true", help = "Render one rgb_array frame at the end.")
    p.add_argument("--out", type = str, default = "", help = "Optional path to write JSON result.")
    args = p.parse_args()

    result = run_smoke_test(
        env_id = args.env_id,
        seed = args.seed,
        horizon = args.horizon,
        render = args.render,
    )

    d = to_dict(result)
    print("SMOKE TEST: OK")
    print(json.dumps(d, indent = 4))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        out_path.write_text(json.dumps(d, indent = 4), encoding = "utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
