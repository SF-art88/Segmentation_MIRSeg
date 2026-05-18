from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirseg.metrics import evaluate_prediction_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted NIfTI masks against a manifest CSV.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="outputs/eval_metrics.json")
    args = parser.parse_args()
    result = evaluate_prediction_directory(args.manifest, args.pred_dir, args.split, output_json=args.out)
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
