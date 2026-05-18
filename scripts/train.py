from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirseg.engine import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the public MIRSeg or shared-baseline model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/mirseg.yaml")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
