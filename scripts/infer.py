from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirseg.engine import build_model, load_model_checkpoint
from mirseg.inference import run_manifest_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIRSeg sliding-window inference from a manifest CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.setdefault("data", {})["manifest"] = args.manifest
    device = torch.device(cfg.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg)
    load_model_checkpoint(model, args.checkpoint, strict=False)
    run_manifest_inference(
        model=model,
        manifest_csv=args.manifest,
        split=args.split,
        out_dir=args.out_dir,
        roi_size=cfg.get("runtime", {}).get("sliding_window_roi", [96, 96, 96]),
        overlap=float(cfg.get("runtime", {}).get("sliding_window_overlap", 0.5)),
        device=device,
    )


if __name__ == "__main__":
    main()
