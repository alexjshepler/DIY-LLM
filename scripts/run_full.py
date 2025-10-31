#!/usr/bin/env python3
"""End-to-end helper that prepares data then starts training."""

import argparse

from prepare_data import prepare_dataset
from train import run_training
from src.utils import resolve_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data and train in one go using a config preset."
    )
    parser.add_argument("--config", help="Path or name of the config preset to use.")
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config presets (default: %(default)s).",
    )
    args = parser.parse_args()

    config_path, config = resolve_config(args.config, args.config_dir)
    print(f"Using config: {config_path}")

    stats = prepare_dataset(config)
    print(
        "Starting training with prepared data (train tokens: {train:,}, val tokens: {val:,}).".format(
            train=stats["train_tokens"],
            val=stats["val_tokens"],
        )
    )
    run_training(config)


if __name__ == "__main__":
    main()
