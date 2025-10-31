#!/usr/bin/env python3
"""Fast dataset preparation using the shared config presets."""

import argparse
import glob
import os
from typing import Dict, Any

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.utils import resolve_config


def prepare_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare tokenizer files and token arrays based on ``config``.

    Returns a dictionary with a few summary statistics about the artifacts.
    """

    raw_dir = config["io"]["raw_dir"]
    processed_dir = config["io"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    vocab_size = int(config["tokenizer"]["vocab_size"])
    min_freq = int(config["tokenizer"].get("min_freq", 1))

    print(f"Training fast BPE on {len(paths)} file(s)...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<unk>", "</w>", "<pad>", "<bos>", "<eos>", " "],
        show_progress=True,
    )
    tokenizer.train(files=paths, trainer=trainer)

    hf_path = os.path.join(processed_dir, "hf_tokenizer.json")
    tokenizer.save(hf_path)
    print(f"Saved HF tokenizer to {hf_path}")

    print("Encoding with fast tokenizer...")
    texts = []
    for file_path in tqdm(paths, desc="Reading files", unit="file"):
        with open(file_path, "r", encoding="utf-8") as handle:
            texts.append(handle.read())

    encodings = tokenizer.encode_batch(texts)
    ids = np.fromiter((token_id for enc in encodings for token_id in enc.ids), dtype=np.int32)

    if len(ids) < 2:
        raise ValueError("Tokenized corpus is too small (need at least 2 tokens).")

    split = float(config["data"]["train_split"])
    n_train = int(len(ids) * split)
    n_train = min(max(n_train, 1), len(ids) - 1)
    train_ids, val_ids = ids[:n_train], ids[n_train:]

    np.save(os.path.join(processed_dir, "train.npy"), train_ids)
    np.save(os.path.join(processed_dir, "val.npy"), val_ids)

    stats = {
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "tokenizer_path": hf_path,
        "processed_dir": processed_dir,
    }

    print(
        "Saved token arrays to {dir}\n"
        "- train tokens: {train:,}\n"
        "-   val tokens: {val:,}\n"
        "- vocab size: {vocab:,}".format(
            dir=processed_dir,
            train=len(train_ids),
            val=len(val_ids),
            vocab=tokenizer.get_vocab_size(),
        )
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenizer + dataset from config presets.")
    parser.add_argument("--config", help="Path or name of the config preset to use.")
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config presets (default: %(default)s).",
    )
    args = parser.parse_args()

    config_path, config = resolve_config(args.config, args.config_dir)
    print(f"Using config: {config_path}")
    prepare_dataset(config)


if __name__ == "__main__":
    main()
