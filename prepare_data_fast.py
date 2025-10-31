#!/usr/bin/env python3
import os, argparse, glob
import numpy as np
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    C = load_config(args.config)

    raw_dir = C["io"]["raw_dir"]
    proc_dir = C["io"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    # 1) Train Rust BPE (very fast)
    print(f"Training fast BPE on {len(paths)} file(s)...")
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=int(C["tokenizer"]["vocab_size"]),
        min_frequency=int(C["tokenizer"].get("min_freq", 1)),
        special_tokens=["<unk>", "</w>", "<pad>", "<bos>", "<eos>", " "],
        show_progress=True,
    )
    tok.train(files=paths, trainer=trainer)

    # Save tokenizer
    hf_path = os.path.join(proc_dir, "hf_tokenizer.json")
    tok.save(hf_path)
    print(f"Saved HF tokenizer to {hf_path}")

    # 2) Encode all files quickly (batch encode)
    print("Encoding with fast tokenizer...")
    texts = []
    for p in tqdm(paths, desc="Reading files", unit="file"):
        with open(p, "r", encoding="utf-8") as f:
            texts.append(f.read())

    # batch encode → list of Encoding objects
    encs = tok.encode_batch(texts)
    ids = np.fromiter((tid for e in encs for tid in e.ids), dtype=np.int32)

    # 3) Split & save arrays
    split = float(C["data"]["train_split"])
    n_train = int(len(ids) * split)
    train_ids, val_ids = ids[:n_train], ids[n_train:]

    np.save(os.path.join(proc_dir, "train.npy"), train_ids)
    np.save(os.path.join(proc_dir, "val.npy"), val_ids)

    print(
        f"Saved token arrays to {proc_dir}\n"
        f"- train tokens: {len(train_ids):,}\n"
        f"-   val tokens: {len(val_ids):,}\n"
        f"- vocab size: {tok.get_vocab_size():,}"
    )


if __name__ == "__main__":
    main()
