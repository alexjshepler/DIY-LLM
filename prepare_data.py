#!/usr/bin/env python3
import os, argparse, glob
import numpy as np
from tqdm import tqdm

from src.tokenizer import BPETokenizer
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    C = load_config(args.config)

    raw_dir = C["io"]["raw_dir"]
    proc_dir = C["io"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    # --- 1) Read all .txt into one string (with a progress bar over files) ---
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    texts = []
    total_bytes = 0
    for p in paths:
        total_bytes += os.path.getsize(p)

    print(f"Found {len(paths)} text file(s) (~{total_bytes/1e6:.2f} MB). Reading...")
    for p in tqdm(paths, desc="Reading files", unit="file"):
        with open(p, "r", encoding="utf-8") as f:
            texts.append(f.read())

    full_text = "\n".join(texts)
    if len(full_text.strip()) == 0:
        raise ValueError("Input text appears empty after reading files.")

    # --- 2) Train tokenizer (BPE) ---
    print("Training BPE tokenizer...")
    tok = BPETokenizer(
        vocab_size=C["tokenizer"]["vocab_size"],
        min_freq=C["tokenizer"].get("min_freq", 1),
    )
    tok.train(full_text)
    tok.save(proc_dir)

    # --- 3) Encode corpus to token IDs (multiprocessing) ---
    print("Encoding corpus to token IDs (multiprocessing)...")

    words = full_text.split()
    del full_text  # free RAM early

    from multiprocessing import Pool, cpu_count
    import math

    CHUNK = 100_000  # adjust: bigger chunk = fewer merges = faster
    pieces = []
    for i in range(0, len(words), CHUNK):
        pieces.append(" ".join(words[i:i+CHUNK]))
    del words

    def _encode_piece(piece):
        from src.tokenizer import BPETokenizer
        tok = BPETokenizer.load(proc_dir)
        return tok.encode(piece)

    with Pool(cpu_count()) as pool:
        ids_lists = list(tqdm(
            pool.imap_unordered(_encode_piece, pieces, chunksize=1),
            total=len(pieces),
            desc="Encoding (CPU parallel)"
        ))

    # flatten lists to one int array
    ids = np.fromiter((x for seg in ids_lists for x in seg), dtype=np.int32)
    del ids_lists


    # --- 4) Split train/val and save ---
    split = float(C["data"]["train_split"])
    n_train = int(len(ids) * split)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError(
            f"Train/val too small after split ({len(train_ids)} / {len(val_ids)}). "
            f"Add more data or reduce block_size."
        )

    np.save(os.path.join(proc_dir, "train.npy"), train_ids)
    np.save(os.path.join(proc_dir, "val.npy"), val_ids)

    print(
        f"Saved tokenizer and token arrays to {proc_dir}\n"
        f"- train tokens: {len(train_ids):,}\n"
        f"-   val tokens: {len(val_ids):,}"
    )


if __name__ == "__main__":
    main()
