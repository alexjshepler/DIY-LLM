#!/usr/bin/env python3
"""Training script that supports interactive config selection."""

import argparse
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.model import GPT
from src.data import load_token_arrays, TokenBatcher
from src.utils import resolve_config, device_and_dtype


def _as_float(x, name):
    try:
        return float(x)
    except Exception as exc:  # noqa: F841
        raise ValueError(f"Config value '{name}' must be a float, got: {x!r}")


def _as_int(x, name):
    try:
        return int(x)
    except Exception:
        raise ValueError(f"Config value '{name}' must be an int, got: {x!r}")


def _as_float_tuple(xs, name):
    try:
        return tuple(float(v) for v in xs)
    except Exception:
        raise ValueError(
            f"Config value '{name}' must be a list/tuple of floats, got: {xs!r}"
        )


def run_training(config: Dict[str, Any]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    io_cfg = config["io"]
    tok_cfg = config["tokenizer"]

    lr = _as_float(train_cfg["lr"], "train.lr")
    weight_decay = _as_float(train_cfg.get("weight_decay", 0.0), "train.weight_decay")
    betas = _as_float_tuple(train_cfg.get("betas", [0.9, 0.95]), "train.betas")
    grad_clip = _as_float(train_cfg.get("grad_clip", 1.0), "train.grad_clip")
    warmup_steps = _as_int(train_cfg.get("warmup_steps", 0), "train.warmup_steps")
    max_steps = _as_int(train_cfg.get("max_steps", 1000), "train.max_steps")
    batch_size = _as_int(train_cfg.get("batch_size", 32), "train.batch_size")
    grad_accum_steps = max(1, _as_int(train_cfg.get("grad_accum_steps", 1), "train.grad_accum_steps"))
    log_interval = _as_int(train_cfg.get("log_interval", 50), "train.log_interval")
    ckpt_interval = _as_int(train_cfg.get("ckpt_interval", 500), "train.ckpt_interval")
    amp_flag = bool(train_cfg.get("amp", True))

    n_layer = _as_int(model_cfg["n_layer"], "model.n_layer")
    n_head = _as_int(model_cfg["n_head"], "model.n_head")
    n_embd = _as_int(model_cfg["n_embd"], "model.n_embd")
    block_size = _as_int(model_cfg["block_size"], "model.block_size")
    dropout = _as_float(model_cfg.get("dropout", 0.0), "model.dropout")

    seed = _as_int(data_cfg.get("seed", 1337), "data.seed")
    _as_float(data_cfg.get("train_split", 0.9), "data.train_split")

    processed_dir = io_cfg["processed_dir"]
    out_dir = io_cfg["out_dir"]
    vocab_size = _as_int(tok_cfg["vocab_size"], "tokenizer.vocab_size")

    device, dtype = device_and_dtype(amp_flag)
    torch.manual_seed(seed)

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print("CUDA memory (GiB) free/total:", f"{free / 1024**3:.2f}/{total / 1024**3:.2f}")
        print("CUDA memory allocated/reserved (GiB):",
              f"{torch.cuda.memory_allocated() / 1024**3:.2f}/"
              f"{torch.cuda.memory_reserved() / 1024**3:.2f}")
    else:
        print("Using CPU for training.")

    # --- Data ---
    train_ids, val_ids = load_token_arrays(processed_dir)
    train_iter = TokenBatcher(train_ids, block_size, batch_size, seed=seed)
    val_iter = TokenBatcher(val_ids, block_size, batch_size, seed=seed + 1)

    # --- Model ---
    model = GPT(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
    ).to(device=device, dtype=dtype)

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=(device == "cuda" and amp_flag))

    # --- LR schedule (cosine with warmup) ---
    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        min_lr = lr * 0.1
        return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    os.makedirs(out_dir, exist_ok=True)

    progress = tqdm(total=max_steps, desc="Training", unit="step")
    best_val = float("inf")
    step = 0
    while step < max_steps:
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(grad_accum_steps):
            x_np, y_np = train_iter.next_batch()
            x = torch.as_tensor(x_np, device=device, dtype=torch.long)
            y = torch.as_tensor(y_np, device=device, dtype=torch.long)
            with autocast("cuda", enabled=(device == "cuda" and amp_flag)):
                _, loss = model(x, y)
            total_loss += loss.item()
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_interval == 0:
            model.eval()
            with torch.no_grad():
                vx_np, vy_np = val_iter.next_batch()
                vx = torch.as_tensor(vx_np, device=device, dtype=torch.long)
                vy = torch.as_tensor(vy_np, device=device, dtype=torch.long)
                _, vloss = model(vx, vy)
            avg_loss = total_loss / grad_accum_steps
            print(
                f"step {step:6d} | train_loss {avg_loss:.3f} | val_loss {vloss.item():.3f} | lr {lr_now:.2e}"
            )

            if vloss.item() < best_val:
                best_val = vloss.item()
                torch.save(
                    {"model": model.state_dict(), "config": config},
                    os.path.join(out_dir, "best.pt"),
                )

        if step % ckpt_interval == 0 and step > 0:
            torch.save(
                {"model": model.state_dict(), "config": config},
                os.path.join(out_dir, f"step_{step}.pt"),
            )

        step += 1
        progress.update(1)

    torch.save(
        {"model": model.state_dict(), "config": config}, os.path.join(out_dir, "final.pt")
    )
    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT using YAML config presets.")
    parser.add_argument("--config", help="Path or name of the config preset to use.")
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config presets (default: %(default)s).",
    )
    args = parser.parse_args()

    config_path, config = resolve_config(args.config, args.config_dir)
    print(f"Using config: {config_path}")
    run_training(config)


if __name__ == "__main__":
    main()
