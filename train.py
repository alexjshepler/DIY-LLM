import os, argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from src.model import GPT
from src.data import load_token_arrays, TokenBatcher
from src.utils import load_config, device_and_dtype

free, total = torch.cuda.mem_get_info()
print("free/total (GiB):", free / 1024**3, total / 1024**3)
print("allocated (GiB):", torch.cuda.memory_allocated() / 1024**3)
print("reserved  (GiB):", torch.cuda.memory_reserved() / 1024**3)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def _as_float(x, name):
    try:
        return float(x)
    except Exception:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    C = load_config(args.config)

    # --- Coerce critical config fields to correct types (robust to YAML strings) ---
    tcfg = C["train"]
    dcfg = C["data"]
    mcfg = C["model"]
    icfg = C["io"]
    tokg = C["tokenizer"]

    lr = _as_float(tcfg["lr"], "train.lr")
    weight_decay = _as_float(tcfg.get("weight_decay", 0.0), "train.weight_decay")
    betas = _as_float_tuple(tcfg.get("betas", [0.9, 0.95]), "train.betas")
    grad_clip = _as_float(tcfg.get("grad_clip", 1.0), "train.grad_clip")
    warmup_steps = _as_int(tcfg.get("warmup_steps", 0), "train.warmup_steps")
    max_steps = _as_int(tcfg.get("max_steps", 1000), "train.max_steps")
    batch_size = _as_int(tcfg.get("batch_size", 32), "train.batch_size")
    log_interval = _as_int(tcfg.get("log_interval", 50), "train.log_interval")
    ckpt_interval = _as_int(tcfg.get("ckpt_interval", 500), "train.ckpt_interval")
    amp_flag = bool(tcfg.get("amp", True))

    n_layer = _as_int(mcfg["n_layer"], "model.n_layer")
    n_head = _as_int(mcfg["n_head"], "model.n_head")
    n_embd = _as_int(mcfg["n_embd"], "model.n_embd")
    block_size = _as_int(mcfg["block_size"], "model.block_size")
    dropout = _as_float(mcfg.get("dropout", 0.0), "model.dropout")

    seed = _as_int(dcfg.get("seed", 1337), "data.seed")
    train_split = _as_float(
        dcfg.get("train_split", 0.9), "data.train_split"
    )  # not used here, but OK

    processed_dir = icfg["processed_dir"]
    out_dir = icfg["out_dir"]
    vocab_size = _as_int(tokg["vocab_size"], "tokenizer.vocab_size")

    device, dtype = device_and_dtype(amp_flag)
    torch.manual_seed(seed)

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
    ).to(device=device)

    # --- Optimizer ---
    opt = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=(device == "cuda" and amp_flag))

    # --- LR schedule (cosine with warmup) ---
    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        # cosine decay to 10% of base lr
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        min_lr = lr * 0.1
        return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    os.makedirs(out_dir, exist_ok=True)

    from tqdm import tqdm
    progress = tqdm(total=max_steps, desc="Training", unit="step")
    step = 0
    best_val = float("inf")
    while step < max_steps:
        # batch
        x_np, y_np = train_iter.next_batch()
        x = torch.tensor(x_np, device=device, dtype=torch.long)
        y = torch.tensor(y_np, device=device, dtype=torch.long)

        # set lr
        lr_now = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        # train step
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device == "cuda" and amp_flag)):
            logits, loss = model(x, y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()

        # log + quick val
        if step % log_interval == 0:
            model.eval()
            with torch.no_grad():
                vx_np, vy_np = val_iter.next_batch()
                vx = torch.tensor(vx_np, device=device, dtype=torch.long)
                vy = torch.tensor(vy_np, device=device, dtype=torch.long)
                _, vloss = model(vx, vy)
            print(
                f"step {step:6d} | train_loss {loss.item():.3f} | val_loss {vloss.item():.3f} | lr {lr_now:.2e}"
            )

            if vloss.item() < best_val:
                best_val = vloss.item()
                torch.save(
                    {"model": model.state_dict(), "config": C},
                    os.path.join(out_dir, "best.pt"),
                )

        # periodic checkpoints
        if step % ckpt_interval == 0 and step > 0:
            torch.save(
                {"model": model.state_dict(), "config": C},
                os.path.join(out_dir, f"step_{step}.pt"),
            )

        step += 1
        progress.update(1)

    torch.save(
        {"model": model.state_dict(), "config": C}, os.path.join(out_dir, "final.pt")
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
