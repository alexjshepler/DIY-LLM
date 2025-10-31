#!/usr/bin/env python3
import os, argparse, glob, torch
from src.model import GPT
from src.utils import load_config, device_and_dtype, load_any_tokenizer

SYSTEM = ""


def _find_ckpt(C, user_path=None, use_best=False):
    out_dir = C["io"]["out_dir"]

    # 1) explicit --ckpt always wins
    if user_path:
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Checkpoint not found: {user_path}")
        return user_path

    # 2) explicit -b/--best flag
    if use_best:
        best = os.path.join(out_dir, "best.pt")
        if not os.path.exists(best):
            raise FileNotFoundError(f"--best requested but no best.pt in {out_dir}")
        return best

    # 3) otherwise choose latest step_*.pt if exists
    steps = sorted(
        glob.glob(os.path.join(out_dir, "step_*.pt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )
    if steps:
        return steps[-1]

    # 4) fallback: best.pt
    best = os.path.join(out_dir, "best.pt")
    if os.path.exists(best):
        return best

    # 5) final fallback: final.pt
    final = os.path.join(out_dir, "final.pt")
    if os.path.exists(final):
        return final

    raise FileNotFoundError(f"No checkpoint found in {out_dir}")


def _infer_vocab_from_state_dict(sd):
    # try tok_emb then head
    if "tok_emb.weight" in sd:
        return sd["tok_emb.weight"].shape[0]
    if "head.weight" in sd:
        return sd["head.weight"].shape[0]
    raise RuntimeError("Cannot infer vocab size from checkpoint state_dict")


def load_model_and_tok(C, device, ckpt_path):
    tok = load_any_tokenizer(C["io"]["processed_dir"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"]
    vocab_in_ckpt = _infer_vocab_from_state_dict(sd)

    # Build model with vocab from the checkpoint to avoid size mismatches
    model = GPT(
        vocab_size=vocab_in_ckpt,
        n_layer=C["model"]["n_layer"],
        n_head=C["model"]["n_head"],
        n_embd=C["model"]["n_embd"],
        block_size=C["model"]["block_size"],
        dropout=C["model"]["dropout"],
    ).to(device=device)
    model.load_state_dict(sd)
    model.eval()

    # sanity: warn if tokenizer vocab != checkpoint vocab
    try:
        tv = tok.vocab_size()
        if tv != vocab_in_ckpt:
            print(
                f"[warn] Tokenizer vocab_size ({tv}) != checkpoint vocab_size ({vocab_in_ckpt}). "
                f"If outputs look garbled, re-run prepare/train so they match, or load a matching ckpt."
            )
    except Exception:
        pass

    return tok, model


@torch.no_grad()
def generate(
    model, tok, device, prompt, max_new_tokens=200, temperature=0.9, top_k=200
):
    ids = tok.encode(prompt)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    for _ in range(max_new_tokens):
        idx = x[:, -model.block_size :]
        logits, _ = model(idx)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return tok.decode(x[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/4090-fast.yaml")
    ap.add_argument("--ckpt", default=None, help="Path to a specific checkpoint .pt")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("-b","--best",action="store_true",help="Force loading best.pt instead of latest")
    args = ap.parse_args()

    C = load_config(args.config)
    device, _ = device_and_dtype(C["train"]["amp"])

    ckpt_path = _find_ckpt(C, user_path=args.ckpt, use_best=args.best)
    print(f"[info] Loading checkpoint: {ckpt_path}")
    tok, model = load_model_and_tok(C, device, ckpt_path)

    history = []
    print("🔥 Chat REPL — Ctrl+C to exit")
    while True:
        try:
            user = input("\nYou: ").strip()
            if not user:
                continue
            history.append(("user", user))
            formatted = SYSTEM + "\n"
            for role, content in history[-6:]:
                formatted += f"{role.upper()}: {content}\n"
            formatted += "ASSISTANT: "
            out = generate(
                model,
                tok,
                device,
                formatted,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            resp = out.split("ASSISTANT:")[-1].strip()
            history.append(("assistant", resp))
            print(f"Assistant: {resp}")
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
