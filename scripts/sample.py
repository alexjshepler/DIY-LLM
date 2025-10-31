
import os, argparse
import torch

from src.model import GPT
from src.utils import device_and_dtype, resolve_config, load_any_tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path or name of the config preset to use.")
    ap.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config presets (default: %(default)s).",
    )
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=None)
    args = ap.parse_args()

    config_path, C = resolve_config(args.config, args.config_dir)
    print(f"Using config: {config_path}")
    device, dtype = device_and_dtype(bool(C["train"].get("amp", True)))

    tok = load_any_tokenizer(C["io"]["processed_dir"])
    ckpt_path = os.path.join(C["io"]["out_dir"], "final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(C["io"]["out_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = GPT(
        vocab_size=C["tokenizer"]["vocab_size"],
        n_layer=C["model"]["n_layer"],
        n_head=C["model"]["n_head"],
        n_embd=C["model"]["n_embd"],
        block_size=C["model"]["block_size"],
        dropout=C["model"]["dropout"],
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(args.prompt)
    import numpy as np
    x = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    out = tok.decode(y[0].tolist())
    print(out)

if __name__ == "__main__":
    main()
