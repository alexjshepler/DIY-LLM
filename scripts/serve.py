
import os, argparse, uvicorn, torch
from fastapi import FastAPI
from pydantic import BaseModel
from src.model import GPT
from src.tokenizer import BPETokenizer
from src.utils import load_config, device_and_dtype

app = FastAPI(title="MiniGPT API")

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.9
    top_k: int | None = 200

def load_everything(config_path):
    C = load_config(config_path)
    device, dtype = device_and_dtype(C["train"]["amp"])
    tok = BPETokenizer.load(C["io"]["processed_dir"])
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
    return C, device, tok, model

@app.on_event("startup")
def _startup():
    global C, device, tok, model
    config_path = os.environ.get("LLM_CONFIG", "configs/small.yaml")
    C, device, tok, model = load_everything(config_path)

@app.post("/generate")
def generate(req: GenRequest):
    ids = tok.encode(req.prompt)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_k=req.top_k)
    out = tok.decode(y[0].tolist())
    return {"text": out}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    os.environ["LLM_CONFIG"] = args.config
    uvicorn.run("scripts.serve:app", host=args.host, port=args.port, reload=False)
