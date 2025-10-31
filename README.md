# llm-from-scratch (minimal GPT pretraining)

A tiny, self-contained repo to train a GPT-style language model **from scratch** on your own text.
Drop `.txt` files into `data/raw/`, run one command, and it will:
1) train a Byte Pair Encoding (BPE) tokenizer,
2) build a train/val token dataset,
3) train a small Transformer decoder (GPT),
4) save checkpoints and a final model, and
5) let you sample text.

> **Goal:** clarity over speed. This is not meant to beat SOTA; it's a clean starting point you can actually understand.

---

## Quickstart

```bash
# 0) (optional) create venv
python -m venv .venv && source .venv/bin/activate

# 1) install
pip install -r requirements.txt

# 2) add your text data
# Put one or more .txt files into data/raw/

# 3) prepare data (train tokenizer + tokenized dataset)
python scripts/prepare_data.py --config configs/small.yaml

# 4) train
python train.py --config configs/small.yaml

# 5) sample
python scripts/sample.py --config configs/small.yaml --prompt "Once upon a time"
```

### Hardware
Runs on CPU or GPU. If CUDA is available, it uses **automatic mixed precision**.

### Where things go
- Tokenizer files: `data/processed/tokenizer.json`, `data/processed/vocab.json`, `data/processed/merges.txt`
- Token dataset: `data/processed/train.npy`, `data/processed/val.npy`
- Checkpoints: `checkpoints/step_XXXX.pt`
- Final model: `checkpoints/final.pt`

---

## Configs

Edit the YAML in `configs/*.yaml`. Key knobs:
- `tokenizer.vocab_size`: default 32_000
- `model.n_layer`, `model.n_head`, `model.n_embd`, `model.block_size`
- `train.batch_size`, `train.max_steps`, `train.lr`, `train.weight_decay`, `train.grad_clip`
- `data.train_split` (e.g., 0.9)
- `data.min_freq` to drop ultra-rare chars before BPE (optional)

---

## Bring your own data

Just plain UTF-8 text files in `data/raw/`. If you have structured data, first convert it to text.
For very small datasets, reduce `vocab_size` and `block_size`.

---

## Sampling

```bash
python scripts/sample.py --config configs/small.yaml --prompt "The function returns" --max_new_tokens 200 --temperature 0.8 --top_k 200
```

---

## Caveats

- This is a teaching implementation. For billion-parameter scale, consider:
  - [nanoGPT] or [lit-gpt] for fast single-node runs
  - [Megatron-LM] + [DeepSpeed] for multi-node scale

MIT licensed, have fun!
