import os, yaml, torch

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def device_and_dtype(amp: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (amp and device=="cuda") else torch.float32
    return device, dtype


def load_any_tokenizer(processed_dir: str):
    """
    Returns an object with encode(text)->List[int], decode(ids)->str, vocab_size().
    Prefers HF tokenizer if present; otherwise uses our BPETokenizer.
    """
    import os

    hf_path = os.path.join(processed_dir, "hf_tokenizer.json")
    if os.path.exists(hf_path):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(hf_path)

        class _HFWrapper:
            def encode(self, text: str):
                return tok.encode(text).ids

            def decode(self, ids):
                return tok.decode(ids)

            def vocab_size(self):
                return tok.get_vocab_size()

        return _HFWrapper()
    else:
        from src.tokenizer import BPETokenizer

        tok = BPETokenizer.load(processed_dir)

        class _BPEWrapper:
            def encode(self, text: str):
                return tok.encode(text)

            def decode(self, ids):
                return tok.decode(ids)

            def vocab_size(self):
                return len(tok.token_to_id)

        return _BPEWrapper()
