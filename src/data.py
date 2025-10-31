
import os
import numpy as np
from typing import Tuple

def load_token_arrays(processed_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    train = np.load(os.path.join(processed_dir, "train.npy"))
    val = np.load(os.path.join(processed_dir, "val.npy"))
    return train, val

class TokenBatcher:
    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int, seed: int = 1337):
        self.tokens = tokens
        self.block = block_size
        self.bs = batch_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self):
        ix = self.rng.integers(0, len(self.tokens) - self.block - 1, size=self.bs)
        x = np.stack([self.tokens[i:i+self.block] for i in ix])
        y = np.stack([self.tokens[i+1:i+1+self.block] for i in ix])
        return x, y
