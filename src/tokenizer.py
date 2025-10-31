import json
from collections import Counter
from typing import List, Tuple
from tqdm import tqdm


class BPETokenizer:
    """
    Very small BPE tokenizer (character baseline -> BPE merges).
    Trains merges on a text corpus and provides encode/decode.
    """

    def __init__(self, vocab_size: int = 32000, min_freq: int = 1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id = {}
        self.id_to_token = {}

    @staticmethod
    def _get_words(text: str):
        words = []
        for w in text.split():
            words.append(tuple(list(w) + ["</w>"]))
        return words

    def train(self, text: str):
        # Build initial char-level corpus
        corpus = self._get_words(text)
        vocab_counter = Counter(corpus)
        token_freq = Counter()
        for word, freq in vocab_counter.items():
            for ch in word:
                token_freq[ch] += freq

        # prune low-frequency chars
        for t in list(token_freq.keys()):
            if token_freq[t] < self.min_freq:
                del token_freq[t]

        # initialize vocab
        self.vocab = {t: i for i, t in enumerate(sorted(token_freq.keys()))}
        self.vocab["<unk>"] = len(self.vocab)

        def get_stats(vocab_counter):
            pairs = Counter()
            for word, freq in vocab_counter.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            return pairs

        def merge_vocab(pair, v_in):
            v_out = Counter()
            (a, b) = pair
            for word, freq in v_in.items():
                w = list(word)
                i = 0
                new_word = []
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                        new_word.append(a + b)
                        i += 2
                    else:
                        new_word.append(w[i])
                        i += 1
                v_out[tuple(new_word)] += freq
            return v_out

        # --- Progress bar for the merge loop ---
        num_initial = len(self.vocab)
        target_merges = self.vocab_size - num_initial
        if target_merges < 0:
            target_merges = 0

        with tqdm(
            total=target_merges, desc="Training BPE Merges", unit="merge"
        ) as pbar:
            while True:
                if len(self.vocab) >= self.vocab_size:
                    break

                pairs = get_stats(vocab_counter)
                if not pairs:
                    break

                best = pairs.most_common(1)[0][0]
                new_token = best[0] + best[1]
                self.merges.append(best)
                self.vocab[new_token] = len(self.vocab)
                vocab_counter = merge_vocab(best, vocab_counter)

                pbar.update(1)

                # Safety stop if nothing changes
                if len(pairs) == 0:
                    break

        # finalize id maps
        self.token_to_id = dict(self.vocab)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def save(self, dirpath: str):
        import os

        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "hf_tokenizer.json"), "w") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "min_freq": self.min_freq,
                    "vocab": self.vocab,
                    "merges": self.merges,
                },
                f,
            )

    @classmethod
    def load(cls, dirpath: str):
        import os

        with open(os.path.join(dirpath, "hf_tokenizer.json")) as f:
            obj = json.load(f)
        tok = cls(obj["vocab_size"], obj["min_freq"])
        tok.vocab = obj["vocab"]
        tok.merges = [tuple(m) for m in obj["merges"]]
        tok.token_to_id = dict(tok.vocab)
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        return tok

    def _bpe_encode_word(self, w: str):
        if not self.merges:
            return list(w) + ["</w>"]
        tokens = list(w) + ["</w>"]
        changed = True
        while changed:
            changed = False
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.merges:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                    changed = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str):
        ids = []
        for w in text.split():
            for t in self._bpe_encode_word(w):
                ids.append(self.token_to_id.get(t, self.token_to_id["<unk>"]))
            ids.append(self.token_to_id.get(" ", self.token_to_id["<unk>"]))
        return ids

    def decode(self, ids: List[int]):
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        out = ""
        for t in tokens:
            if t == "</w>":
                continue
            out += t
        return out
