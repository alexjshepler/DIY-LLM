import glob
import os
import sys
from typing import Dict, List, Tuple

import yaml
import torch

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def _candidate_config_paths(config_path: str, config_dir: str) -> List[str]:
    base = os.path.expanduser(config_path)
    if os.path.isabs(base):
        candidates = [base]
    else:
        candidates = [base, os.path.join(config_dir, base)]

    if not base.endswith(('.yaml', '.yml')):
        candidates.extend([c + '.yaml' for c in list(candidates)])
        candidates.extend([c + '.yml' for c in list(candidates)])

    # remove duplicates while preserving order
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _load_config_metadata(path: str) -> Tuple[str, str, Dict]:
    config = load_config(path)
    meta = config.get('meta', {}) if isinstance(config, dict) else {}
    name = meta.get('name') or os.path.splitext(os.path.basename(path))[0]
    description = (meta.get('description') or '').strip()
    return name, description, config


def list_config_presets(config_dir: str = 'configs') -> List[Dict]:
    paths = sorted(
        glob.glob(os.path.join(config_dir, '*.yaml'))
        + glob.glob(os.path.join(config_dir, '*.yml'))
    )
    presets: List[Dict] = []
    for path in paths:
        try:
            name, description, config = _load_config_metadata(path)
        except Exception as exc:
            print(f"Warning: failed to load config '{path}': {exc}", file=sys.stderr)
            continue
        presets.append(
            {
                'path': os.path.abspath(path),
                'name': name,
                'description': description,
                'config': config,
            }
        )
    return presets


def resolve_config(config_path: str = None, config_dir: str = 'configs'):
    """Resolve a config path or prompt the user to choose a preset."""

    if config_path:
        for candidate in _candidate_config_paths(config_path, config_dir):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate), load_config(candidate)
        raise FileNotFoundError(
            f"Could not find config '{config_path}'. Looked in: "
            + ", ".join(_candidate_config_paths(config_path, config_dir))
        )

    presets = list_config_presets(config_dir)
    if not presets:
        raise FileNotFoundError(
            f"No config presets found in '{config_dir}'. Specify --config to provide a path."
        )

    print("Available config presets:")
    for idx, preset in enumerate(presets, start=1):
        desc = preset['description'] or 'No description provided.'
        print(f"  [{idx}] {preset['name']} ({os.path.relpath(preset['path'])})")
        print(f"      {desc}")

    if not sys.stdin.isatty():
        raise RuntimeError(
            "No config provided and input is not interactive. Rerun with --config to choose a preset."
        )

    while True:
        choice = input("Select a config by number: ").strip()
        if not choice:
            continue
        if not choice.isdigit():
            print("Please enter a number corresponding to the preset you want.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(presets):
            preset = presets[idx - 1]
            return preset['path'], preset['config']
        print(f"Choice must be between 1 and {len(presets)}.")

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
