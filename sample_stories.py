"""
Generate stories from a trained TinyStories checkpoint (no retraining).

Run:   uv run sample_stories.py                       # a few default prompts
       uv run sample_stories.py "The dragon was"      # custom prompt
       uv run sample_stories.py "One day" 3 0.7        # prompt, count, temperature
"""

import os
import sys

import torch

from train_cpu import GPT, GPTConfig
from prepare import Tokenizer
from prepare_stories import TOK_DIR
from train_stories import sample, CKPT

DEFAULT_PROMPTS = ["Once upon a time", "The little dog", "One day, a girl named Lily"]


def main():
    if not os.path.exists(CKPT):
        sys.exit(f"No checkpoint at {CKPT} — run `uv run train_stories.py` first.")
    ckpt = torch.load(CKPT, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tokenizer = Tokenizer.from_directory(TOK_DIR)

    args = sys.argv[1:]
    prompts = [args[0]] if args else DEFAULT_PROMPTS
    count = int(args[1]) if len(args) > 1 else 1
    temp = float(args[2]) if len(args) > 2 else 0.8

    for p in prompts:
        for i in range(count):
            print(f'\n"{p}{sample(model, tokenizer, p, temp=temp)}"')
            print("-" * 60)


if __name__ == "__main__":
    main()
