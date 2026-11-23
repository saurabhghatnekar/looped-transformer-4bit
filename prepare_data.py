"""Tokenize text into token shards for train.py.

Local text file, byte-level (laptop check):
    python prepare_data.py --source data/raw/tinystories-valid.txt --tokenizer bytes --out data/tinystories-bytes

Hugging Face dataset, streamed, GPT-2 BPE (real runs):
    python prepare_data.py --source hf:HuggingFaceFW/fineweb-edu:sample-100BT --tokenizer gpt2 \
        --out /data/fineweb-edu-100b --max-tokens 100e9

Inkling 2 Mini's own tokenizer (200k vocabulary, fetched from Hugging Face on first use):
    python prepare_data.py --source hf:HuggingFaceFW/fineweb-edu:sample-100BT \
        --tokenizer hf:thinkingmachines/Inkling-2-Mini --out /data/fineweb-edu-100b-inkling --max-tokens 100e9

Writes meta.json (token dtype, vocabulary size) next to the shards.
"""

import argparse
import json
import os

import numpy as np


DTYPE = np.uint16


def make_encoder(name):
    """Returns (encode_batch(texts) -> list of token lists, document separator id, vocabulary size)."""
    if name == "bytes":
        return (lambda texts: [list(t.encode("utf-8")) for t in texts]), 0, 256
    if name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode_ordinary_batch, enc.eot_token, enc.n_vocab
    from tokenizers import Tokenizer  # hf:<repo>
    tok = Tokenizer.from_pretrained(name[3:])
    sep = tok.token_to_id("<|endoftext|>")
    encode = lambda texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
    return encode, (0 if sep is None else sep), tok.get_vocab_size()


def write_shard(out, split, n, tokens):
    path = os.path.join(out, f"{split}_{n:05d}.bin")
    np.asarray(tokens, dtype=DTYPE).tofile(path)
    print(f"wrote {path}  {len(tokens):,} tokens", flush=True)


def from_file(args):
    with open(args.source, "rb") as f:
        raw = f.read()
    if args.tokenizer == "bytes":
        toks = np.frombuffer(raw, dtype=np.uint8).astype(DTYPE)
    else:
        toks = np.array(args.encode([raw.decode("utf-8", "replace")])[0], dtype=DTYPE)
    n_val = max(int(len(toks) * args.val_frac), 1)
    write_shard(args.out, "val", 0, toks[-n_val:])
    train = toks[:-n_val]
    for n, i in enumerate(range(0, len(train), args.shard_tokens)):
        write_shard(args.out, "train", n, train[i : i + args.shard_tokens])


def from_hf(args):
    from datasets import load_dataset

    parts = args.source[3:].split(":")
    ds = load_dataset(parts[0], parts[1] if len(parts) > 1 else None, split=args.split, streaming=True)
    encode = lambda texts: [[args.sep] + list(t) for t in args.encode(texts)]

    buf, total, n_shard, wrote_val = [], 0, 0, False
    for batch in ds.iter(batch_size=1024):
        for toks in encode(batch[args.text_field]):
            buf.extend(toks)
        while len(buf) >= args.shard_tokens:
            chunk, buf = buf[: args.shard_tokens], buf[args.shard_tokens :]
            if not wrote_val:  # first shard is held out for validation
                write_shard(args.out, "val", 0, chunk)
                wrote_val = True
            else:
                write_shard(args.out, "train", n_shard, chunk)
                n_shard += 1
                total += len(chunk)
        if args.max_tokens and total >= args.max_tokens:
            return
    if buf:
        write_shard(args.out, "train" if wrote_val else "val", n_shard, buf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to a text file, or hf:<dataset>[:<config>]")
    ap.add_argument("--tokenizer", default="gpt2", help="bytes | gpt2 | hf:<repo with a tokenizer.json>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--shard-tokens", type=lambda s: int(float(s)), default=100_000_000)
    ap.add_argument("--max-tokens", type=lambda s: int(float(s)), default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.encode, args.sep, vocab = make_encoder(args.tokenizer)
    DTYPE = np.uint16 if vocab <= 65536 else np.uint32
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"dtype": np.dtype(DTYPE).name, "vocab_size": vocab, "tokenizer": args.tokenizer}, f)
    (from_hf if args.source.startswith("hf:") else from_file)(args)
