"""Generate text from a trained checkpoint (byte-level, tiktoken gpt2, or HuggingFace tokenizer).

    python sample.py runs/ink-smoke/ckpt.pt "Once upon a time" [n_loops]
    python sample.py runs/5b/ckpt.pt --prompt "Once upon a time" --tokenizer gpt2 --loops 3
"""

import argparse
import contextlib

import torch

from loopfp4 import load_checkpoint


def _pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_tokenizer(spec: str, vocab: int):
    if spec == "auto":
        if vocab <= 256:
            spec = "bytes"
        elif vocab in (50257, 50304):
            spec = "gpt2"
        else:
            raise SystemExit(
                f"checkpoint vocab size is {vocab}; pass --tokenizer gpt2 or a HuggingFace tokenizer id/path"
            )
    if spec == "bytes":
        return (
            lambda s: list(s.encode("utf-8")),
            lambda ids: bytes(i for i in ids if 0 <= i < 256).decode("utf-8", "replace"),
        )
    if spec == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, enc.decode
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec)
    return (lambda s: tok.encode(s, add_special_tokens=False)), tok.decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to ckpt.pt or exported model-nvfp4.pt")
    ap.add_argument("prompt_pos", nargs="?", default=None)
    ap.add_argument("loops_pos", nargs="?", type=int, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--loops", type=int, default=None, help="passes through the looped section")
    ap.add_argument("--tokenizer", default="auto", help="auto | bytes | gpt2 | <hf-tokenizer-id-or-path>")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = _pick_device(args.device)
    model, ck = load_checkpoint(args.ckpt, map_location="cpu")
    model = model.to(device).eval()

    vocab = ck["config"].get("unpadded_vocab_size") or ck["config"].get("vocab_size")
    encode, decode = _make_tokenizer(args.tokenizer, vocab)
    prompt = args.prompt if args.prompt is not None else (
        args.prompt_pos if args.prompt_pos is not None else "Once upon a time"
    )
    n_loops = args.loops if args.loops is not None else args.loops_pos

    prompt_ids = encode(prompt) or [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else contextlib.nullcontext()
    torch.manual_seed(args.seed)
    with amp:
        out = model.generate(
            idx, args.max_tokens, temperature=args.temperature, top_k=args.top_k, n_loops=n_loops
        )[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()
