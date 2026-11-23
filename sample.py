"""Generate text from a byte-level checkpoint: python sample.py runs/ink-smoke/ckpt.pt "Once upon a time" [n_loops]"""

import sys

import torch

from loopfp4 import load_checkpoint

model, _ = load_checkpoint(sys.argv[1])
model.eval()
prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
n_loops = int(sys.argv[3]) if len(sys.argv) > 3 else None
idx = torch.tensor([list(prompt.encode())])
torch.manual_seed(0)
out = model.generate(idx, 300, n_loops=n_loops)[0].tolist()
print(bytes(out).decode("utf-8", "replace"))
