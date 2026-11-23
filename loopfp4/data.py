"""Token shards on disk: flat files named {split}_NNNNN.bin, sampled at random offsets.

Tokens are uint16 unless a meta.json next to the shards says otherwise (vocabularies above
65,535 entries, such as Inkling's 200k, need uint32).
"""

import glob
import json
import os

import numpy as np
import torch


class ShardData:
    def __init__(self, data_dir: str, split: str, block_size: int, batch_size: int, device, seed: int = 0):
        files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        if not files:
            raise FileNotFoundError(f"no {split}_*.bin shards in {data_dir}; run prepare_data.py first")
        meta_path = os.path.join(data_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {"dtype": "uint16"}
        dtype = np.dtype(self.meta["dtype"])
        nonempty = [f for f in files if os.path.getsize(f) >= dtype.itemsize * (block_size + 2)]
        if not nonempty:
            raise ValueError(f"all {split}_*.bin shards in {data_dir} are <= block_size + 1 ({block_size + 1}) tokens")
        self.shards = [np.memmap(f, dtype=dtype, mode="r") for f in nonempty]
        sizes = np.array([len(s) for s in self.shards], dtype=np.float64)
        self.probs = sizes / sizes.sum()
        self.total_tokens = int(sizes.sum())
        self.block_size, self.batch_size, self.device = block_size, batch_size, device
        self.rng = np.random.default_rng(seed)

    def batch(self):
        T = self.block_size
        shard = self.shards[self.rng.choice(len(self.shards), p=self.probs)]
        starts = self.rng.integers(0, len(shard) - T - 1, size=self.batch_size)
        buf = np.stack([shard[s : s + T + 1] for s in starts]).astype(np.int64)
        buf = torch.from_numpy(buf)
        if torch.device(self.device).type == "cuda":
            buf = buf.pin_memory().to(self.device, non_blocking=True)
        else:
            buf = buf.to(self.device)
        return buf[:, :-1], buf[:, 1:]
