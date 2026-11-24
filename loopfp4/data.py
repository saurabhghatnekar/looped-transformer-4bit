"""Token shards on disk: flat files named {split}_NNNNN.bin, sampled at random offsets.

Tokens are uint16 unless a meta.json next to the shards says otherwise (vocabularies above
65,535 entries, such as Inkling's 200k, need uint32).
"""

import copy
import glob
import json
import os
import queue
import threading

import numpy as np
import torch


class ShardData:
    def __init__(self, data_dir: str, split: str, block_size: int, batch_size: int, device, seed: int = 0,
                 prefetch: int = 0):
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
        self.is_cuda = torch.device(self.device).type == "cuda"
        self.rng = np.random.default_rng(seed)
        self._last_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        self.prefetch = 0
        self._thread = None
        self._stop = None
        self._queue = None
        if prefetch > 0:
            self.start_prefetch(prefetch)

    def _next_cpu_batch(self):
        T = self.block_size
        shard = self.shards[self.rng.choice(len(self.shards), p=self.probs)]
        starts = self.rng.integers(0, len(shard) - T - 1, size=self.batch_size)
        buf = np.stack([shard[s : s + T + 1] for s in starts]).astype(np.int64)
        buf = torch.from_numpy(buf)
        if self.is_cuda:
            buf = buf.pin_memory()
        return buf

    def _prefetch_worker(self, stop_evt: threading.Event, q: queue.Queue):
        try:
            while not stop_evt.is_set():
                buf = self._next_cpu_batch()
                st = copy.deepcopy(self.rng.bit_generator.state)
                while not stop_evt.is_set():
                    try:
                        q.put(("ok", buf, st), timeout=0.05)
                        break
                    except queue.Full:
                        pass
        except Exception as err:
            try:
                q.put(("err", err, None), timeout=0.5)
            except queue.Full:
                pass

    def start_prefetch(self, prefetch: int):
        self.close()
        self.prefetch = max(0, int(prefetch))
        if self.prefetch <= 0:
            return
        self._last_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._thread = threading.Thread(
            target=self._prefetch_worker, args=(self._stop, self._queue), daemon=True
        )
        self._thread.start()

    def rng_state(self):
        if self._thread is not None:
            return copy.deepcopy(self._last_rng_state)
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        prefetch = self.prefetch
        self.close()
        self.rng.bit_generator.state = state
        self._last_rng_state = copy.deepcopy(state)
        if prefetch > 0:
            self.start_prefetch(prefetch)

    def close(self):
        if self._thread is not None:
            self._stop.set()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._thread.join(timeout=2.0)
            self._thread = None
            self._stop = None
            self._queue = None

    def __del__(self):
        self.close()

    def batch(self):
        if self._thread is not None:
            kind, payload, st = self._queue.get()
            if kind == "err":
                raise payload
            self._last_rng_state = st
            buf = payload
        else:
            buf = self._next_cpu_batch()
        if self.is_cuda:
            buf = buf.to(self.device, non_blocking=True)
        else:
            buf = buf.to(self.device)
        return buf[:, :-1], buf[:, 1:]
