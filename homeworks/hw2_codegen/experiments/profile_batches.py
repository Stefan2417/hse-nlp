"""CPU-only padding check on official prepared examples; no training or GPU use."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers.trainer_pt_utils import get_length_grouped_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data = load_from_disk(args.data_dir / "dataset")
    train = data["train"]
    indices = np.random.default_rng(42).choice(len(train), min(32768, len(train)), replace=False)
    sample = train.select(indices)
    columns = sample.select_columns(["source_tokens", "target_tokens", "length"]).with_format("numpy")[:]
    source = np.minimum(columns["source_tokens"], 256)
    target = columns["target_tokens"]
    results = []
    for name, lengths, batch_size in [
        ("source_only", source, 32),
        ("source_plus_target", columns["length"], 32),
        ("source_plus_target", columns["length"], 128),
    ]:
        order = get_length_grouped_indices(lengths, 128, generator=torch.Generator().manual_seed(42))
        real = padded = 0
        largest = (0, 0)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            src = int(source[batch].max() + 7) // 8 * 8
            tgt = int(target[batch].max() + 7) // 8 * 8
            real += int((source[batch] + target[batch]).sum())
            padded += len(batch) * (src + tgt)
            if src + tgt > sum(largest):
                largest = (src, tgt)
        results.append({"grouping": name, "batch_size": batch_size,
                        "padding_fraction": 1 - real / padded,
                        "padded_positions": padded, "largest_batch_lengths": largest})
    fetch_started = time.perf_counter()
    fetched = 0
    for start in range(0, 4096, 128):
        fetched += len(train[indices[start:start + 128].tolist()]["input_ids"])
    result = {"data_dir": str(args.data_dir), "rows": {k: len(v) for k, v in data.items()},
              "sample_rows": len(sample), "batches": results,
              "random_fetch_rows": fetched, "random_fetch_seconds": time.perf_counter() - fetch_started,
              "total_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
