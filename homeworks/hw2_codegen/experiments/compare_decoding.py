import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from execution_validation import ExecutionEvaluator, generate_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Maximum generated hypotheses per batch, including beams/samples")
    parser.add_argument("--max-samples", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.validation.read_text().splitlines()][:args.max_samples]
    torch.set_num_threads(4)
    evaluator = ExecutionEvaluator([row["reference"] for row in rows])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = T5ForConditionalGeneration.from_pretrained(args.model, dtype=dtype,
                                                       local_files_only=True).to(args.device).eval()
    manifest = dict(model=str(args.model.resolve()), device=args.device, dtype=str(dtype),
                    model_sha256=hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest(),
                    validation_sha256=hashlib.sha256(args.validation.read_bytes()).hexdigest(),
                    examples=len(rows), max_new_tokens=256, seed=42,
                    hypothesis_batch_size=args.batch_size,
                    rtol=1e-6, atol=1e-6, label_source="execution of official held-out code", t4_timing=False)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output belongs to a different evaluation configuration")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = {}
    for name, beams in (("greedy", 1), ("beam2", 2), ("beam4", 4)):
        path = args.output / f"{name}.generated.json"
        started = time.perf_counter()
        cache_used = path.exists()
        batch_size = max(1, args.batch_size // beams)
        if cache_used:
            candidates = json.loads(path.read_text())
        else:
            candidates = generate_candidates(model, tokenizer, rows, batch_size,
                                               num_beams=beams, num_return_sequences=beams)
        path.write_text(json.dumps(candidates, ensure_ascii=False) + "\n")
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device.startswith("cuda"):
            torch.cuda.synchronize()
        generation_seconds = None if cache_used else time.perf_counter() - started
        metrics, details = evaluator.evaluate([[candidate["code"] for candidate in group]
                                              for group in candidates])
        metrics.update(generation_seconds=generation_seconds,
                       prompt_batch_size=batch_size,
                       total_seconds=time.perf_counter() - started,
                       hit_length_limit=sum(candidate["hit_length_limit"] for group in candidates for candidate in group),
                       cache_used=cache_used)
        (args.output / f"{name}.outcomes.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in details))
        summaries[name] = metrics
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(name, json.dumps(metrics), flush=True)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
