import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, set_seed
from transformers.trainer_pt_utils import get_length_grouped_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    set_seed(42)
    train = load_from_disk(args.data / "dataset")["train"]
    count = args.batch_size * args.batches
    indices = np.random.default_rng(42).choice(len(train), count, replace=False)
    sample = train.select(indices)
    lengths = sample.select_columns(["length"]).with_format("numpy")[:]["length"]
    order = get_length_grouped_indices(lengths, args.batch_size,
                                       generator=torch.Generator().manual_seed(42))
    tokenizer = AutoTokenizer.from_pretrained(args.data / "tokenizer", local_files_only=True)
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    batches = []
    digest = hashlib.sha256()
    for start in range(0, count, args.batch_size):
        rows = sample.select(order[start:start + args.batch_size])
        batch = collator([{k: row[k] for k in ("input_ids", "attention_mask", "labels")}
                          for row in rows])
        for key in sorted(batch):
            digest.update(key.encode())
            digest.update(str(tuple(batch[key].shape)).encode())
            digest.update(batch[key].numpy().tobytes())
        batches.append({k: v.pin_memory() for k, v in batch.items()})
    real_tokens = sum(int((b["labels"] != -100).sum() + b["attention_mask"].sum()) for b in batches)
    metadata = dict(model=str(args.model), data=str(args.data), batch_size=args.batch_size,
                    batches=args.batches, batches_sha256=digest.hexdigest(), seed=42,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    precision="bf16 autocast, fp32 parameters", optimizer="AdamW foreach default",
                    lr=5e-5, betas=[0.9, 0.98], weight_decay=0.01, gradient_clip=1.0,
                    scope="precollated CPU batches; transfer+forward+backward+optimizer; no DataLoader/eval/checkpoint I/O",
                    dynamic=True, compile_mode="default", identical_initial_weights=True,
                    dropout_note="Same seed; compiled dropout RNG may differ from eager.",
                    shapes=[{k: list(v.shape) for k, v in b.items()} for b in batches])
    (args.output / "batches.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = {}
    reference_loss = None
    reference_grads = None
    for variant in ("eager", "compile"):
        set_seed(42)
        model = T5ForConditionalGeneration.from_pretrained(args.model, local_files_only=True)
        model.config.use_cache = False
        model = model.cuda()
        runner = torch.compile(model, dynamic=True) if variant == "compile" else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), weight_decay=0.01)
        model.eval()
        batch = {k: v.cuda(non_blocking=True) for k, v in batches[-1].items()}
        torch.cuda.synchronize()
        validation_start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = runner(**batch).loss
        loss.backward()
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - validation_start
        scalar = loss.item()
        grads = {k: p.grad.detach().float().cpu() for k, p in model.named_parameters()
                 if p.grad is not None}
        if variant == "eager":
            reference_loss, reference_grads = scalar, grads
            check = dict(loss=scalar, seconds=validation_seconds)
        else:
            delta = sum((grads[k] - reference_grads[k]).square().sum().item() for k in grads)
            norm = sum(v.square().sum().item() for v in reference_grads.values())
            grad_error = (delta / max(norm, 1e-30)) ** 0.5
            loss_error = abs(scalar - reference_loss) / max(abs(reference_loss), 1e-9)
            check = dict(loss=scalar, seconds=validation_seconds,
                         loss_relative_error=loss_error, gradient_relative_l2_error=grad_error,
                         passed=loss_error < 0.01 and grad_error < 0.05)
            if not check["passed"]:
                results[variant] = dict(correctness=check, status="correctness_failed")
                (args.output / "result.json").write_text(json.dumps(dict(metadata=metadata, variants=results), indent=2) + "\n")
                raise RuntimeError(f"Compiled numerical check failed: {check}")
        del batch, loss, grads
        model.zero_grad(set_to_none=True)
        model.train()
        set_seed(123)
        torch.cuda.reset_peak_memory_stats()
        phases = []
        for phase in ("cold_pass", "steady_1", "steady_2"):
            times, losses = [], []
            for cpu_batch in batches:
                torch.cuda.synchronize()
                started = time.perf_counter()
                batch = {k: v.cuda(non_blocking=True) for k, v in cpu_batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = runner(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                losses.append(loss.item())
            if not np.isfinite(losses).all():
                raise RuntimeError(f"Nonfinite losses in {variant}/{phase}")
            total = sum(times)
            phases.append(dict(phase=phase, seconds=total, examples_per_second=count / total,
                               nonpadding_tokens_per_second=real_tokens / total,
                               median_step_ms=float(np.median(times) * 1000),
                               p95_step_ms=float(np.quantile(times, 0.95) * 1000),
                               step_seconds=times, losses=losses))
            print(json.dumps(dict(variant=variant, **{k: v for k, v in phases[-1].items()
                                                    if k not in ("step_seconds", "losses")})), flush=True)
        results[variant] = dict(correctness=check, phases=phases,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(), status="ok")
        result = dict(metadata=metadata, variants=results)
        if variant == "compile":
            eager = sum(p["seconds"] for p in results["eager"]["phases"][1:])
            compiled = sum(p["seconds"] for p in phases[1:])
            extra = max(0, phases[0]["seconds"] - compiled / 2)
            check_extra = max(0, check["seconds"] - results["eager"]["correctness"]["seconds"])
            saved_per_step = (eager - compiled) / (2 * args.batches)
            result.update(steady_speedup=eager / compiled, cold_pass_extra_seconds=extra,
                          correctness_check_extra_seconds=check_extra,
                          break_even_training_steps=(extra + check_extra) / saved_per_step
                          if saved_per_step > 0 else None)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        del batch, loss, runner, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        if variant == "compile":
            from torch._dynamo.utils import counters
            (args.output / "compile_counters.json").write_text(json.dumps(
                {str(k): {str(a): int(b) for a, b in v.items()} for k, v in counters.items()}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "variants" and k != "metadata"}), flush=True)


if __name__ == "__main__":
    main()
