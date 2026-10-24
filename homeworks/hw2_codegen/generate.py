import json
import time
from pathlib import Path

import hydra
import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    started = time.perf_counter()
    device = cfg.generation.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    precision = cfg.generation.dtype
    if precision == "auto":
        precision = "float16" if device.type == "cuda" else "float32"
    tokenizer = AutoTokenizer.from_pretrained(cfg.generation.model_dir)
    prepared_tokenizer = AutoTokenizer.from_pretrained(Path(cfg.data_dir) / "tokenizer")
    if tokenizer.backend_tokenizer.to_str() != prepared_tokenizer.backend_tokenizer.to_str():
        raise ValueError("Model tokenizer differs from prepared data; use the matching data_dir.")
    model = T5ForConditionalGeneration.from_pretrained(
        cfg.generation.model_dir, dtype=getattr(torch, precision)
    ).to(device).eval()
    data = load_from_disk(Path(cfg.data_dir) / "dataset")[cfg.generation.split]
    if "question" not in data.column_names:
        raise ValueError("Generate on validation or test; training text is not stored.")
    if cfg.generation.max_samples is not None:
        data = data.select(range(min(len(data), cfg.generation.max_samples)))
    output = Path(cfg.generation.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_tokens = capped = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    with output.open("w") as stream, torch.inference_mode():
        for start in tqdm(range(0, len(data), cfg.generation.batch_size), desc="Generate"):
            rows = data.select(range(start, min(start + cfg.generation.batch_size, len(data))))
            batch = tokenizer.pad(
                [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in rows],
                padding=True, pad_to_multiple_of=8, return_tensors="pt",
            ).to(device)
            predictions = model.generate(
                **batch, max_new_tokens=cfg.generation.max_new_tokens,
                num_beams=cfg.generation.num_beams, do_sample=False, use_cache=True,
            ).cpu().tolist()
            for index, (row, tokens) in enumerate(zip(rows, predictions), start=start):
                tokens = tokens[1:]  # T5 starts with decoder_start_token_id.
                finished = tokenizer.eos_token_id in tokens
                count = tokens.index(tokenizer.eos_token_id) + 1 if finished else len(tokens)
                generated_tokens += count
                capped += not finished
                record = {"id": index, "question": row["question"],
                          "code": tokenizer.decode(tokens, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False),
                          "generated_tokens": count, "hit_length_limit": not finished}
                if "code" in row:
                    record["reference"] = row["code"]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - generation_started
    metrics = {"examples": len(data), "generation_seconds": seconds,
               "including_load_seconds": time.perf_counter() - started,
               "examples_per_second": len(data) / seconds, "generated_tokens": generated_tokens,
               "hit_length_limit": capped, "device": str(device), "dtype": precision,
               "model_dir": cfg.generation.model_dir, "split": cfg.generation.split}
    output.with_suffix(".timing.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
