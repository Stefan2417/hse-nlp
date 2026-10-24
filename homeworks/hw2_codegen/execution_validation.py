import math
from concurrent.futures import ThreadPoolExecutor

from sandbox_runner import equal, run_program


def numeric(result):
    value = result.get("value")
    try:
        return result.get("ok", False) and type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def vote(results, scores, rtol=1e-6, atol=1e-6):
    groups = []
    for index, result in enumerate(results):
        if not numeric(result):
            continue
        for group in groups:
            if equal(result["value"], results[group[0]]["value"], rtol, atol):
                group.append(index)
                break
        else:
            groups.append([index])
    if not groups:
        return 0
    group = max(groups, key=lambda g: (len(g), max(scores[i] for i in g)))
    return max(group, key=lambda i: scores[i])


class ExecutionEvaluator:
    def __init__(self, references, timeout=2, workers=4, backend="auto", rtol=1e-6, atol=1e-6):
        self.timeout, self.workers, self.backend = timeout, workers, backend
        self.rtol, self.atol = rtol, atol
        self.cache = {}
        check = run_program("def solve():\n    return 2\n", timeout, backend)
        if not check.get("ok") or check.get("value") != 2:
            raise RuntimeError(f"Execution sandbox failed its preflight: {check}")
        self.references = self.execute(references)

    def execute(self, codes):
        missing = list(dict.fromkeys(code for code in codes if code not in self.cache))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            values = pool.map(lambda code: run_program(code, self.timeout, self.backend), missing)
            self.cache.update(zip(missing, values))
        return [self.cache[code] for code in codes]

    def evaluate(self, candidates, scores=None, voting=False):
        if len(candidates) != len(self.references):
            raise ValueError("Candidate/reference count differs")
        self.execute([code for group in candidates for code in group])
        details = []
        for index, (group, reference) in enumerate(zip(candidates, self.references)):
            results = [self.cache[code] for code in group]
            matches = [numeric(reference) and numeric(result)
                       and equal(result["value"], reference["value"], self.rtol, self.atol)
                       for result in results]
            selected = vote(results, scores[index], self.rtol, self.atol) if voting else 0
            details.append(dict(id=index, reference=reference, candidates=group,
                                scores=scores[index] if scores is not None else None,
                                executions=results, selected=selected,
                                correct=matches[selected], oracle_correct=any(matches)))
        valid = [row for row in details if numeric(row["reference"])]
        if not valid:
            raise RuntimeError("No numeric reference programs executed successfully")
        metrics = dict(examples=len(details), valid_references=len(valid),
                       invalid_references=len(details) - len(valid),
                       pass_at_1=sum(row["correct"] for row in valid) / len(valid),
                       oracle_at_k=sum(row["oracle_correct"] for row in valid) / len(valid),
                       pass_at_1_all=sum(row["correct"] for row in details) / len(details),
                       prediction_numeric_rate=sum(numeric(row["executions"][row["selected"]])
                                                   for row in details) / len(details),
                       candidates_per_question=len(candidates[0]))
        return metrics, details


def generate_candidates(model, tokenizer, rows, batch_size=32, max_new_tokens=256,
                        num_beams=1, num_return_sequences=1, sample=False, seed=42):
    import torch
    device = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]), reverse=True)
    answers = [None] * len(rows)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(seed)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = tokenizer.pad([{k: rows[i][k] for k in ("input_ids", "attention_mask")}
                                   for i in indices], padding=True, pad_to_multiple_of=8,
                                  return_tensors="pt").to(device)
            options = dict(max_new_tokens=max_new_tokens, num_beams=num_beams,
                           num_return_sequences=num_return_sequences, do_sample=sample, use_cache=True)
            if sample:
                options.update(temperature=0.5, top_p=0.95)
            predictions = model.generate(**batch, **options)[:, 1:].cpu().tolist()
            for offset, index in enumerate(indices):
                group = []
                for tokens in predictions[offset * num_return_sequences:(offset + 1) * num_return_sequences]:
                    finished = tokenizer.eos_token_id in tokens
                    if finished:
                        tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                    group.append(dict(code=tokenizer.decode(tokens, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False),
                                      tokens=tokens, hit_length_limit=not finished))
                answers[index] = group
            print(f"generated {min(start + batch_size, len(order))}/{len(order)} "
                  f"beam={num_beams} samples={num_return_sequences if sample else 0}", flush=True)
    return answers


def candidate_scores(model, tokenizer, rows, candidates, batch_size=8):
    import torch
    from transformers import DataCollatorForSeq2Seq
    flat = [(index, candidate) for index, group in enumerate(candidates) for candidate in group]
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    device = next(model.parameters()).device
    scores = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = collator([{**{k: rows[i][k] for k in ("input_ids", "attention_mask")},
                               "labels": list(candidate["tokens"])}
                              for i, candidate in flat[start:start + batch_size]])
            batch = {k: value.to(device) for k, value in batch.items()}
            output = model(**batch, use_cache=False)
            losses = torch.nn.functional.cross_entropy(output.logits.float().transpose(1, 2),
                                                       batch["labels"], reduction="none", ignore_index=-100)
            values = -losses.sum(1) / (batch["labels"] != -100).sum(1)
            scores.extend(values.cpu().tolist())
    size = len(candidates[0])
    return [scores[i:i + size] for i in range(0, len(scores), size)]
