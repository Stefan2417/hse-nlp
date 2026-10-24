import json
from pathlib import Path
import time

import hydra
import torch
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
    Seq2SeqTrainingArguments, T5Config, T5ForConditionalGeneration, set_seed,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from execution_validation import ExecutionEvaluator, generate_candidates


class LengthAwareTrainer(Seq2SeqTrainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        was_training = self.model.training
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        config = getattr(self, "execution_validation", None)
        if not config or not config.enabled:
            return metrics
        started = time.perf_counter()
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        dataset = dataset.select(range(min(len(dataset), config.samples)))
        rows = [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"],
                 "reference": row["code"]} for row in dataset]
        references = [row["reference"] for row in rows]
        if getattr(self, "_execution_references", None) != references:
            self._execution_evaluator = ExecutionEvaluator(
                references, timeout=config.timeout, workers=config.workers, backend=config.backend,
            )
            self._execution_references = references
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        try:
            with self.compute_loss_context_manager():
                candidates = generate_candidates(model, self.processing_class, rows,
                                                 batch_size=config.batch_size,
                                                 max_new_tokens=config.max_new_tokens,
                                                 num_beams=config.get("num_beams", 1),
                                                 seed=self.args.seed)
        finally:
            model.train(was_training)
        result, details = self._execution_evaluator.evaluate(
            [[candidate["code"] for candidate in group] for group in candidates])
        execution_metrics = {f"{metric_key_prefix}_pass_at_1": result["pass_at_1"],
                             f"{metric_key_prefix}_pass_at_1_all": result["pass_at_1_all"],
                             f"{metric_key_prefix}_execution_samples": result["examples"],
                             f"{metric_key_prefix}_valid_references": result["valid_references"],
                             f"{metric_key_prefix}_invalid_references": result["invalid_references"],
                             f"{metric_key_prefix}_numeric_rate": result["prediction_numeric_rate"],
                             f"{metric_key_prefix}_num_beams": config.get("num_beams", 1),
                             f"{metric_key_prefix}_execution_seconds": time.perf_counter() - started}
        directory = Path(self.args.output_dir).parent / "execution_validation"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"step-{self.state.global_step}.json").write_text(
            json.dumps({"metrics": execution_metrics, "outcomes": details}, ensure_ascii=False) + "\n")
        self.log(execution_metrics)
        metrics.update(execution_metrics)
        return metrics

    def _get_train_sampler(self, train_dataset=None):
        # Trainer removes metadata columns before calling this method. Read the
        # prepared source+target lengths from the original dataset instead.
        if self.args.group_by_length:
            column = self.args.length_column_name
            lengths = self.train_dataset.select_columns([column]).with_format("numpy")[:][column]
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
            )
        return super()._get_train_sampler(train_dataset)


def load_optimizer(trainer, checkpoint):
    optimizer = trainer.create_optimizer()
    optimizer.load_state_dict(torch.load(
        Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=True,
    ))
    for group in optimizer.param_groups:
        group["lr"] = trainer.args.learning_rate
        group["initial_lr"] = trainer.args.learning_rate
    steps = [int(state["step"]) for state in optimizer.state.values()]
    if not steps:
        raise ValueError("Checkpoint has no optimizer state.")
    print(json.dumps({"optimizer_checkpoint": str(checkpoint), "optimizer_states": len(steps),
                      "optimizer_step_min": min(steps), "optimizer_step_max": max(steps),
                      "learning_rate": trainer.args.learning_rate}))


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.resume and cfg.init_checkpoint:
        raise ValueError("Use either resume (full state) or init_checkpoint (weights only).")
    if cfg.init_optimizer and not cfg.init_checkpoint:
        raise ValueError("init_optimizer requires init_checkpoint.")
    checkpoint_dir = Path(cfg.training.output_dir)
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not cfg.resume:
        raise FileExistsError("Checkpoint directory is not empty; set resume or use another run_dir.")
    set_seed(cfg.seed)
    prepared = Path(cfg.data_dir)
    data = load_from_disk(prepared / "dataset")
    tokenizer = AutoTokenizer.from_pretrained(prepared / "tokenizer")
    preparation = OmegaConf.load(prepared / "config.yaml")
    validation = data["validation"].filter(
        lambda n: n <= preparation.data.max_target_length, input_columns=["target_tokens"]
    )
    if cfg.data.eval_samples is not None:
        validation = validation.shuffle(seed=cfg.seed).select(range(min(len(validation), cfg.data.eval_samples)))
    if not len(validation):
        raise ValueError("No validation programs fit the target length limit.")

    config = T5Config(
        **OmegaConf.to_container(cfg.model, resolve=True),
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    if cfg.init_checkpoint:
        # Only a checkpoint trained in this project.
        model = T5ForConditionalGeneration.from_pretrained(
            cfg.init_checkpoint, config=config, local_files_only=True,
        )
    else:
        model = T5ForConditionalGeneration(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    args = Seq2SeqTrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    trainer = LengthAwareTrainer(
        model=model, args=args, train_dataset=data["train"], eval_dataset=validation,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8),
    )
    trainer.execution_validation = cfg.get("execution_validation")
    if cfg.init_optimizer:
        load_optimizer(trainer, cfg.init_checkpoint)
    result = trainer.train(resume_from_checkpoint=cfg.resume)
    trainer.save_model(str(Path(cfg.run_dir) / "model"))
    trainer.save_metrics("train", result.metrics)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(Path(cfg.run_dir) / "model")
        OmegaConf.save(cfg, Path(cfg.run_dir) / "config.yaml", resolve=True)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
