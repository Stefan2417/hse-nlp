import unittest
import tempfile
from pathlib import Path

import torch
from unittest.mock import patch

from datasets import Dataset
from omegaconf import OmegaConf
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, T5Config, T5ForConditionalGeneration

from evaluate import has_solve, parses
from prepare import encode_batch, is_validation, make_tokenizer
from sandbox_runner import equal
from train import LengthAwareTrainer, load_optimizer
from execution_validation import ExecutionEvaluator, vote


class PipelineTests(unittest.TestCase):
    def test_vote_and_oracle_do_not_use_reference_to_select(self):
        values = {"right": {"ok": True, "value": 9}, "wrong": {"ok": True, "value": 4.5},
                  "bad": {"ok": False, "error": "SyntaxError"}}
        self.assertEqual(vote([values["right"], values["right"], values["wrong"], values["bad"]],
                              [-2, -1, 0, 10]), 1)
        with patch("execution_validation.run_program", side_effect=lambda code, *args: values.get(
                code, {"ok": True, "value": 2})):
            evaluator = ExecutionEvaluator(["right", "bad"])
            metrics, details = evaluator.evaluate([["wrong", "right"], ["wrong", "wrong"]],
                                                  scores=[[0, -1], [0, -1]], voting=True)
        self.assertEqual(metrics["pass_at_1"], 0)
        self.assertEqual(metrics["oracle_at_k"], 1)
        self.assertEqual(metrics["valid_references"], 1)
        self.assertEqual(details[0]["selected"], 0)

    def test_execution_pass_at_one_is_logged_during_validation(self):
        codes = ["def solve():\n    return 2\n", "def solve():\n    return 3\n"]
        raw = Dataset.from_dict({"question": ["Two?", "Three?"], "code": codes})
        cfg = OmegaConf.create({"seed": 42, "tokenizer": {"vocab_size": 300,
                                 "min_frequency": 1, "train_samples": 2}})
        tokenizer = make_tokenizer(raw, cfg)
        data = Dataset.from_dict({**encode_batch(raw[:], tokenizer, 32), "code": codes})
        model = T5ForConditionalGeneration(T5Config(
            vocab_size=len(tokenizer), d_model=16, d_ff=32, d_kv=8, num_heads=2,
            num_layers=1, num_decoder_layers=1, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.pad_token_id))
        with tempfile.TemporaryDirectory() as directory:
            args = Seq2SeqTrainingArguments(output_dir=str(Path(directory) / "checkpoints"),
                                            use_cpu=True, report_to="none", prediction_loss_only=True)
            trainer = LengthAwareTrainer(model=model, args=args, eval_dataset=data,
                                         processing_class=tokenizer,
                                         data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))
            trainer.execution_validation = OmegaConf.create(dict(
                enabled=True, samples=2, timeout=2, workers=1, backend="auto", batch_size=2,
                max_new_tokens=8, num_beams=4))
            predictions = [[{"code": codes[0]}], [{"code": codes[0]}]]
            with patch("train.generate_candidates", return_value=predictions) as generate, patch(
                    "execution_validation.run_program", side_effect=lambda code, *args: {
                        "ok": True, "value": 3 if code == codes[1] else 2}):
                metrics = trainer.evaluate()
            self.assertEqual(metrics["eval_pass_at_1"], 0.5)
            self.assertEqual(generate.call_args.kwargs["num_beams"], 4)
            self.assertEqual(metrics["eval_num_beams"], 4)
            self.assertEqual(metrics["eval_valid_references"], 2)
            self.assertTrue(any(row.get("eval_pass_at_1") == 0.5 for row in trainer.state.log_history))
            self.assertTrue((Path(directory) / "execution_validation/step-0.json").exists())
            self.assertTrue(model.training)

    def test_optimizer_continuation_preserves_moments_and_resets_schedule(self):
        config = T5Config(vocab_size=16, d_model=16, d_ff=32, d_kv=8, num_heads=2,
                          num_layers=1, num_decoder_layers=1, pad_token_id=0,
                          eos_token_id=1, decoder_start_token_id=0)
        with tempfile.TemporaryDirectory() as directory:
            args = Seq2SeqTrainingArguments(output_dir=directory, use_cpu=True,
                                            report_to="none", learning_rate=1e-4)
            old = LengthAwareTrainer(model=T5ForConditionalGeneration(config), args=args)
            optimizer = old.create_optimizer()
            for parameter in old.model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            state = optimizer.state_dict()
            torch.save(state, Path(directory) / "optimizer.pt")
            new_args = Seq2SeqTrainingArguments(output_dir=directory, use_cpu=True,
                                                report_to="none", learning_rate=5e-5,
                                                warmup_ratio=0, lr_scheduler_type="cosine")
            new = LengthAwareTrainer(model=T5ForConditionalGeneration(config), args=new_args)
            load_optimizer(new, directory)
            restored = new.optimizer.state_dict()
            for index, saved in state["state"].items():
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(saved[name], restored["state"][index][name]))
            new.create_optimizer_and_scheduler(num_training_steps=100)
            self.assertEqual(new.lr_scheduler.last_epoch, 0)
            self.assertEqual(new.optimizer.param_groups[0]["lr"], 5e-5)
            self.assertAlmostEqual(new.lr_scheduler.lr_lambdas[0](50), 0.5)
            self.assertAlmostEqual(new.lr_scheduler.lr_lambdas[0](100), 0.0)

    def test_sampler_preserves_prepared_lengths_after_column_removal(self):
        # Source lengths alone would give a different order. The selected view
        # also checks that lengths follow dataset indices rather than Arrow rows.
        data = Dataset.from_dict({
            "input_ids": [[3, 1], [4, 5, 1], [6, 1]],
            "attention_mask": [[1, 1], [1, 1, 1], [1, 1]],
            "labels": [[7, 8, 9, 1], [7, 1], [8, 9, 1]],
            "length": [6, 5, 5],
        }).select([1, 0, 2])
        model = T5ForConditionalGeneration(T5Config(
            vocab_size=16, d_model=16, d_ff=32, d_kv=8, num_heads=2,
            num_layers=1, num_decoder_layers=1, pad_token_id=0,
            eos_token_id=1, decoder_start_token_id=0,
        ))
        with tempfile.TemporaryDirectory() as directory:
            args = Seq2SeqTrainingArguments(
                output_dir=directory, use_cpu=True, report_to="none",
                group_by_length=True, per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
            )
            trainer = LengthAwareTrainer(model=model, args=args, train_dataset=data)
            stripped = trainer._remove_unused_columns(data)
            self.assertNotIn("length", stripped.column_names)
            sampler = trainer._get_train_sampler(stripped)
            self.assertEqual(list(sampler.lengths), [5, 6, 5])
            self.assertEqual(sampler.batch_size, 4)
            self.assertEqual(sorted(sampler), list(range(len(data))))

    def test_question_copies_share_split(self):
        for seed in range(10):
            self.assertEqual(is_validation("How many apples?", 0.5, seed),
                             is_validation(" How  many\napples? ", 0.5, seed))

    def test_code_roundtrip_eos_and_label_padding(self):
        # A tokenizer/collator fixture, never used to train a solution model.
        code = "def solve():\n    x = 1.25\n    return x + 2\n"
        data = Dataset.from_dict({"question": ["What is the sum?", "Sum?"],
                                  "code": [code, "def solve():\n    return 3\n"]})
        cfg = OmegaConf.create({"seed": 42, "tokenizer": {"vocab_size": 300,
                                  "min_frequency": 1, "train_samples": 2}})
        tokenizer = make_tokenizer(data, cfg)
        encoded = encode_batch(data[:], tokenizer, max_source_length=4)
        self.assertEqual(tokenizer.decode(encoded["labels"][0], skip_special_tokens=True), code)
        self.assertEqual(encoded["input_ids"][0][-1], tokenizer.eos_token_id)
        self.assertEqual(encoded["labels"][0][-1], tokenizer.eos_token_id)
        self.assertLessEqual(len(encoded["input_ids"][0]), 4)
        model = T5ForConditionalGeneration(T5Config(
            vocab_size=len(tokenizer), d_model=16, d_ff=32, d_kv=8, num_heads=2,
            num_layers=1, num_decoder_layers=1, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.pad_token_id,
        ))
        features = [{key: encoded[key][i] for key in ("input_ids", "attention_mask", "labels")}
                    for i in range(2)]
        batch = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)(features)
        self.assertTrue((batch["labels"] == -100).any())
        self.assertTrue((batch["decoder_input_ids"][:, 0] == tokenizer.pad_token_id).all())
        self.assertTrue(model(**batch).loss.isfinite())

    def test_checks_and_comparison(self):
        self.assertFalse(parses("return 1"))
        self.assertFalse(parses("def solve(:"))
        self.assertTrue(has_solve("def solve():\n    return 1\n"))
        self.assertFalse(has_solve("x = 1"))
        self.assertTrue(equal([1, 2.000001], [1.0, 2], 1e-4, 1e-6))
        self.assertFalse(equal([1, 2], [1], 1e-4, 1e-6))
        self.assertFalse(equal("1", 1, 1e-4, 1e-6))


if __name__ == "__main__":
    unittest.main()
