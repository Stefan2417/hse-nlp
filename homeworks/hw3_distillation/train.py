from pathlib import Path

import torch
import wandb
from torch.nn import functional as F
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from .data import IGNORE_LABEL, compute_metrics
from .models import BertForNER


RUNS_DIR = Path(__file__).parent / 'runs'


def relations(hidden, mask, heads):
    batch, length, _ = hidden.shape
    x = hidden.reshape(batch, length, heads, -1).transpose(1, 2)
    scores = (x @ x.transpose(-1, -2)).float() / x.shape[-1] ** 0.5
    return scores.masked_fill(~mask[:, None, None, :], -1e4)


class DistillationTrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, relation_heads=0, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher.to(self.args.device).eval().requires_grad_(False)
        self.temperature = temperature
        self.projection = getattr(self.model, 'hidden_projection', None)
        self.relation_heads = relation_heads
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        need_states = self.projection is not None or bool(self.relation_heads)
        student = model(**inputs, output_hidden_states=need_states)
        teacher_inputs = dict(inputs)
        labels = teacher_inputs.pop('labels')
        with torch.no_grad():
            teacher = self.teacher(**teacher_inputs, output_hidden_states=need_states)

        mask = labels != IGNORE_LABEL
        t = self.temperature
        log_probs = F.log_softmax(student.logits[mask].float() / t, dim=-1)
        teacher_probs = F.softmax(teacher.logits[mask].float() / t, dim=-1)
        kl = F.kl_div(log_probs, teacher_probs, reduction='batchmean')
        loss = student.loss + t ** 2 * kl
        if self.projection is not None:
            layers = student.hidden_states[1:]
            teacher_depth = len(teacher.hidden_states) - 1
            teacher_ids = [round(i * teacher_depth / len(layers)) for i in range(1, len(layers) + 1)]
            student_states = torch.stack([layer[mask] for layer in layers])
            teacher_states = torch.stack([teacher.hidden_states[i][mask] for i in teacher_ids])
            loss += F.mse_loss(self.projection(student_states).float(), teacher_states.float())
        if self.relation_heads:
            mask = inputs['attention_mask'].bool()
            loss += self.minilm_loss(
                model,
                student.hidden_states[-2],
                teacher.hidden_states[-2],
                mask
            )
        return (loss, {'logits': student.logits}) if return_outputs else loss

    def minilm_loss(self, model, student, teacher, mask):
        model = self.accelerator.unwrap_model(model)
        student_attn = model.bert.encoder.layer[-1].attention.self
        teacher_attn = self.teacher.bert.encoder.layer[-1].attention.self
        losses = []
        for name in ('query', 'key', 'value'):
            qkv = getattr(student_attn, name)(student)
            scores = relations(qkv, mask, self.relation_heads)
            with torch.no_grad():
                qkv = getattr(teacher_attn, name)(teacher)
                targets = relations(qkv, mask, self.relation_heads).softmax(dim=-1)
            kl = F.kl_div(
                scores.log_softmax(dim=-1),
                targets,
                reduction='none'
            ).sum(dim=-1)
            losses.append(kl.transpose(1, 2)[mask].mean())
        return torch.stack(losses).sum()


def make_trainer(
    model,
    dataset,
    tokenizer,
    name,
    teacher_checkpoint=None,
    temperature=2.0,
    relation_heads=0,
    callbacks=None,
    **kwargs
):
    train_args = dict(
        output_dir=str(RUNS_DIR / name),
        run_name=name,
        num_train_epochs=4,
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    train_args.update(kwargs)
    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(report_to='wandb', **train_args),
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=DataCollatorForTokenClassification(
            tokenizer,
            label_pad_token_id=IGNORE_LABEL
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1),
        callbacks=callbacks
    )
    if teacher_checkpoint is None:
        return Trainer(**trainer_kwargs)
    model.config.temperature = temperature
    if relation_heads:
        model.config.relation_heads = relation_heads
    teacher = BertForNER.from_pretrained(teacher_checkpoint)
    return DistillationTrainer(
        teacher,
        temperature=temperature,
        relation_heads=relation_heads,
        **trainer_kwargs
    )


def train(
    model,
    dataset,
    tokenizer,
    name,
    teacher_checkpoint=None,
    resume_from_checkpoint=None,
    relation_heads=0,
    **kwargs
):
    trainer = make_trainer(
        model,
        dataset,
        tokenizer,
        name,
        teacher_checkpoint=teacher_checkpoint,
        relation_heads=relation_heads,
        **kwargs
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(Path(trainer.args.output_dir) / 'best'))
    trainer.save_state()
    metrics = trainer.evaluate(dataset['test'], metric_key_prefix='test')
    result = dict(
        model=name, params=sum(p.numel() for p in model.parameters()),
        val_f1=trainer.state.best_metric,
        test_f1=metrics['test_f1']
    )
    metrics.update(result)
    trainer.save_metrics('test', metrics)
    wandb.finish()
    return result
