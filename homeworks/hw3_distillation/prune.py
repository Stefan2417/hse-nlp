import torch
import wandb
from torch import nn
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from .data import IGNORE_LABEL, calibration_batches, compute_metrics
from .models import BertForNER
from .train import RUNS_DIR


def wanda(weight, gram, sparsity):
    scores = weight.abs() * gram.diagonal().sqrt()
    cols = scores.topk(int(weight.shape[1] * sparsity), largest=False, dim=1).indices
    return weight.scatter_(1, cols, 0)


def sparsegpt(weight, gram, sparsity, block_size=128):
    diag = gram.diagonal()
    dead = diag == 0
    weight[:, dead] = 0
    diag[dead] = 1
    diag.add_(0.01 * diag.mean())
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(gram))
    chol = torch.linalg.cholesky(inverse, upper=True)

    for start in range(0, weight.shape[1], block_size):
        end = min(start + block_size, weight.shape[1])
        block = weight[:, start:end]
        scores = (block / chol.diagonal()[start:end]).square().reshape(-1)
        ids = scores.topk(int(block.numel() * sparsity), largest=False).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[ids] = True
        mask = mask.reshape(block.shape)
        errors = torch.zeros_like(block)

        for i in range(end - start):
            col = start + i
            value = block[:, i].masked_fill(mask[:, i], 0)
            error = (block[:, i] - value) / chol[col, col]
            errors[:, i] = error
            block[:, i] = value
            block[:, i + 1:] -= error[:, None] * chol[col, col + 1:end]

        weight[:, end:] -= errors @ chol[start:end, end:]

    return weight


@torch.no_grad()
def prune(checkpoint, dataset, tokenizer, method, sparsity=0.5, samples=256):
    name = method.__name__
    args = TrainingArguments(
        output_dir=str(RUNS_DIR / name),
        run_name=name,
        per_device_eval_batch_size=64,
        use_cpu=not torch.cuda.is_available(),
        report_to='wandb',
        disable_tqdm=True
    )
    model = BertForNER.from_pretrained(checkpoint).to(args.device).eval()
    collator = DataCollatorForTokenClassification(
        tokenizer,
        label_pad_token_id=IGNORE_LABEL
    )
    batches = calibration_batches(dataset, collator, samples)

    for block in model.bert.encoder.layer:
        layers = [layer for layer in block.modules() if isinstance(layer, nn.Linear)]
        grams = {
            layer: torch.zeros(layer.in_features, layer.in_features, device=args.device)
            for layer in layers
        }

        def collect(layer, inputs):
            x = inputs[0][mask].float()
            grams[layer].addmm_(x.T, x)

        hooks = [layer.register_forward_pre_hook(collect) for layer in layers]
        for batch in batches:
            batch = batch.to(args.device)
            mask = batch['attention_mask'].bool()
            model(**batch)
        for hook in hooks:
            hook.remove()
        for layer in layers:
            layer.weight.copy_(method(layer.weight.float().clone(), grams[layer], sparsity))

    wandb.init(
        name=name,
        config=dict(sparsity=sparsity, samples=len(batches.dataset))
    )
    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1)
    )
    val = trainer.evaluate(dataset['validation'])
    test = trainer.evaluate(dataset['test'], metric_key_prefix='test')
    result = dict(
        model=name, params=sum(p.numel() for p in model.parameters()),
        nonzero=sum(p.count_nonzero().item() for p in model.parameters()),
        val_f1=val['eval_f1'],
        test_f1=test['test_f1']
    )
    trainer.save_model(str(RUNS_DIR / name / 'best'))
    trainer.save_metrics('test', result)
    wandb.log(result)
    wandb.finish()
    return result
