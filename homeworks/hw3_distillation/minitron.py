import torch
from torch.nn import functional as F
from transformers import DataCollatorForTokenClassification

from .data import calibration_batches
from .models import BertForNER, make_student


def top_ids(scores, count):
    return scores.topk(count).indices.sort().values


def copy_linear(new, old, rows, cols):
    new.weight.copy_(old.weight[rows][:, cols])
    new.bias.copy_(old.bias[rows])


def copy_norm(new, old, ids):
    new.weight.copy_(old.weight[ids])
    new.bias.copy_(old.bias[ids])


@torch.no_grad()
def pruned_student(checkpoint, dataset, tokenizer, samples=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = BertForNER.from_pretrained(checkpoint).to(device).eval()
    student = make_student(tokenizer).to(device)
    source = teacher.config
    target = student.config
    layers = teacher.bert.encoder.layer
    layer_scores = torch.zeros(len(layers), device=device)
    hidden_scores = torch.zeros(source.hidden_size, device=device)
    head_scores = torch.zeros(len(layers), source.num_attention_heads, device=device)
    neuron_scores = torch.zeros(len(layers), source.intermediate_size, device=device)
    linear_scores = {}
    for i, layer in enumerate(layers):
        linear_scores[layer.attention.output.dense] = head_scores[i]
        linear_scores[layer.output.dense] = neuron_scores[i]

    def collect(layer, inputs):
        scores = linear_scores[layer]
        x = inputs[0][mask].float()
        x = x.reshape(x.shape[0], scores.numel(), -1)
        scores.add_(x.square().sum(dim=(0, 2)))

    batches = calibration_batches(
        dataset,
        DataCollatorForTokenClassification(tokenizer),
        samples
    )
    hooks = [layer.register_forward_pre_hook(collect) for layer in linear_scores]
    for batch in batches:
        batch = batch.to(device)
        mask = batch['attention_mask'].bool()
        states = teacher(**batch, output_hidden_states=True).hidden_states
        states = torch.stack([layer[mask].float() for layer in states])
        hidden_scores += states.square().sum(dim=(0, 1))
        layer_scores += (1 - F.cosine_similarity(states[:-1], states[1:], dim=-1)).sum(dim=1)

    for hook in hooks:
        hook.remove()

    layer_ids = top_ids(layer_scores, target.num_hidden_layers).tolist()
    hidden = top_ids(hidden_scores, target.hidden_size)
    head_size = source.hidden_size // source.num_attention_heads
    offsets = torch.arange(head_size, device=device)
    old = teacher.bert.embeddings
    new = student.bert.embeddings
    new.word_embeddings.init_svd(old.word_embeddings.weight[:, hidden])
    new.position_embeddings.weight.copy_(old.position_embeddings.weight[:, hidden])
    new.token_type_embeddings.weight.copy_(old.token_type_embeddings.weight[:, hidden])
    copy_norm(new.LayerNorm, old.LayerNorm, hidden)

    for new, i in zip(student.bert.encoder.layer, layer_ids):
        old = layers[i]
        heads = top_ids(head_scores[i], target.num_attention_heads)
        head_ids = (heads[:, None] * head_size + offsets).reshape(-1)
        neurons = top_ids(neuron_scores[i], target.intermediate_size)
        for name in ('query', 'key', 'value'):
            copy_linear(
                getattr(new.attention.self, name),
                getattr(old.attention.self, name),
                head_ids,
                hidden
            )

        copy_linear(
            new.attention.output.dense,
            old.attention.output.dense,
            hidden,
            head_ids
        )
        copy_norm(new.attention.output.LayerNorm, old.attention.output.LayerNorm, hidden)
        copy_linear(new.intermediate.dense, old.intermediate.dense, neurons, hidden)
        copy_linear(new.output.dense, old.output.dense, hidden, neurons)
        copy_norm(new.output.LayerNorm, old.output.LayerNorm, hidden)

    labels = torch.arange(target.num_labels, device=device)
    copy_linear(student.classifier, teacher.classifier, labels, hidden)
    student.config.teacher_layers = [i + 1 for i in layer_ids]
    return student
