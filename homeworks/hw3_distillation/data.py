from datasets import load_dataset
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


BERT_NAME = 'bert-base-cased'
IGNORE_LABEL = -100
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


def load_data():
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    dataset = load_dataset('ashaba1in/conll2003')

    def tokenize(batch):
        inputs = tokenizer(batch['tokens'], is_split_into_words=True, truncation=True)
        inputs['labels'] = []
        for i, tags in enumerate(batch['ner_tags']):
            labels = []
            prev_id = None
            for word_id in inputs.word_ids(i):
                if word_id is None:
                    labels.append(IGNORE_LABEL)
                    continue
                label = tags[word_id]
                if word_id == prev_id and label_names[label].startswith('B-'):
                    label = label_names.index('I-' + label_names[label][2:])
                labels.append(label)
                prev_id = word_id
            inputs['labels'].append(labels)
        return inputs

    columns = dataset['train'].column_names
    return dataset.map(tokenize, batched=True, remove_columns=columns), tokenizer


def calibration_batches(dataset, collator, samples):
    train = dataset['train'].shuffle(seed=42)
    rows = train.select(range(min(samples, len(train)))).remove_columns('labels')
    return DataLoader(rows, batch_size=16, collate_fn=collator)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    true_tags, pred_tags = [], []
    for pred, target in zip(preds, labels):
        mask = target != IGNORE_LABEL
        true_tags.append([label_names[i] for i in target[mask]])
        pred_tags.append([label_names[i] for i in pred[mask]])
    return {'f1': f1_score(true_tags, pred_tags, zero_division=0)}
