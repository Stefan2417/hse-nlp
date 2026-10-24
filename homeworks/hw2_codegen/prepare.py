import hashlib
import json
from pathlib import Path

import hydra
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


def is_validation(question, fraction, seed):
    # All copies of a question stay together, even if their programs differ.
    question = " ".join(question.split())
    digest = hashlib.blake2b(f"{seed}:{question}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < fraction


def make_tokenizer(dataset, cfg):
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.tokenizer.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=["<pad>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    sample = dataset.shuffle(seed=cfg.seed).select(
        range(min(len(dataset), cfg.tokenizer.train_samples))
    )
    texts = (text for row in sample for text in (row["question"], row["code"]))
    backend.train_from_iterator(texts, trainer=trainer, length=2 * len(sample))
    backend.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", backend.token_to_id("</s>"))]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>",
        unk_token="<unk>", clean_up_tokenization_spaces=False,
    )


def encode_batch(batch, tokenizer, max_source_length):
    inputs = tokenizer(batch["question"], truncation=False)
    source_lengths = [len(ids) for ids in inputs["input_ids"]]
    # Keep EOS when shortening a source. Never truncate a target program.
    inputs["input_ids"] = [
        ids if len(ids) <= max_source_length
        else ids[:max_source_length - 1] + [tokenizer.eos_token_id]
        for ids in inputs["input_ids"]
    ]
    inputs["attention_mask"] = [[1] * len(ids) for ids in inputs["input_ids"]]
    inputs.pop("token_type_ids", None)
    inputs["source_tokens"] = source_lengths
    if "code" in batch:
        inputs["labels"] = tokenizer(batch["code"], truncation=False)["input_ids"]
        inputs["target_tokens"] = [len(ids) for ids in inputs["labels"]]
        inputs["length"] = [len(x) + len(y) for x, y in zip(inputs["input_ids"], inputs["labels"])]
    return inputs


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    output = Path(cfg.data_dir)
    if (output / "dataset").exists():
        raise FileExistsError(f"{output}/dataset exists; choose a new data_dir.")
    if not 0 < cfg.data.validation_fraction < 1 or cfg.data.max_source_length < 2:
        raise ValueError("Use 0 < validation_fraction < 1 and max_source_length >= 2.")

    if cfg.data.train_file:
        files = {"train": cfg.data.train_file}
        if cfg.data.test_file:
            files["test"] = cfg.data.test_file
        loader = "parquet" if str(cfg.data.train_file).endswith(".parquet") else "json"
        raw = load_dataset(loader, data_files=files)
    else:
        raw = load_dataset(cfg.data.name, revision=cfg.data.revision)
    num_proc = cfg.data.num_proc if cfg.data.num_proc and cfg.data.num_proc > 1 else None
    train = raw["train"]
    if cfg.data.max_train_samples is not None:
        train = train.shuffle(seed=cfg.seed).select(range(min(len(train), cfg.data.max_train_samples)))
    fingerprints = {name: split._fingerprint for name, split in raw.items()}
    train = train.map(
        lambda batch: {"validation": [is_validation(q, cfg.data.validation_fraction, cfg.seed)
                                      for q in batch["question"]]},
        batched=True, num_proc=num_proc,
    )
    splits = DatasetDict(
        train=train.filter(lambda value: not value, input_columns=["validation"], num_proc=num_proc),
        validation=train.filter(bool, input_columns=["validation"], num_proc=num_proc),
    ).remove_columns("validation")
    if not len(splits["train"]) or not len(splits["validation"]):
        raise ValueError("Empty train or validation split; increase sample size or validation_fraction.")
    tokenizer = make_tokenizer(splits["train"], cfg)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output / "tokenizer")
    if "test" in raw:
        # JSON/Parquet loaders may add a nullable code column to the test split.
        splits["test"] = raw["test"].select_columns(["question"])

    stats = {"vocab_size": len(tokenizer), "raw_fingerprints": fingerprints, "splits": {}}
    for name, split in splits.items():
        encoded = split.map(
            encode_batch, batched=True, num_proc=num_proc,
            fn_kwargs={"tokenizer": tokenizer, "max_source_length": cfg.data.max_source_length},
            remove_columns=split.column_names if name == "train" else None,
        )
        info = {"rows": len(encoded), "source_tokens": 0, "target_tokens": 0, "source_truncated": 0,
                "targets_over_limit": 0}
        columns = ["source_tokens"] + (["target_tokens"] if "code" in split.column_names else [])
        for batch in encoded.select_columns(columns).iter(batch_size=10000):
            info["source_tokens"] += sum(batch["source_tokens"])
            info["source_truncated"] += sum(n > cfg.data.max_source_length for n in batch["source_tokens"])
            targets = batch.get("target_tokens", [])
            info["target_tokens"] += sum(targets)
            info["targets_over_limit"] += sum(n > cfg.data.max_target_length for n in targets)
        if name == "train":
            encoded = encoded.filter(lambda n: n <= cfg.data.max_target_length,
                                     input_columns=["target_tokens"], num_proc=num_proc)
            if not len(encoded):
                raise ValueError("All training programs exceed max_target_length.")
        info["saved_rows"] = len(encoded)
        stats["splits"][name] = info
        splits[name] = encoded
    splits.save_to_disk(output / "dataset")
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
