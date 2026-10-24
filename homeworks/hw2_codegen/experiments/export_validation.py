import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validation_1024.jsonl"))
    args = parser.parse_args()
    config = OmegaConf.load(args.data / "config.yaml")
    data = load_from_disk(args.data / "dataset")["validation"]
    data = data.filter(lambda n: n <= config.data.max_target_length,
                       input_columns=["target_tokens"]).shuffle(seed=config.seed)
    data = data.select(range(min(len(data), config.data.eval_samples)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for index, row in enumerate(data):
            stream.write(json.dumps({"id": index, "question": row["question"],
                                     "reference": row["code"], "input_ids": row["input_ids"],
                                     "attention_mask": row["attention_mask"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
