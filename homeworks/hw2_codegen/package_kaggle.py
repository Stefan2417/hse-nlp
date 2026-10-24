import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, T5ForConditionalGeneration


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    assert metadata["task_exit"] == 0
    assert metadata["source_sha256"] == sha256(args.source / "model.safetensors")
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    assert len(questions) == 1919 and all(isinstance(row["question"], str) for row in questions)
    questions_hash = sha256(args.questions)
    assert questions_hash == "a3a128e5acf694f9b9a828dcc582bacb79367eecf11af2f6041c0c5f788e7003"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    verification_path = args.output.with_suffix(".verification.json")
    if args.output.exists():
        saved = json.loads(verification_path.read_text())
        assert saved["source_sha256"] == metadata["source_sha256"]
        assert saved["archive_sha256"] == sha256(args.output)
        with zipfile.ZipFile(args.output) as archive:
            assert archive.testzip() is None and archive.namelist() == saved["members"]
        print(json.dumps(saved, indent=2))
        return

    torch.set_num_threads(4)
    model = T5ForConditionalGeneration.from_pretrained(
        args.source, torch_dtype=torch.float16, local_files_only=True,
    ).half().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    expected = {f"kaggle_hw2/model/{name}" for name in (
        "config.json", "generation_config.json", "model.safetensors",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    )} | {"kaggle_hw2/test_questions.jsonl"}
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".kaggle-export-") as directory:
        stage = Path(directory)
        bundle = stage / "kaggle_hw2"
        model.save_pretrained(bundle / "model", safe_serialization=True)
        tokenizer.save_pretrained(bundle / "model")
        shutil.copyfile(args.questions, bundle / "test_questions.jsonl")
        with safe_open(bundle / "model/model.safetensors", framework="pt", device="cpu") as weights:
            keys = list(weights.keys())
            assert len(keys) == 121
            for key in keys:
                tensor = weights.get_tensor(key)
                assert tensor.dtype == torch.float16 and torch.isfinite(tensor).all(), key
        reloaded = T5ForConditionalGeneration.from_pretrained(
            bundle / "model", torch_dtype=torch.float32, local_files_only=True,
        ).eval()
        batch = tokenizer([questions[0]["question"]], truncation=True, max_length=256,
                          return_tensors="pt")
        batch = {key: value for key, value in batch.items() if key in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            output = reloaded(**batch, decoder_input_ids=torch.tensor([[tokenizer.pad_token_id]]))
        assert torch.isfinite(output.logits).all()
        temporary_archive = stage / args.output.name
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage))
        with zipfile.ZipFile(temporary_archive) as archive:
            members = archive.namelist()
            assert len(members) == 7 and set(members) == expected
            assert archive.testzip() is None
        verification = {**metadata, "created_at": datetime.now(timezone.utc).isoformat(),
                        "export_dtype": "float16", "tensors": len(keys), "questions": len(questions),
                        "questions_sha256": questions_hash, "archive": str(args.output.resolve()),
                        "archive_bytes": temporary_archive.stat().st_size,
                        "archive_sha256": sha256(temporary_archive), "members": members,
                        "archive_integrity": True, "cpu_reload_forward_finite": True,
                        "t4_runtime_verified": False}
        temporary_archive.replace(args.output)
        verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
