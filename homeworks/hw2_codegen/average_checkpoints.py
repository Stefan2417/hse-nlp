import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / "result.json").exists(), "Already completed"
    first, second = args.first.resolve(), args.second.resolve()
    source_hashes = [sha256(p / "model.safetensors") for p in (first, second)]
    questions = args.questions.resolve()
    questions_hash = sha256(questions)
    assert questions_hash == "a3a128e5acf694f9b9a828dcc582bacb79367eecf11af2f6041c0c5f788e7003"
    rows = [json.loads(line) for line in questions.read_text().splitlines()]
    assert len(rows) == 1919 and all(isinstance(row["question"], str) for row in rows)
    bundle = run / "kaggle_hw2"
    model = bundle / "model"
    model.mkdir(parents=True, exist_ok=True)
    tensors = {}
    different = 0
    with safe_open(first / "model.safetensors", framework="np") as a, \
            safe_open(second / "model.safetensors", framework="np") as b:
        assert set(a.keys()) == set(b.keys()) and len(a.keys()) == 121
        for name in a.keys():
            left, right = a.get_tensor(name), b.get_tensor(name)
            assert left.shape == right.shape and left.dtype == right.dtype == np.float32
            assert np.isfinite(left).all() and np.isfinite(right).all()
            different += not np.array_equal(left, right)
            tensors[name] = ((left + right) * np.float32(0.5)).astype(np.float16)
            assert np.isfinite(tensors[name]).all(), name
    assert different > 0
    save_file(tensors, model / "model.safetensors", metadata={"format": "pt"})
    del tensors
    names = ("config.json", "generation_config.json", "special_tokens_map.json",
             "tokenizer.json", "tokenizer_config.json")
    for name in names:
        shutil.copyfile(first / name, model / name)
    config = json.loads((model / "config.json").read_text())
    for name in ("dtype", "torch_dtype"):
        if name in config:
            config[name] = "float16"
    (model / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(questions, bundle / "test_questions.jsonl")
    with safe_open(model / "model.safetensors", framework="np") as weights:
        assert len(weights.keys()) == 121
        for name in weights.keys():
            tensor = weights.get_tensor(name)
            assert tensor.dtype == np.float16 and np.isfinite(tensor).all()
    archive_path = run / "kaggle_model.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    expected_members = {"kaggle_hw2/model/model.safetensors", "kaggle_hw2/test_questions.jsonl"}
    expected_members.update("kaggle_hw2/model/" + name for name in names)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run))
    with zipfile.ZipFile(temporary) as archive:
        members = archive.namelist()
        assert len(members) == 7 and set(members) == expected_members
        assert archive.testzip() is None
    temporary.replace(archive_path)
    result = {"task_exit": 0,
              "checkpoints": [str(first), str(second)], "source_sha256": source_hashes,
              "weights": [0.5, 0.5], "averaging_dtype": "float32", "export_dtype": "float16",
              "different_tensors": different, "tensors": 121, "questions": 1919,
              "questions_sha256": questions_hash,
              "model_sha256": sha256(model / "model.safetensors"),
              "archive": str(archive_path), "archive_bytes": archive_path.stat().st_size,
              "archive_sha256": sha256(archive_path), "members": members,
              "archive_integrity": True, "validation_run": False, "generation_run": False}
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
