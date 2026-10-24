import ast
import json
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig


def parses(code):
    try:
        compile(code, "<prediction>", "exec")
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def has_solve(code):
    try:
        return any(isinstance(node, ast.FunctionDef) and node.name == "solve"
                   for node in ast.parse(code).body)
    except (SyntaxError, ValueError, TypeError):
        return False


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    path = Path(cfg.evaluation.predictions)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No predictions to evaluate.")
    references = [row for row in rows if "reference" in row]
    metrics = {"examples": len(rows),
               "syntax_valid_rate": sum(parses(row["code"]) for row in rows) / len(rows),
               "solve_defined_rate": sum(has_solve(row["code"]) for row in rows) / len(rows)}
    if references:
        metrics["text_exact_match"] = sum(row["code"].strip() == row["reference"].strip()
                                           for row in references) / len(references)
    if cfg.evaluation.execute:
        runner = Path(__file__).with_name("sandbox_runner.py").resolve()
        command = [
            "docker", "run", "--rm", "-i", "--network=none", "--read-only",
            "--memory=512m", "--cpus=1", "--pids-limit=64", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--user=65534:65534",
            "--mount", f"type=bind,src={runner},dst=/runner.py,readonly",
            cfg.evaluation.docker_image, "python", "-I", "/runner.py",
            str(cfg.evaluation.timeout), str(cfg.evaluation.rtol), str(cfg.evaluation.atol),
        ]
        try:
            result = subprocess.run(command, input=json.dumps(rows), text=True, capture_output=True,
                                    timeout=len(rows) * (2 * cfg.evaluation.timeout + 2) + 60)
        except FileNotFoundError as error:
            raise RuntimeError("Docker is required for execution; use evaluation.execute=false for syntax checks.") from error
        if result.returncode:
            raise RuntimeError(f"Docker evaluation failed:\n{result.stderr}")
        outcomes = json.loads(result.stdout)
        path.with_suffix(".execution.json").write_text(json.dumps(outcomes, indent=2) + "\n")
        valid_references = [row for row in outcomes if row.get("reference_ok")]
        metrics["execution_success_rate"] = sum(row["prediction_ok"] for row in outcomes) / len(outcomes)
        metrics["valid_references"] = len(valid_references)
        if references:
            metrics["invalid_references"] = len(references) - len(valid_references)
            metrics["reference_agreement"] = (
                sum(row["match"] for row in valid_references) / len(valid_references)
                if valid_references else None
            )
    path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
