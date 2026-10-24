"""Executed inside a disposable Docker container by evaluate.py."""
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


PROGRAM = r'''
import contextlib
import json
import math
import resource
import sys

if sys.platform != "darwin":
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
cpu_seconds = max(1, math.ceil(float(sys.argv[1])))
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
source = sys.stdin.read()
namespace = {}
try:
    with open("/dev/null", "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(source, "<program>", "exec"), namespace)
        answer = namespace["solve"]()
    encoded = json.dumps(answer, allow_nan=False)
    if len(encoded) > 4096:
        raise ValueError("Result is too large")
    print(json.dumps({"ok": True, "value": json.loads(encoded)}))
except BaseException as error:
    print(json.dumps({"ok": False, "error": type(error).__name__}))
'''


def sandbox_command(timeout, backend):
    if backend == "auto":
        backend = "macos" if platform.system() == "Darwin" else "bwrap"
    if backend == "macos":
        python = str(Path(sys._base_executable).resolve())
        profile = ('(version 1)(allow default)(deny network*)(deny process-fork)'
                   '(deny file-write* (require-not (literal "/dev/null")))'
                   '(deny file-read* (subpath "/Users") (subpath "/private/tmp")'
                   ' (subpath "/private/var/folders"))')
        return ["sandbox-exec", "-p", profile, python, "-I", "-c", PROGRAM, str(timeout)]
    if backend == "bwrap":
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError("Execution validation requires bubblewrap on Linux.")
        command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
        for directory in ("/usr", "/lib", "/lib64", sys.prefix):
            if Path(directory).exists():
                command += ["--ro-bind", directory, directory]
        return command + ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
                          sys.executable, "-I", "-c", PROGRAM, str(timeout)]
    if backend is None:
        return [sys.executable, "-I", "-c", PROGRAM, str(timeout)]
    raise ValueError(f"Unknown execution backend: {backend}")


def run_program(code, timeout, backend=None):
    try:
        command = sandbox_command(timeout, backend)
        environment = {"PATH": os.defpath, "HOME": "/nonexistent", "LC_ALL": "C"}
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=environment,
                              start_new_session=True) as process:
            try:
                if platform.system() == "Darwin":
                    import psutil
                    child = psutil.Process(process.pid)
                    deadline = time.monotonic() + timeout
                    pending_input = code
                    while True:
                        try:
                            stdout, stderr = process.communicate(pending_input, timeout=0.02)
                            break
                        except subprocess.TimeoutExpired:
                            pending_input = None
                            if time.monotonic() >= deadline:
                                raise
                            try:
                                memory = child.memory_info().rss
                            except psutil.NoSuchProcess:
                                continue
                            if memory > 256 * 1024**2:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.communicate()
                                return {"ok": False, "error": "memory_limit"}
                else:
                    stdout, stderr = process.communicate(code, timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return {"ok": False, "error": "timeout"}
            if process.returncode:
                return {"ok": False, "error": f"exit_{process.returncode}", "stderr": stderr[-500:]}
            return json.loads(stdout)
    except (json.JSONDecodeError, UnicodeError):
        return {"ok": False, "error": "invalid_output"}


def equal(left, right, rtol, atol):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return abs(left - right) <= atol + rtol * abs(right)
        except (OverflowError, ValueError):
            return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal(x, y, rtol, atol) for x, y in zip(left, right))
    return type(left) is type(right) and left == right


def main():
    timeout, rtol, atol = map(float, sys.argv[1:])
    outcomes = []
    for row in json.load(sys.stdin):
        prediction = run_program(row["code"], timeout)
        item = {"id": row["id"], "prediction_ok": prediction["ok"], "prediction": prediction}
        if "reference" in row:
            reference = run_program(row["reference"], timeout)
            item.update(reference_ok=reference["ok"], reference=reference,
                        match=prediction["ok"] and reference["ok"]
                        and equal(prediction["value"], reference["value"], rtol, atol))
        outcomes.append(item)
    print(json.dumps(outcomes))


if __name__ == "__main__":
    main()
