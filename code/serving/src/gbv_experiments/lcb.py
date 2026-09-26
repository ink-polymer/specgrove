"""Adapter to the pinned, unmodified official LiveCodeBench single-task judge."""
from __future__ import annotations

import json
import math

from .common import ROOT


UPSTREAM_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

WORKER = '''import ctypes, importlib.util, json, os, resource, signal, sys
from pathlib import Path
cpu_seconds = int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
if sys.platform == "linux":
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
if hasattr(resource, "RLIMIT_NPROC"):
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
spec = importlib.util.spec_from_file_location("lcb_official", "testing_util.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sample = json.loads(Path("sample.json").read_text())
candidate = Path("candidate.py").read_text()
expected_count = len(json.loads(sample["input_output"])["inputs"])
result_path = Path("result.json")
if "GBV_EVAL_UID" in os.environ:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    os.setgroups([])
    os.setgid(int(os.environ["GBV_EVAL_GID"]))
    os.setuid(int(os.environ["GBV_EVAL_UID"]))
    if os.geteuid() == 0:
        raise RuntimeError("process evaluator refused to execute candidate code as root")
try:
    checks, metadata = module.run_test(sample, candidate, timeout=int(sys.argv[1]))
    passed = len(checks) == expected_count and all(value > 0 for value in checks)
    result = {"passed": bool(passed), "reason": "passed" if passed else metadata.get("error_message", "failed"),
              "test_count": expected_count, "executed_tests": len(checks),
              "test_results": [int(value) for value in checks]}
except BaseException as exc:
    result = {"passed": False, "reason": type(exc).__name__, "test_count": expected_count}
finally:
    signal.alarm(0)
result_path.write_text(json.dumps(result))
'''


def evaluate_lcb(text, evaluation, backend="docker", timeout=6):
    from .scoring import extract_code, run_sandbox
    if timeout <= 0 or not evaluation["tests"]:
        raise ValueError("LCB needs nonempty tests and a positive per-test timeout")
    inputs = {"inputs": [t["input"] for t in evaluation["tests"]],
              "outputs": [t["output"] for t in evaluation["tests"]], "fn_name": evaluation["fn_name"]}
    total = (len(evaluation["tests"]) + 1) * math.ceil(timeout) + 5
    files = {"candidate.py": extract_code(text), "worker.py": WORKER,
             "sample.json": json.dumps({"input_output": json.dumps(inputs)}),
             "testing_util.py": (ROOT / "third_party/livecodebench_official/testing_util.py").read_text()}
    result = run_sandbox(files, [str(math.ceil(timeout)), str(total)], backend, total, memory_gb=4)
    result["official_scorer_commit"] = UPSTREAM_COMMIT
    return result
