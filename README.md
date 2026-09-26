# SpecGrove

**Reusable Tree Capacity for Concurrent Diffusion Speculation**

SpecGrove reuses one canonical DDTree expansion as nested candidate tiers.
An exact-row allocator selects a tier for each request under a shared target
verification budget, using draft-prefix coverage, service weights, and a
measured target-cost curve. Selected trees share a request-isolated target pass.

This repository contains the research implementations, experiment drivers,
model configurations, and regression tests. It does not contain the paper,
experimental result archives, model weights, or benchmark data.

## Quick Start: No GPU Required

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
make check
make test
```

`make check` parses the Python sources, checks the frozen code fingerprints,
and scans tracked release files for selected credential patterns.
`make test` runs release-tool tests plus the 14 natural-protocol and migration
tests. These checks do not run the target/draft models.

## Source Layout

| Path | Purpose |
| --- | --- |
| `code/natural/` | Frozen natural-task throughput, budget, quality, mixed-workload, and ablation experiments |
| `code/serving/` | Later matched scheduler controls, Poisson serving, and numerical audit implementation |
| `code/dp-paper-natural-suite-20260917/` | Predecessor snapshot required by protocol-migration regression tests |
| `scripts/prepare_benchmarks.py` | Prepare the four public benchmark sources |
| `scripts/check_repository.py` | CPU-only release integrity checks |
| `.github/workflows/checks.yml` | CPU-only continuous integration |

The snapshots are intentionally separate: they correspond to different source
revisions and must not be merged when reproducing their original protocols.
Most users should start with `code/natural/`; use `code/serving/` for the
additional scheduling and serving experiments.

### Main Implementation

- `code/natural/src/gbv_experiments/continuous_tree_block_decode.py`:
  reusable tree tiers, exact-row allocation, request-local caches and packed
  target verification.
- `code/natural/scripts/paper_dp_allocators.py`: matched allocation controls.
- `code/natural/scripts/run_natural_paper_phase.py`: natural-task experiment
  entrypoint and measurement contract.
- `code/serving/scripts/run_openloop_scheduler_supplement.py`: matched
  policy controls and real-time Poisson arrivals.

## GPU Environment

The recorded formal experiments used Linux, Python 3.11.16,
PyTorch 2.8.0+cu128, and NVIDIA H20. Install a compatible CUDA PyTorch wheel
separately, then install the other experiment dependencies:

```bash
python -m pip install -r requirements-experiment.txt
```

The requirements file is not a recovered full runtime lockfile. Model revisions
are pinned in `code/*/configs/adaptive_block_qwen3_{4b,8b}.json`; download the
corresponding Qwen3 and DFlash-b16 weights separately. Matched inference uses
BF16 eager SDPA with FP32 probability bookkeeping. Legacy optional benchmark
paths may have additional third-party requirements.

## Prepare Benchmarks

After installing the experiment dependencies:

```bash
python scripts/prepare_benchmarks.py --output data/prepared
```

This prepares GSM8K, MATH-500, HumanEval, and sanitized MBPP through the
existing formatter and sample-selection implementation. The preparation
policy retains 164 HumanEval examples; the formal runner uses the first 128.
The other prepared subsets contain 128 examples each.

Preparation resolves the current upstream dataset revisions. **The historical
frozen prepared files were not found locally.** New preparation is not
guaranteed to reproduce their exact bytes. Historical dataset hashes and
sample identities are recorded in the separately distributed evidence bundle;
match those before describing a rerun as an exact reproduction.

## Run Natural-Task Experiments

Provide an independently measured calibration JSON with a `curve` array of
`[verification_rows, target_cost]` points, in increasing row order. Calibration
is specific to the hardware and execution stack; a historical H20 curve is
not a universal model of target-pass cost.

From the repository root:

```bash
python code/natural/scripts/run_natural_paper_phase.py \
  --model qwen3_4b --phase main128 \
  --output results/qwen3_4b/main128 --data-dir data/prepared \
  --calibration /path/to/calibration.json
```

Use `qwen3_8b` and its model-specific calibration for the 8B pair. The other
formal phases are `budget`, `heterogeneous`, `ablation`, `native_c1`,
`quality`, and `correctness`. `--smoke` uses two output tokens and is not
a formal measurement. Use a fresh output directory; historical resume/reuse
hooks are for the recorded protocol, not new reproductions.

Main performance runs use temperature 1 and a 256-token output cap; quality
uses a separate 1,024-token cap and five seeds. Infeasible methods are excluded
by row-capacity checks, not recorded as zero throughput. Native DDTree/DFlash
comparisons are confined to concurrency 1.

## Run Poisson Serving

```bash
python code/serving/scripts/run_openloop_scheduler_supplement.py \
  --model qwen3_8b \
  --config code/serving/configs/adaptive_block_qwen3_8b.json \
  --data-dir data/prepared --calibration /path/to/calibration.json \
  --output results/serving_long256 --phase open_loop \
  --open-loop-requests 256 --max-new-tokens 256
```

Use `--phase scheduler --scheduler-requests 128` for matched closed-batch
scheduler comparisons. The TETRIS-style and ECHO-style controls transfer
allocation principles onto the same DDTree backend; they are not native
reproductions of the original serving systems.

## Additional Tests and Grading

With PyTorch and Transformers installed:

```bash
PYTHONPATH=code/serving/src:code/serving/scripts python -m pytest -q \
  code/serving/tests/test_scheduler_style_baselines.py \
  code/serving/tests/gbv_paper/test_shared_budget_sequence_law.py
```

The sequence-law tests use exact finite-state fixtures. They do not certify
real-model BF16 output-law equality. Surrogate-optimal row allocation is also
not guaranteed to maximize wall-time throughput.

The natural quality grader is
`code/natural/scripts/grade_natural_paper_quality.py`. Code-task grading
requires the Linux sandbox, reference self-tests, resource limits, Landlock,
and seccomp. Do not execute model-generated benchmark programs unsandboxed.

## Distribution and Licensing

Private remote launchers, credentials, Git history, generated caches, results,
and weights are excluded. `SOURCE_MANIFEST.sha256` fingerprints the frozen
code snapshots. Public upstream author attribution and licenses are retained
under `code/*/third_party/`; see `THIRD_PARTY_NOTICES.md`.

No new license is assigned to the authors' research code in this release.
An author-owned public repository identifies its authors; do not use such a
URL as an anonymous submission artifact without considering venue policy.
