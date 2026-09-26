#!/usr/bin/env python3
"""Tune cross-request architectures against fixed official baselines."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time

import torch

from benchmark_continuous_tree_block_e2e import prepared_prompts, summarize
from gbv_experiments.config import Variant, load_config
from gbv_experiments.continuous_tree_block_decode import (
    ContinuousDecodeResult,
    generate_continuous_tree_blocks,
)
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.sampling import probabilities, sample
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


DEFAULT_DATASETS = ("gsm8k", "humaneval", "mt-bench")
DDTREE_BASELINE = "ddtree_full46"
TARGET_BASELINE = "target_ar"


def integers(raw):
    values = tuple(int(value) for value in raw.split(","))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def strings(raw):
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def median_summary(rows):
    result = dict(rows[0])
    numeric = (
        "wall_ms", "output_tokens", "tokens_per_second",
        "physical_target_calls", "physical_target_calls_per_output_token",
        "physical_target_rows", "useful_target_rows", "padded_target_rows",
        "padding_fraction", "logical_rounds", "continuation_blocks",
        "continuation_probability", "gated_continuations",
    )
    for field in numeric:
        result[field] = statistics.median(row[field] for row in rows)
    result["stage_ms"] = {
        field: statistics.median(row["stage_ms"][field] for row in rows)
        for field in rows[0]["stage_ms"]
    }
    return result


@torch.inference_mode()
def generate_batched_target_ar(
        engine, tokenizer, prompts, *, max_new_tokens, stop_ids, seeds,
        target_temperature):
    """Strong eager Target-only baseline with one cached batch per cell."""
    if not prompts or len(prompts) != len(seeds) or max_new_tokens < 1:
        raise ValueError("Invalid batched autoregressive request set")
    batch_size = len(prompts)
    prompt_lengths = tuple(int(prompt.shape[1]) for prompt in prompts)
    maximum_prompt = max(prompt_lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = int(stop_ids[0])
    ids = torch.full(
        (batch_size, maximum_prompt), int(pad_id),
        dtype=torch.long, device=engine.device,
    )
    attention = torch.zeros_like(ids)
    for slot, (prompt, length) in enumerate(zip(prompts, prompt_lengths)):
        ids[slot, maximum_prompt - length:] = prompt[0]
        attention[slot, maximum_prompt - length:] = 1
    positions = attention.cumsum(dim=-1).sub(1).clamp_min(0)
    generators = tuple(
        torch.Generator(device=engine.device).manual_seed(seed)
        for seed in seeds
    )
    stops = set(int(value) for value in stop_ids)
    probability_dtype = torch.float64
    cache = engine.cache_factory()
    stage = {"prefill": 0.0, "target": 0.0, "select_commit": 0.0}
    engine.sync()
    wall_started = time.perf_counter()

    started = time.perf_counter()
    output = engine.target_forward(
        ids, cache, hidden=False, positions=positions,
        mask=attention, last_only=True,
    )
    engine.sync()
    stage["prefill"] += 1000 * (time.perf_counter() - started)

    started = time.perf_counter()
    anchor_logits = output.logits[:, -1]
    if target_temperature == 0.0:
        generated = [
            [int(torch.argmax(anchor_logits[slot]).item())]
            for slot in range(batch_size)
        ]
    else:
        generated = [
            [int(sample(probabilities(anchor_logits[slot], target_temperature,
                                      probability_dtype), generator))]
            for slot, generator in enumerate(generators)
        ]
    done = [
        tokens[-1] in stops or len(tokens) >= max_new_tokens
        for tokens in generated
    ]
    engine.sync()
    stage["select_commit"] += 1000 * (time.perf_counter() - started)
    del output, anchor_logits

    target_calls = 0
    physical_rows = 0
    useful_rows = 0
    while not all(done):
        active = [not value and len(generated[index]) < max_new_tokens
                  for index, value in enumerate(done)]
        current = torch.tensor(
            [generated[index][-1] if active[index] else int(pad_id)
             for index in range(batch_size)],
            dtype=torch.long, device=engine.device,
        )[:, None]
        attention = torch.cat((
            attention,
            torch.tensor(active, dtype=attention.dtype,
                         device=engine.device)[:, None],
        ), dim=1)
        step_positions = torch.tensor(
            [prompt_lengths[index] + len(generated[index]) - 1
             if active[index] else 0
             for index in range(batch_size)],
            dtype=torch.long, device=engine.device,
        )[:, None]
        started = time.perf_counter()
        output = engine.target_forward(
            current, cache, hidden=False, positions=step_positions,
            mask=attention, last_only=True,
        )
        engine.sync()
        stage["target"] += 1000 * (time.perf_counter() - started)
        target_calls += 1
        physical_rows += batch_size
        useful_rows += sum(active)

        started = time.perf_counter()
        logits = output.logits[:, -1]
        for slot, generator in enumerate(generators):
            if not active[slot]:
                continue
            token = (
                int(torch.argmax(logits[slot]).item())
                if target_temperature == 0.0
                else int(sample(
                    probabilities(
                        logits[slot], target_temperature, probability_dtype,
                    ),
                    generator,
                ))
            )
            generated[slot].append(token)
            done[slot] = token in stops or len(generated[slot]) >= max_new_tokens
        engine.sync()
        stage["select_commit"] += 1000 * (time.perf_counter() - started)
        del output, logits

    engine.sync()
    wall_ms = 1000 * (time.perf_counter() - wall_started)
    outputs = tuple(tuple(tokens) for tokens in generated)
    request_rounds = tuple(
        tuple({"continuation_blocks": 0} for _ in tokens[1:])
        for tokens in outputs
    )
    return ContinuousDecodeResult(
        outputs=outputs,
        request_rounds=request_rounds,
        wall_ms=wall_ms,
        prefill_ms=stage["prefill"],
        proposal_ms=0.0,
        target_ms=stage["target"],
        pack_ms=0.0,
        select_commit_ms=stage["select_commit"],
        physical_target_calls=target_calls,
        physical_target_rows=physical_rows,
        useful_target_rows=useful_rows,
        padded_target_rows=physical_rows - useful_rows,
        effective_row_cap_counts=(),
        effective_tree_budget_counts=(),
        gated_continuations=0,
    )


def method_specs(target_temperature=1.0):
    if target_temperature not in (0.0, 1.0):
        raise ValueError("Target temperature must be 0 or 1")
    greedy = target_temperature == 0.0
    # DDTree is a frozen baseline, not a search family. Keep the repository's
    # official length-15, 45-edge/46-row configuration; only the target rule
    # changes between ancestral sampling at T=1 and greedy verification at T=0.
    specs = {
        TARGET_BASELINE: {
            "family": "target",
            "variant": Variant(
                name=TARGET_BASELINE, method="target", paths=1,
                temperature=target_temperature, probability_dtype="float64",
            ),
            "decode": {},
        },
        DDTREE_BASELINE: {
            "family": "ddtree",
            "variant": Variant(
                name=DDTREE_BASELINE, method="ddtree", paths=1, length=15,
                temperature=target_temperature, draft_temperature=1.0,
                probability_dtype="float64", tree_budget=45,
            ),
            "decode": {
                "row_cap": 46,
                "verifier": "greedy" if greedy else "ancestral_batched",
            },
        }
    }
    if not greedy:
        specs["diffusion_block_verify_r16"] = {
            "family": "ours",
            "variant": Variant(
                name="diffusion_block_verify_r16", method="bv", paths=1,
                length=15, temperature=target_temperature,
                draft_temperature=1.0,
                probability_dtype="float64", tree_budget=45,
            ),
            "decode": {
                "row_cap": 16,
                "verifier": "block_verify",
            },
        }
    specs["ours_same_draw_r12"] = {
        "family": "ours",
        "variant": Variant(
            name="ours_same_draw_r12", method="ddtree", paths=1, length=15,
            temperature=target_temperature, draft_temperature=1.0,
            probability_dtype="float64", tree_budget=11,
        ),
        "decode": {
            "row_cap": 12,
            "verifier": "greedy" if greedy else "same_draw_fused",
        },
    }
    specs["ours_sparse_tree12_propfp32"] = {
        "family": "ours",
        "variant": Variant(
            name="ours_sparse_tree12_propfp32", method="ddtree", paths=1,
            length=15, temperature=target_temperature,
            draft_temperature=1.0,
            probability_dtype="float64", tree_budget=11,
        ),
        "decode": {
            "row_cap": 12,
            "verifier": "greedy" if greedy else "sparse_exit_fused_scan",
            "proposal_probability_dtype": "float32",
        },
    }
    tiered = {
        "ours_12_24_46": {
            "row_cap": 12,
            "mid_occupancy_row_cap": 24,
            "mid_occupancy_threshold": 8,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
        },
        "ours_16_24_46": {
            "row_cap": 16,
            "mid_occupancy_row_cap": 24,
            "mid_occupancy_threshold": 8,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
        },
        "ours_16_32_46": {
            "row_cap": 16,
            "mid_occupancy_row_cap": 32,
            "mid_occupancy_threshold": 8,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
        },
        "ours_24_46": {
            "row_cap": 24,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
        },
    }
    for name, decode in tiered.items():
        specs[name] = {
            "family": "ours",
            "variant": Variant(
                name=name, method="ddtree", paths=1, length=15,
                temperature=target_temperature, draft_temperature=1.0,
                probability_dtype="float64", tree_budget=45,
            ),
            "decode": {
                **decode,
                "verifier": "greedy" if greedy else "sparse_exit_fused_scan",
            },
        }
    gated = {
        "ours_rebuild12": {
            "row_cap": 12,
            "continuation_ready_threshold": 0,
        },
        "ours_continue1_r12": {
            "row_cap": 12,
            "max_continuation_blocks": 1,
        },
        "ours_tail4_c1_r12": {
            "row_cap": 12,
            "continuation_ready_threshold": 4,
            "max_continuation_blocks": 1,
        },
        "ours_tail8_c1_r12": {
            "row_cap": 12,
            "continuation_ready_threshold": 8,
            "max_continuation_blocks": 1,
        },
        "ours_tail4_cascade_r12": {
            "row_cap": 12,
            "continuation_ready_threshold": 4,
        },
        "ours_tail4_c1_12_24": {
            "row_cap": 12,
            "low_occupancy_row_cap": 24,
            "low_occupancy_threshold": 4,
            "continuation_ready_threshold": 4,
            "max_continuation_blocks": 1,
        },
        "ours_tail4_c1_12_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
            "continuation_ready_threshold": 4,
            "max_continuation_blocks": 1,
        },
        "ours_sparse_tree12": {
            "row_cap": 12,
            "proposal_tree_budget": 11,
            "continuation_ready_threshold": 0,
        },
        "ours_load_tree12_tail3_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 3,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 3,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_load_tree12_tail4_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 4,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_load_tree12_tail5_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 5,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 5,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_load_tree12_tail4_priority1_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 4,
            "low_occupancy_expansion_slots": 1,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_load_tree12_tail4_priority2_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 4,
            "low_occupancy_expansion_slots": 2,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_spare_tree12_tail13_priority1_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 13,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 13,
            "low_occupancy_expansion_slots": 1,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_spare_tree12_tail10_priority2_46": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 10,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 10,
            "low_occupancy_expansion_slots": 2,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_spare_tree12_tail15_priority1_24": {
            "row_cap": 12,
            "low_occupancy_row_cap": 24,
            "low_occupancy_threshold": 15,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 23,
            "low_occupancy_tree_threshold": 15,
            "low_occupancy_expansion_slots": 1,
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_load_tree12_tail4_priority1_46_propfp32": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 4,
            "low_occupancy_expansion_slots": 1,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
        "ours_load_tree12_tail4_priority1_46_propfp32_nodrain": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 4,
            "low_occupancy_expansion_slots": 1,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": False,
        },
        "ours_global_hetero_11_23_45_f32_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            "global_tree_fixed_row_equivalent": 32.0,
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_global_hetero_11_23_45_f96_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            "global_tree_fixed_row_equivalent": 96.0,
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_global_hetero_11_23_45_f192_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            "global_tree_fixed_row_equivalent": 192.0,
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_global_hetero_11_23_45_f384_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            "global_tree_fixed_row_equivalent": 384.0,
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_global_hetero_11_23_45_h20_curve_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            # Target-only H20 BF16/SDPA diagnostic curve.  The measured
            # 16-row latency is clamped down to the B11 block's 12 rows. This
            # stays separate until a full packed-wave curve is frozen.
            "global_tree_row_cost_curve": (
                (12, 24.547),
                (46, 24.636),
                (96, 24.803),
                (193, 25.142),
            ),
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_global_hetero_11_23_45_h20_curve_tierfit_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            "global_tree_row_cost_curve": (
                (12, 24.547),
                (46, 24.636),
                (96, 24.803),
                (193, 25.142),
            ),
            # Do not offer a tier unless that tier could cover every request
            # admitted to the current wave.  Smaller feasible tiers remain
            # available, so B23 is still usable at C5-C8.
            "global_tree_require_tier_fit": True,
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_global_hetero_11_23_45_h20_curve384_tierfit_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            "global_tree_row_cost_curve": (
                (12, 24.547),
                (46, 24.636),
                (96, 24.803),
                (193, 25.142),
                # Mean of two independent H20 target-only measurements near
                # the 384-row packed-wave operating point.
                (384, 39.032),
            ),
            "global_tree_require_tier_fit": True,
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_global_hetero_11_23_45_h20_curve384_tierfit_a8_c1_propfp32": {
            "row_cap": 12,
            "global_tree_budget_options": (11, 23, 45),
            "global_tree_row_cost_curve": (
                (12, 24.547),
                (46, 24.636),
                (96, 24.803),
                (193, 25.142),
                (384, 39.032),
            ),
            "global_tree_require_tier_fit": True,
            # Preserve maximum cross-request parallelism above C8; larger
            # diffusion trees become eligible only after the active set drains.
            "global_tree_expansion_active_limit": 8,
            "global_tree_criticality_weight": 1.0,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
        },
        "ours_load_tree12_tail4_46_propfp32": {
            "row_cap": 12,
            "low_occupancy_row_cap": 46,
            "low_occupancy_threshold": 4,
            "proposal_tree_budget": 11,
            "low_occupancy_tree_budget": 45,
            "low_occupancy_tree_threshold": 4,
            "proposal_probability_dtype": "float32",
            "continuation_ready_threshold": 0,
            "adaptive_expansion_requires_drain": True,
        },
    }
    for name, decode in gated.items():
        specs[name] = {
            "family": "ours",
            "variant": Variant(
                name=name, method="ddtree", paths=1, length=15,
                temperature=target_temperature, draft_temperature=1.0,
                probability_dtype="float64", tree_budget=45,
            ),
            "decode": {
                **decode,
                "verifier": "greedy" if greedy else "sparse_exit_fused_scan",
            },
        }
    for slots in (1, 2):
        name = f"ours_load_tree12_tail4_priority{slots}_46_draftkv"
        base = gated[f"ours_load_tree12_tail4_priority{slots}_46"]
        specs[name] = {
            "family": "ours",
            "variant": Variant(
                name=name, method="ddtree", paths=1, length=15,
                temperature=target_temperature, draft_temperature=1.0,
                probability_dtype="float64", tree_budget=45,
            ),
            "decode": {
                **base,
                "verifier": (
                    "greedy" if greedy else "sparse_exit_fused_scan"
                ),
                "reuse_request_draft_cache": True,
            },
        }
    specs["dflash_r16"] = {
        "family": "dflash",
        "variant": Variant(
            name="dflash_r16", method="dflash", paths=1, length=15,
            temperature=target_temperature, draft_temperature=None,
            probability_dtype="float64", tree_budget=45,
        ),
        "decode": {
            "row_cap": 16,
            "verifier": "greedy" if greedy else "terminal_mass",
        },
    }
    return specs


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def analyze(rows, specs, concurrencies):
    by_concurrency = {}
    for concurrency in concurrencies:
        selected = [row for row in rows if row["concurrency"] == concurrency]
        methods = {}
        for name in specs:
            tokens = sum(row["methods"][name]["output_tokens"] for row in selected)
            wall_ms = sum(row["methods"][name]["wall_ms"] for row in selected)
            methods[name] = {
                "family": specs[name]["family"],
                "tokens_per_second": 1000 * tokens / wall_ms,
                "wall_ms": wall_ms,
                "target_calls": sum(
                    row["methods"][name]["physical_target_calls"]
                    for row in selected
                ),
            }
        ddtree_baseline = DDTREE_BASELINE
        best_ours = max(
            (name for name in methods if specs[name]["family"] == "ours"),
            key=lambda name: methods[name]["tokens_per_second"],
        )
        dflash = "dflash_r16"
        methods[best_ours]["speedup_vs_ddtree"] = (
            methods[best_ours]["tokens_per_second"]
            / methods[ddtree_baseline]["tokens_per_second"]
        )
        methods[best_ours]["speedup_vs_dflash"] = (
            methods[best_ours]["tokens_per_second"]
            / methods[dflash]["tokens_per_second"]
        )
        methods[ddtree_baseline]["speedup_vs_dflash"] = (
            methods[ddtree_baseline]["tokens_per_second"]
            / methods[dflash]["tokens_per_second"]
        )
        if TARGET_BASELINE in methods:
            target_rate = methods[TARGET_BASELINE]["tokens_per_second"]
            for name in methods:
                if name != TARGET_BASELINE:
                    methods[name]["speedup_vs_target_ar"] = (
                        methods[name]["tokens_per_second"] / target_rate
                    )
        by_concurrency[str(concurrency)] = {
            "ddtree_baseline": ddtree_baseline,
            "best_ours": best_ours,
            "methods": methods,
        }

    combined = {}
    for name in specs:
        tokens = sum(row["methods"][name]["output_tokens"] for row in rows)
        wall_ms = sum(row["methods"][name]["wall_ms"] for row in rows)
        combined[name] = {
            "family": specs[name]["family"],
            "tokens_per_second": 1000 * tokens / wall_ms,
            "wall_ms": wall_ms,
        }
    ddtree_baseline = DDTREE_BASELINE
    best_ours = max(
        (name for name in combined if specs[name]["family"] == "ours"),
        key=lambda name: combined[name]["tokens_per_second"],
    )
    target_speedups = (
        {
            "ours_vs_target_ar": (
                combined[best_ours]["tokens_per_second"]
                / combined[TARGET_BASELINE]["tokens_per_second"]
            ),
            "ddtree_vs_target_ar": (
                combined[ddtree_baseline]["tokens_per_second"]
                / combined[TARGET_BASELINE]["tokens_per_second"]
            ),
            "dflash_vs_target_ar": (
                combined["dflash_r16"]["tokens_per_second"]
                / combined[TARGET_BASELINE]["tokens_per_second"]
            ),
        }
        if TARGET_BASELINE in combined else {}
    )
    return {
        "by_concurrency": by_concurrency,
        "combined": {
            "ddtree_baseline": ddtree_baseline,
            "best_ours": best_ours,
            "ours_vs_ddtree": (
                combined[best_ours]["tokens_per_second"]
                / combined[ddtree_baseline]["tokens_per_second"]
            ),
            "ours_vs_dflash": (
                combined[best_ours]["tokens_per_second"]
                / combined["dflash_r16"]["tokens_per_second"]
            ),
            "ddtree_vs_dflash": (
                combined[ddtree_baseline]["tokens_per_second"]
                / combined["dflash_r16"]["tokens_per_second"]
            ),
            **target_speedups,
            "methods": combined,
        },
    }


def paired_cell_bootstrap(
        rows, candidate, baseline, *, samples=20_000, seed=20260915):
    """Bootstrap the ratio-of-summed-throughputs over paired result cells."""

    def ratio(selected):
        candidate_tokens = sum(
            row["methods"][candidate]["output_tokens"] for row in selected
        )
        candidate_wall = sum(
            row["methods"][candidate]["wall_ms"] for row in selected
        )
        baseline_tokens = sum(
            row["methods"][baseline]["output_tokens"] for row in selected
        )
        baseline_wall = sum(
            row["methods"][baseline]["wall_ms"] for row in selected
        )
        return (
            candidate_tokens / candidate_wall
            / (baseline_tokens / baseline_wall)
        )

    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [rows[rng.randrange(len(rows))] for _ in rows]
        draws.append(ratio(selected))
    draws.sort()
    return {
        "estimate": ratio(rows),
        "lower_95": draws[int(0.025 * samples)],
        "upper_95": draws[int(0.975 * samples)],
        "paired_cells": len(rows),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", type=strings, default=DEFAULT_DATASETS)
    parser.add_argument("--concurrencies", type=integers, default=(16, 8))
    parser.add_argument(
        "--methods", type=strings,
        help=("Optional comma-separated subset; must include fixed "
              "ddtree_full46, ours, and DFlash; target_ar is optional"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--seeds", type=integers,
        help="Optional comma-separated experiment seeds; overrides --seed",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--row-budget", type=int, default=193,
        help="Maximum total tree rows packed into one Target verification wave",
    )
    parser.add_argument(
        "--target-temperature", type=float, choices=(0.0, 1.0), default=1.0,
        help="Target sampling temperature: 0 for greedy or 1 for sampling",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layout", choices=("packed_sequence", "padded"),
        default="packed_sequence",
    )
    parser.add_argument("--dynamic-padded-slots", action="store_true")
    parser.add_argument(
        "--fixed-speculative-slot-capacity", type=int,
        help=("Optional isolation control: cap DDTree, DFlash, and our method "
              "to the same number of requests per Target wave"),
    )
    args = parser.parse_args()
    if args.repeats < 2 or args.max_new_tokens < 4:
        raise ValueError("Tuning requires >=2 repeats and >=4 output tokens")
    if args.row_budget < 46:
        raise ValueError("Row budget must accommodate the fixed DDTree block")
    if (args.fixed_speculative_slot_capacity is not None
            and args.fixed_speculative_slot_capacity < 1):
        raise ValueError("Fixed speculative slot capacity must be positive")

    experiment_seeds = args.seeds or (args.seed,)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config.resolve())["model"]
    specs = method_specs(args.target_temperature)
    if args.methods:
        unknown = set(args.methods) - set(specs)
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        specs = {name: specs[name] for name in args.methods}
        families = {spec["family"] for spec in specs.values()}
        if not {"ddtree", "ours", "dflash"}.issubset(families):
            raise ValueError("--methods must include DDTree, ours, and DFlash")
    manifest = {
        "kind": "cross_request_architecture_search_fixed_baselines",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": cfg,
        "temperature": args.target_temperature,
        "datasets": args.datasets,
        "concurrencies": args.concurrencies,
        "seed": args.seed,
        "seeds": experiment_seeds,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "row_budget": args.row_budget,
        "layout": args.layout,
        "dynamic_padded_slots": args.dynamic_padded_slots,
        "fixed_speculative_slot_capacity": (
            args.fixed_speculative_slot_capacity
        ),
        "persistent_request_target_cache": args.layout == "packed_sequence",
        "target_verification_backend": "eager",
        "compile_draft": False,
        "ddtree_baseline_policy": "fixed_official_length15_tree_budget45",
        "methods": {
            name: {
                "family": spec["family"],
                "variant": spec["variant"].__dict__,
                "decode": spec["decode"],
            }
            for name, spec in specs.items()
        },
    }
    write_json(args.output / "manifest.json", manifest)

    allocation_gate(args.device)
    engine, tokenizer = load_models(cfg, args.device)
    model_gate(engine)
    engine.set_target_verification_backend("eager")
    stops = stop_token_ids(engine, tokenizer)

    def execute(spec, request_prompts, request_seeds, output_tokens):
        if spec["family"] == "target":
            return generate_batched_target_ar(
                engine, tokenizer, request_prompts,
                max_new_tokens=output_tokens, stop_ids=stops,
                seeds=request_seeds,
                target_temperature=args.target_temperature,
            )
        return generate_continuous_tree_blocks(
            engine, request_prompts, spec["variant"],
            max_new_tokens=output_tokens,
            stop_ids=stops, seeds=request_seeds,
            slot_capacity=(
                args.fixed_speculative_slot_capacity
                if args.fixed_speculative_slot_capacity is not None
                else args.row_budget // spec["decode"]["row_cap"]
            ),
            layout=args.layout, row_budget=args.row_budget,
            persistent_request_target_cache=(
                args.layout == "packed_sequence"
            ),
            fixed_padded_slots=not args.dynamic_padded_slots,
            **spec["decode"],
        )

    warm_prompts, _ = prepared_prompts(
        tokenizer, cfg, engine.device, args.data_dir, "gsm8k", 0, 2,
    )
    for spec in specs.values():
        execute(spec, warm_prompts, (17, 29), 4)

    results_path = args.output / "results.jsonl"
    rows = []
    names = tuple(specs)
    total_cells = (
        len(args.datasets) * len(args.concurrencies) * len(experiment_seeds)
    )
    ordinal = 0
    with results_path.open("w", encoding="utf-8") as stream:
        for concurrency in args.concurrencies:
            for dataset in args.datasets:
                prompts, identities = prepared_prompts(
                    tokenizer, cfg, engine.device, args.data_dir,
                    dataset, 0, concurrency,
                )
                for experiment_seed in experiment_seeds:
                    request_seeds = [
                        experiment_seed * 1_000_003 + index * 104_729
                        for index in range(concurrency)
                    ]
                    runs = {name: [] for name in names}
                    outputs = {name: [] for name in names}
                    for repeat in range(args.repeats):
                        shift = (ordinal + repeat) % len(names)
                        order = names[shift:] + names[:shift]
                        for name in order:
                            spec = specs[name]
                            result = execute(
                                spec, prompts, request_seeds,
                                args.max_new_tokens,
                            )
                            runs[name].append(summarize(result))
                            outputs[name].append(result.outputs)
                    row = {
                        "dataset": dataset,
                        "concurrency": concurrency,
                        "seed": experiment_seed,
                        "prompt_identities": identities,
                        "methods": {
                            name: median_summary(runs[name]) for name in names
                        },
                        "within_method_repeatable": {
                            name: all(value == outputs[name][0] for value in outputs[name])
                            for name in names
                        },
                        "output_equal_to_ddtree_by_repeat": {
                            name: [
                                value == reference
                                for value, reference in zip(
                                    outputs[name], outputs.get(DDTREE_BASELINE, [])
                                )
                            ]
                            for name in names
                        },
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, default=str) + "\n")
                    stream.flush()
                    interim = analyze(rows, specs, tuple(
                        value for value in args.concurrencies
                        if any(row["concurrency"] == value for row in rows)
                    ))
                    print(json.dumps({
                        "progress": f"{len(rows)}/{total_cells}",
                        "dataset": dataset,
                        "concurrency": concurrency,
                        "seed": experiment_seed,
                        "ddtree_baseline": interim["combined"]["ddtree_baseline"],
                        "current_best_ours": interim["combined"]["best_ours"],
                        "ours_vs_ddtree": interim["combined"]["ours_vs_ddtree"],
                    }), flush=True)
                    ordinal += 1

    analysis = analyze(rows, specs, args.concurrencies)
    analysis["all_methods_repeatable"] = all(
        all(row["within_method_repeatable"].values()) for row in rows
    )
    analysis["output_equal_to_ddtree_all_repeats"] = {
        name: all(
            all(row["output_equal_to_ddtree_by_repeat"][name])
            for row in rows
        )
        for name in names
    }
    best_ours = analysis["combined"]["best_ours"]
    ddtree_baseline = analysis["combined"]["ddtree_baseline"]
    analysis["paired_cell_bootstrap"] = {
        "combined": {
            "ours_vs_ddtree": paired_cell_bootstrap(
                rows, best_ours, ddtree_baseline,
            ),
            "ours_vs_dflash": paired_cell_bootstrap(
                rows, best_ours, "dflash_r16", seed=20260916,
            ),
        },
        "by_concurrency": {
            str(concurrency): {
                "ours_vs_ddtree": paired_cell_bootstrap(
                    [row for row in rows if row["concurrency"] == concurrency],
                    analysis["by_concurrency"][str(concurrency)]["best_ours"],
                    analysis["by_concurrency"][str(concurrency)]["ddtree_baseline"],
                    seed=20260915 + concurrency,
                )
            }
            for concurrency in args.concurrencies
        },
    }
    if TARGET_BASELINE in specs:
        analysis["paired_cell_bootstrap"]["combined"].update({
            "ours_vs_target_ar": paired_cell_bootstrap(
                rows, best_ours, TARGET_BASELINE, seed=20260917,
            ),
            "ddtree_vs_target_ar": paired_cell_bootstrap(
                rows, ddtree_baseline, TARGET_BASELINE, seed=20260918,
            ),
            "dflash_vs_target_ar": paired_cell_bootstrap(
                rows, "dflash_r16", TARGET_BASELINE, seed=20260919,
            ),
        })
        for concurrency in args.concurrencies:
            selected = [
                row for row in rows if row["concurrency"] == concurrency
            ]
            analysis["paired_cell_bootstrap"]["by_concurrency"][
                str(concurrency)
            ].update({
                "ours_vs_target_ar": paired_cell_bootstrap(
                    selected,
                    analysis["by_concurrency"][str(concurrency)]["best_ours"],
                    TARGET_BASELINE,
                    seed=20260917 + concurrency,
                ),
                "ddtree_vs_target_ar": paired_cell_bootstrap(
                    selected, ddtree_baseline, TARGET_BASELINE,
                    seed=20260918 + concurrency,
                ),
                "dflash_vs_target_ar": paired_cell_bootstrap(
                    selected, "dflash_r16", TARGET_BASELINE,
                    seed=20260919 + concurrency,
                ),
            })
    analysis["telemetry"] = telemetry(args.device)
    write_json(args.output / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
