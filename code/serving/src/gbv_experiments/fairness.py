"""Fail-closed checks for architecture-only DDTree comparisons."""
from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

from .config import Variant


# ``method`` and ``paths`` describe the candidate architecture.  Every other
# Variant field affects the experimental workload or numerical protocol and is
# frozen for an architecture-only comparison.
ARCHITECTURE_FIELDS = frozenset({
    "name", "method", "paths", "tree_proposal_temperature",
    "tree_proposal_temperature_end", "tree_proposal_temperature_schedule",
    "tree_proposal_bias_path", "tree_depth_reward",
    "tree_adaptive_confidence_threshold", "tree_adaptive_min_budget",
    "prefix_pool_factor", "prefix_core_budget",
    "tree_online_ewma", "tree_online_clip",
})
FROZEN_MODEL_SETTINGS = {
    "dtype": "bfloat16",
    "target_attention": "sdpa",
    "draft_attention": "sdpa",
    "enable_thinking": False,
    "allow_tf32": False,
}


def assert_architecture_only_pair(
    baseline: Variant,
    candidate: Variant,
    model: Mapping[str, object],
    *, positive_temperature_study: bool = False,
    greedy_study: bool = False,
    expected_length: int = 15,
    expected_tree_budget: int = 45,
) -> dict[str, object]:
    """Validate and return an auditable single-variable comparison record.

    Both methods must run in the same process with the same loaded model
    objects; callers are responsible for interleaving their execution.  This
    check prevents scripts from changing attention kernels, precision, length,
    budget, temperatures, cache policy, or feature conditioning per method.
    """
    baseline.validate()
    candidate.validate()
    if baseline.method != "ddtree":
        raise ValueError("The frozen baseline must be DDTree")

    left, right = asdict(baseline), asdict(candidate)
    mismatches = {
        field: (left[field], right[field])
        for field in left
        if field not in ARCHITECTURE_FIELDS and left[field] != right[field]
    }
    if mismatches:
        raise ValueError(
            "Architecture-only comparison changed frozen controls: "
            f"{mismatches}"
        )

    actual_model = {
        key: model.get(key, default)
        for key, default in FROZEN_MODEL_SETTINGS.items()
    }
    if actual_model != FROZEN_MODEL_SETTINGS:
        raise ValueError(
            "Architecture-only model execution settings are not frozen: "
            f"expected={FROZEN_MODEL_SETTINGS}, actual={actual_model}"
        )
    if positive_temperature_study and greedy_study:
        raise ValueError(
            "Positive-temperature and greedy study modes are mutually exclusive"
        )
    if greedy_study:
        if baseline.temperature != 0 or baseline.draft_temperature != 1.0:
            raise ValueError(
                "Greedy Target study requires T_target=0 and T_draft=1.0"
            )
    elif positive_temperature_study:
        if baseline.temperature <= 0 or baseline.draft_temperature != baseline.temperature:
            raise ValueError("Diffusion study requires matched positive target/draft temperatures")
    elif baseline.temperature != 1.0 or baseline.draft_temperature != 1.0:
        raise ValueError("Formal comparison requires T_target=T_draft=1.0")
    if baseline.probability_dtype != "float64":
        raise ValueError("Formal comparison requires FP64 sampling probabilities")
    if (baseline.length != expected_length
            or baseline.tree_budget != expected_tree_budget):
        raise ValueError(
            "Architecture comparison length/tree budget drifted: "
            f"expected L={expected_length}, B={expected_tree_budget}"
        )

    return {
        "comparison": "architecture_only",
        "architecture_delta": {
            "baseline": {field: left[field] for field in sorted(ARCHITECTURE_FIELDS)},
            "candidate": {field: right[field] for field in sorted(ARCHITECTURE_FIELDS)},
        },
        "frozen_variant_controls": {
            field: left[field] for field in left if field not in ARCHITECTURE_FIELDS
        },
        "frozen_model_controls": actual_model,
    }


def assert_official_dflash_control(
    baseline: Variant,
    control: Variant,
    model: Mapping[str, object],
) -> dict[str, object]:
    """Validate the official greedy-Draft DFlash comparison controls.

    ``draft_temperature`` is deliberately *not* shared with DDTree.  DDTree
    samples its probability tree at T=1, whereas the official DFlash path is
    the masked-block argmax and therefore records ``draft_temperature=None``.
    Every execution control that actually is common remains fail-closed.
    """
    baseline.validate()
    control.validate()
    if baseline.method != "ddtree" or control.method != "dflash":
        raise ValueError("Expected a DDTree baseline and official DFlash control")

    left, right = asdict(baseline), asdict(control)
    method_specific = frozenset({"name", "method", "draft_temperature"})
    mismatches = {
        field: (left[field], right[field])
        for field in left
        if field not in method_specific and left[field] != right[field]
    }
    if mismatches:
        raise ValueError(
            "Official DFlash control changed shared controls: "
            f"{mismatches}"
        )

    actual_model = {
        key: model.get(key, default)
        for key, default in FROZEN_MODEL_SETTINGS.items()
    }
    if actual_model != FROZEN_MODEL_SETTINGS:
        raise ValueError(
            "Official DFlash model execution settings are not frozen: "
            f"expected={FROZEN_MODEL_SETTINGS}, actual={actual_model}"
        )
    if baseline.temperature != 1.0 or baseline.draft_temperature != 1.0:
        raise ValueError("Formal DDTree control requires T_target=T_draft=1.0")
    if control.temperature != 1.0 or control.draft_temperature is not None:
        raise ValueError(
            "Official DFlash requires T_target=1.0 and greedy Draft "
            "(draft_temperature=None)"
        )
    if baseline.probability_dtype != "float64":
        raise ValueError("Formal comparison requires FP64 sampling probabilities")
    if baseline.length != 15 or baseline.tree_budget != 45:
        raise ValueError("Formal comparison requires L=15 and tree budget=45")

    return {
        "comparison": "official_dflash_control",
        "method_specific_proposals": {
            "ddtree": "T=1 probability_tree",
            "dflash": "greedy_argmax masked block",
        },
        "frozen_shared_variant_controls": {
            field: left[field]
            for field in left
            if field not in method_specific
        },
        "recorded_draft_temperatures": {
            "ddtree": baseline.draft_temperature,
            "dflash": control.draft_temperature,
        },
        "frozen_model_controls": actual_model,
    }
