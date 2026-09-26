from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


ROOT_MARGINAL_METHODS = frozenset({
    "root_marginal_bv", "root_marginal_bv_ref", "root_marginal_token",
    "root_early_bv", "root_shared_ddtree",
})
SHARED_SUFFIX_METHODS = ROOT_MARGINAL_METHODS | {"root_protected_bv"}
ATOM_TREE_METHODS = frozenset({
    "atom_tree_bv", "atom_tree_bv_fixed", "atom_tree_bv_aligned",
    "atom_tree_bv_no_pool", "atom_tree_ancestral",
})
DIFFUSION_TREE_METHODS = frozenset({
    "diffusion_tree_bv", "diffusion_tree_bv_aligned", "diffusion_tree_bv_no_pool",
    "diffusion_tree_bv_unmerged", "diffusion_tree_ancestral",
    "diffusion_coalesced_block_skip",
    "diffusion_entropy_coalesced_bv", "diffusion_coalesced_flow_bv",
})
DIFFUSION_SCAFFOLD_METHODS = frozenset({
    "diffusion_scaffold_bv", "diffusion_scaffold_no_fill", "diffusion_scaffold_no_recycle",
    "diffusion_scaffold_ancestral", "diffusion_core_spur_bv",
})
DIFFUSION_LAW_METHODS = DIFFUSION_TREE_METHODS | DIFFUSION_SCAFFOLD_METHODS
PREFIX_CORE_SPUR_METHODS = frozenset({
    "prefix_core_spur_bv", "prefix_core_spur_bv_lazy",
    "prefix_core_spur_tree",
    "prefix_sampled_spur_tree",
})
PREFIX_RESCORED_TREE_METHODS = frozenset({
    "prefix_rescored_tree", "prefix_rescored_tree_fused_scan",
    "prefix_hybrid_tree", "prefix_beam_tree",
})
PREFIX_TREE_METHODS = PREFIX_CORE_SPUR_METHODS | PREFIX_RESCORED_TREE_METHODS


@dataclass(frozen=True)
class Variant:
    name: str
    method: str = "gbv"
    paths: int = 3
    length: int = 15  # Future tokens; clean anchor is not counted.
    temperature: float = 1.0
    draft_temperature: float | None = None
    share_prefixes: bool = True
    reuse_draft_cache: bool = True
    draft_attention: str = "bidirectional"
    condition_features: str = "target"
    probability_dtype: str = "float64"
    tree_budget: int = 60
    diffusion_support_size: int = 8
    diffusion_spur_length: int = 4
    prefix_strength: float = 1.0
    prefix_pool_factor: int = 2
    prefix_core_budget: int = 0
    # Architecture-only temperature used to score DDTree proposal prefixes.
    # It never changes the Target sampling temperature or verifier law.
    tree_proposal_temperature: float | None = None
    # Optional final value of a linear depth-wise proposal temperature.  The
    # first value is ``tree_proposal_temperature`` (or the Draft temperature).
    tree_proposal_temperature_end: float | None = None
    # Optional learned temperature at every future depth.  This changes only
    # proposal-tree allocation and adds neither a model forward nor a Target
    # distribution change at runtime.
    tree_proposal_temperature_schedule: tuple[float, ...] | None = None
    # Optional serialized proposal-only vocabulary bias learned on disjoint
    # text.  Loading is setup work; applying it is one vector add before the
    # already-required proposal softmax.
    tree_proposal_bias_path: str | None = None
    # Cost-aware tree: when the geometric mean of the first four greedy Draft
    # probabilities reaches this threshold, verify only the smaller budget.
    tree_adaptive_confidence_threshold: float | None = None
    tree_adaptive_min_budget: int | None = None
    # On confident rounds, reserve one complete greedy block-length spine and
    # fill the remaining rows with canonical DDTree prefixes.
    tree_spine_confidence_threshold: float | None = None
    # Per-edge utility reward used only when allocating proposal-tree nodes.
    # Positive values favor deeper prefixes that can commit more tokens.
    tree_depth_reward: float = 0.0
    # Optional learned structural policy.  The first vector supplies an
    # independent expansion reward at every depth; the second caps sibling
    # ranks at every depth.  Both affect proposal topology only.
    tree_depth_reward_schedule: tuple[float, ...] | None = None
    tree_rank_caps: tuple[int, ...] | None = None
    # History-only Target/Draft calibration for proposal-tree allocation.
    tree_online_ewma: float = 0.25
    tree_online_clip: float = 1.0
    # Stateful compute policy.  The first round uses ``warm_budget``; later
    # rounds choose the low or high (tree_budget) shape from accepted-token
    # history.  This changes tree construction only; Target verification is
    # the same ancestral DDTree verifier.
    tree_history_low_budget: int = 24
    tree_history_warm_budget: int = 45
    tree_history_threshold: float = 5.0
    tree_history_ewma: float = 0.5
    tree_history_low_depth: int = 15
    tree_history_high_depth: int = 15

    def validate(self) -> None:
        if self.method not in {"target", "dflash", "token", "bv", "gbv", "tree_gbv",
                               "tree_gbv_full", "tree_gbv_prefix",
                               "tree_gbv_full_dense_ref",
                               "tree_gbv_full_sparse_ref",
                               "tree_gbv_full_sparse_lazy",
                               "tree_gbv_recycle", "tree_gbv_prefix_recycle",
                               "tree_gbv_budgeted_prefix_recycle",
                               "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch",
                               "tree_gbv_budgeted_prefix_recycle_sparse_lazy",
                               "tree_gbv_budgeted_prefix_recycle_host",
                               "tree_gbv_slot_mixer_recycle",
                               "tree_gbv_ratio_transport_recycle",
                               "tree_gbv_embedded_prefix_recycle",
                               "tree_gbv_prefix_recycle_packed",
                               "tree_gbv_packed",
                               "ddtree", "ddtree_refine", "ddtree_online_rank",
                               "ddtree_budget_policy", "ddtree_history_budget",
                               "ddtree_confidence_spine",
                               "ddtree_terminal_block",
                               "ddtree_terminal_serial", "ddtree_terminal_dense",
                               "ddtree_fused", "ddtree_fused_parallel",
                               "ddtree_fused_scan",
                               "ddtree_sparse_exit_fused_scan",
                               "ddtree_compiled_batched",
                               "ddtree_same_draw_fused",
                               "ddtree_direct_logits_fused_scan",
                               "ddtree_lazy_target",
                               "ddtree_lazy_target_deferred_leaf",
                               "ddtree_lazy_target_prefetch1",
                               "ddtree_lazy_target_prefetch2",
                               "ddtree_lazy_target_aligned32",
                               "ddtree_lazy_target_aligned40",
                               "ddtree_lazy_target_terminal_aligned32",
                               "ddtree_lazy_target_terminal_aligned40",
                               "ddtree_tree_block_skip8",
                               "ddtree_tree_block_skip16",
                               "ddtree_tree_block_skip24",
                               "ddtree_tree_block_skip32",
                               "ddtree_tree_block_skip40",
                               "ddtree_tree_block_skip_conditional005",
                               "ddtree_tree_block_skip_cost_dp",
                               "ddtree_lazy_projection",
                               "ddtree_lazy_softmax_fused_scan",
                               "ddtree_lazy_projection_fused_scan",
                               "rank_calibrated_tree",
                               "block_aligned_tree",
                               "ddtree_slot_mixer",
                               "ddtree_markov_branch"} | SHARED_SUFFIX_METHODS | ATOM_TREE_METHODS | DIFFUSION_LAW_METHODS | PREFIX_TREE_METHODS:
            raise ValueError(f"Unknown method: {self.method}")
        if (self.paths < 1 or self.length < 1 or self.temperature < 0
                or not 0 <= self.prefix_strength <= 2):
            raise ValueError("paths/length must be positive and temperature nonnegative")
        if (not math.isfinite(self.temperature)
                or not math.isfinite(self.prefix_strength) or any(
                not isinstance(v, int) or isinstance(v, bool)
                for v in (self.paths, self.length, self.tree_budget,
                          self.diffusion_support_size,
                          self.diffusion_spur_length,
                          self.prefix_pool_factor,
                          self.prefix_core_budget))):
            raise ValueError("Counts must be integers and temperature must be finite")
        if self.prefix_pool_factor < 1:
            raise ValueError("Prefix candidate-pool factor must be positive")
        if not 0 <= self.prefix_core_budget < self.tree_budget:
            raise ValueError("Prefix DDTree core must be below tree_budget")
        if (self.method == "prefix_hybrid_tree") != (
                self.prefix_core_budget > 0):
            raise ValueError(
                "Only prefix_hybrid_tree may declare a positive DDTree core"
            )
        if self.method in {"dflash", "token", "bv"} and self.paths != 1:
            raise ValueError("Single-path token/BV baselines require paths=1")
        if self.draft_temperature is not None and (self.draft_temperature <= 0 or not math.isfinite(self.draft_temperature)):
            raise ValueError("Draft temperature must be positive")
        if (self.tree_proposal_temperature is not None
                and (self.tree_proposal_temperature <= 0
                     or not math.isfinite(self.tree_proposal_temperature))):
            raise ValueError("Tree proposal temperature must be positive")
        if (self.tree_proposal_temperature_end is not None
                and (self.tree_proposal_temperature_end <= 0
                     or not math.isfinite(self.tree_proposal_temperature_end))):
            raise ValueError("Final tree proposal temperature must be positive")
        if self.tree_proposal_temperature_schedule is not None:
            if (len(self.tree_proposal_temperature_schedule) != self.length
                    or any(value <= 0 or not math.isfinite(value)
                           for value in self.tree_proposal_temperature_schedule)):
                raise ValueError(
                    "Tree proposal temperature schedule must contain one "
                    "positive finite value per future depth"
                )
        if (self.tree_proposal_bias_path is not None
                and not self.tree_proposal_bias_path.strip()):
            raise ValueError("Tree proposal bias path must be nonempty")
        if ((self.tree_adaptive_confidence_threshold is None)
                != (self.tree_adaptive_min_budget is None)):
            raise ValueError("Adaptive tree threshold and minimum budget must be paired")
        if self.tree_adaptive_confidence_threshold is not None:
            if not 0 < self.tree_adaptive_confidence_threshold <= 1:
                raise ValueError("Adaptive tree confidence threshold must be in (0, 1]")
            if (not isinstance(self.tree_adaptive_min_budget, int)
                    or isinstance(self.tree_adaptive_min_budget, bool)
                    or not 1 <= self.tree_adaptive_min_budget < self.tree_budget):
                raise ValueError("Adaptive tree minimum budget must be below tree_budget")
        if (self.tree_spine_confidence_threshold is not None
                and (not 0 < self.tree_spine_confidence_threshold <= 1
                     or self.method != "ddtree_confidence_spine")):
            raise ValueError("Invalid confidence-spine threshold or method")
        if ((self.method == "ddtree_confidence_spine") !=
                (self.tree_spine_confidence_threshold is not None)):
            raise ValueError(
                "Confidence-spine method requires exactly one threshold"
            )
        if not math.isfinite(self.tree_depth_reward):
            raise ValueError("Tree depth reward must be finite")
        if self.tree_depth_reward_schedule is not None:
            if (len(self.tree_depth_reward_schedule) != self.length
                    or any(not math.isfinite(value)
                           for value in self.tree_depth_reward_schedule)):
                raise ValueError(
                    "Tree depth-reward schedule must contain one finite "
                    "value per future depth"
                )
        if self.tree_rank_caps is not None:
            if (len(self.tree_rank_caps) != self.length
                    or any(not isinstance(value, int)
                           or isinstance(value, bool) or value < 1
                           for value in self.tree_rank_caps)):
                raise ValueError(
                    "Tree rank caps must contain one positive integer per "
                    "future depth"
                )
        if (not 0 < self.tree_online_ewma <= 1
                or not math.isfinite(self.tree_online_clip)
                or self.tree_online_clip <= 0):
            raise ValueError("Invalid online tree-calibration controls")
        if (self.method == "ddtree_history_budget"
                and (not 1 <= self.tree_history_low_budget < self.tree_budget
                or not 1 <= self.tree_history_warm_budget <= self.tree_budget
                or not 0 < self.tree_history_ewma <= 1
                or not math.isfinite(self.tree_history_threshold)
                or not 1 <= self.tree_history_low_depth <= self.length
                or not 1 <= self.tree_history_high_depth <= self.length)):
            raise ValueError("Invalid history tree-policy controls")
        if self.draft_attention not in {"bidirectional", "causal"}:
            raise ValueError("Invalid draft_attention")
        if self.condition_features not in {"target", "zero"}:
            raise ValueError("Invalid condition_features")
        if self.probability_dtype not in {"float32", "float64"} or self.tree_budget < 1:
            raise ValueError("Invalid numerical precision/tree budget")
        if (self.method in {"ddtree_lazy_target",
                            "ddtree_lazy_target_deferred_leaf",
                            "ddtree_lazy_target_prefetch1",
                            "ddtree_lazy_target_prefetch2",
                            "ddtree_lazy_target_aligned32",
                            "ddtree_lazy_target_aligned40",
                            "ddtree_lazy_target_terminal_aligned32",
                            "ddtree_lazy_target_terminal_aligned40",
                            "ddtree_lazy_softmax_fused_scan",
                            "ddtree_direct_logits_fused_scan",
                            "ddtree_lazy_projection_fused_scan"}
                and (self.temperature <= 0 or self.probability_dtype != "float64")):
            raise ValueError(
                "Lazy-softmax fused DDTree requires T>0 and FP64 probabilities"
            )
        if (self.method.startswith("ddtree_tree_block_skip")
                and self.probability_dtype != "float64"):
            raise ValueError(
                "Tree-block skip verification requires FP64 probabilities"
            )
        if (self.method == "ddtree_tree_block_skip_cost_dp"
                and self.temperature != 1.0):
            raise ValueError("Cost-aware tree-block DP is calibrated for T=1")
        if self.method in DIFFUSION_LAW_METHODS:
            if self.temperature <= 0 or self.probability_dtype != "float64":
                raise ValueError("Diffusion tree verification requires T>0 and FP64 probabilities")
            if not 1 <= self.diffusion_support_size <= 256:
                raise ValueError("Diffusion support size must be in 1..256")
            if self.draft_attention != "bidirectional" or self.condition_features != "target":
                raise ValueError("Diffusion theorem requires the one-step target-conditioned masked block")
            if self.paths * self.length > self.tree_budget:
                raise ValueError("Diffusion trie worst-case nodes exceed tree_budget")
            if self.method == "diffusion_core_spur_bv":
                if (self.paths != 1 or not 1 <= self.diffusion_spur_length < self.length
                        or self.tree_budget <= self.diffusion_spur_length):
                    raise ValueError(
                        "Core-spur BV requires K=1 and 1 <= spur < L <= B"
                    )
            elif self.method in DIFFUSION_SCAFFOLD_METHODS:
                if not self.share_prefixes or (self.paths + 1) * self.length > self.tree_budget:
                    raise ValueError("Scaffold needs prefix sharing and (paths+1)*length <= tree_budget")
        if self.method in PREFIX_TREE_METHODS:
            if (self.temperature <= 0 or self.probability_dtype != "float64"
                    or self.paths != 1
                    or not 1 <= self.diffusion_support_size <= 256
                    or (self.method in PREFIX_CORE_SPUR_METHODS
                        and not 1 <= self.diffusion_spur_length < self.length)
                    or self.tree_budget < self.length
                    or self.draft_attention != "bidirectional"
                    or self.condition_features != "target"):
                raise ValueError(
                    "Prefix core-spur requires T>0, FP64, K=1, "
                    "1 <= spur < L <= B, and the official masked Draft"
                )
        if self.method in SHARED_SUFFIX_METHODS | ATOM_TREE_METHODS:
            if self.probability_dtype != "float64":
                raise ValueError("Root-marginal methods require FP64 probabilities")
            if self.paths * self.length > self.tree_budget:
                raise ValueError("Shared-suffix tree exceeds tree_budget: paths * length")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    allowed = {"model", "datasets", "seeds", "max_new_tokens", "warmup_tokens", "main",
               "ablations", "bootstrap_samples", "scoring", "evaluation",
               "explicit_variants", "method_order",
               "target_verification_backends"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(cfg)-allowed)}")
    if not cfg.get("datasets") or len(set(cfg["datasets"])) != len(cfg["datasets"]):
        raise ValueError("datasets must be a nonempty unique list")
    if not cfg.get("seeds") or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be a nonempty unique list")
    if cfg.get("max_new_tokens", 0) < 1:
        raise ValueError("max_new_tokens must be positive")
    if cfg["model"].get("target_attention", "sdpa") not in {"sdpa", "eager"}:
        raise ValueError("Tree verification requires a backend supporting the explicit 4D branch mask")
    if cfg["model"].get("draft_attention", "sdpa") not in {"sdpa", "eager"}:
        raise ValueError("Attention ablations require sdpa or eager")
    if cfg.get("warmup_tokens", 16) < 2:
        raise ValueError("Warmup must execute at least one decode round")
    if ("target_verification_backends" in cfg
            and not isinstance(cfg["target_verification_backends"], dict)):
        raise ValueError("target_verification_backends must be an object")
    model_keys = {"target", "draft", "target_revision", "draft_revision", "dtype", "target_attention", "draft_attention", "enable_thinking", "allow_tf32"}
    if set(cfg["model"]) - model_keys:
        raise ValueError("Unknown model configuration keys")
    if cfg["model"].get("dtype", "bfloat16") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported model dtype")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in cfg["seeds"]):
        raise ValueError("Seeds must be integers")
    method_order = cfg.get("method_order", {"policy": "seeded_shuffle", "seed": 0})
    if (not isinstance(method_order, dict) or set(method_order) != {"policy", "seed"}
            or method_order["policy"] not in {"seeded_shuffle", "balanced_rotation"}
            or not isinstance(method_order["seed"], int)
            or isinstance(method_order["seed"], bool)):
        raise ValueError("method_order requires a seeded_shuffle or balanced_rotation policy and integer seed")
    from .data import evaluation_policy
    evaluation_policy(cfg["datasets"], cfg.get("evaluation"))
    return cfg


def build_variants(cfg: dict, groups: list[str] | None = None) -> list[dict]:
    variants: dict[str, dict] = {}

    def add(v: Variant, group: str):
        v.validate()
        backend = cfg.get("target_verification_backends", {}).get(
            v.name, "eager"
        )
        key = tuple(
            (k, val) for k, val in v.to_dict().items() if k != "name"
        ) + (("target_verification_backend", backend),)
        for entry in variants.values():
            if entry["key"] == key:
                if group not in entry["groups"]:
                    entry["groups"].append(group)
                return
        if v.name in variants:
            raise ValueError(f"Duplicate variant name: {v.name}")
        variants[v.name] = {"variant": v, "groups": [group], "key": key}

    explicit = cfg.get("explicit_variants")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("explicit_variants must be a nonempty list")
        if "main" in cfg or "ablations" in cfg:
            raise ValueError("explicit_variants cannot be mixed with main/ablations")
        variant_fields = set(Variant.__dataclass_fields__)
        for spec in explicit:
            if not isinstance(spec, dict) or set(spec) != {"variant", "groups"}:
                raise ValueError("Each explicit variant requires variant and groups")
            raw, declared_groups = spec["variant"], spec["groups"]
            if (not isinstance(raw, dict) or "name" not in raw or
                    set(raw) - variant_fields):
                raise ValueError("Invalid explicit variant fields")
            if (not isinstance(declared_groups, list) or not declared_groups or
                    len(set(declared_groups)) != len(declared_groups) or
                    any(not isinstance(group, str) or not group for group in declared_groups)):
                raise ValueError("Explicit variant groups must be nonempty unique strings")
            try:
                variant = Variant(**raw)
            except TypeError as exc:
                raise ValueError("Invalid explicit variant") from exc
            for group in declared_groups:
                add(variant, group)
    else:
        base = Variant(name="gbv", **cfg.get("main", {}))
        base.validate()

    def target_at(temp: float, group: str):
        add(Variant(name=f"target_t{temp:g}", method="target", paths=1,
                    temperature=temp), group)

    if explicit is None:
        target_at(base.temperature, "main")
        add(replace(base, name="dflash_match", method="dflash", paths=1), "main")
        add(replace(base, name="dflash_bv", method="bv", paths=1), "main")
        add(base, "main")
        add(replace(base, name="ddtree", method="ddtree", paths=1), "main")
        ablations = cfg.get("ablations", {})
        for k in ablations.get("paths", []):
            # K=1 GBV is exactly single-path BV; reuse that measured baseline.
            add(replace(base, name=f"gbv_k{k}", paths=k, method="bv" if k == 1 else "gbv"), "paths")
        for length in ablations.get("lengths", []):
            add(replace(base, name=f"gbv_l{length}", length=length), "lengths")
        for temp in ablations.get("temperatures", []):
            target_at(temp, "temperatures")
            add(replace(base, name=f"gbv_t{temp:g}", temperature=temp), "temperatures")
        switches = {
            "prefix_sharing": {"share_prefixes": False},
            "draft_cache": {"reuse_draft_cache": False},
            "bidirectional_attention": {"draft_attention": "causal"},
            "target_features": {"condition_features": "zero"},
            "probability_precision": {"probability_dtype": "float32"},
            "block_verification": {"method": "token", "paths": 1},
        }
        for name in ablations.get("switches", []):
            if name not in switches:
                raise ValueError(f"Unknown ablation: {name}")
            if name != "block_verification":
                add(base, name)
            # BV versus token verification is isolated at K=1.
            if name == "block_verification":
                add(replace(base, name="dflash_bv", method="bv", paths=1), name)
            variant_name = "single_token_rejection" if name == "block_verification" else f"ablate_{name}"
            add(replace(base, name=variant_name, **switches[name]), name)
    known = {g for e in variants.values() for g in e["groups"]}
    if groups and set(groups) - known:
        raise ValueError(f"Unknown groups: {sorted(set(groups)-known)}")
    selected = [e for e in variants.values() if not groups or set(groups) & set(e["groups"])]
    # Always retain an AR baseline at every tested target temperature.
    temps = {e["variant"].temperature for e in selected}
    for temp in temps:
        for e in variants.values():
            if e["variant"].method == "target" and e["variant"].temperature == temp and e not in selected:
                selected.append(e)
    return [{"variant": e["variant"].to_dict(), "groups": e["groups"]} for e in selected]


def select_variants(entries, names=None):
    """Select a scheduling phase without changing the declared full experiment."""
    if names is None:
        return entries
    if not names or len(set(names)) != len(names):
        raise ValueError("Variant selection must be nonempty and unique")
    unknown = set(names) - {e["variant"]["name"] for e in entries}
    if unknown:
        raise ValueError(f"Unknown variants in this experiment: {sorted(unknown)}")
    return [e for e in entries if e["variant"]["name"] in names]
