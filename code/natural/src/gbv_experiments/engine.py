from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import json
import sys
import time

import torch

from .common import ROOT
from .config import Variant
from . import root_marginalized_bv as root_marginal
from . import protected_tree_bv
from . import atom_tree_bv
from . import diffusion_tree_bv
from . import coalesced_block_skip
from .config import (SHARED_SUFFIX_METHODS, ATOM_TREE_METHODS,
                     DIFFUSION_SCAFFOLD_METHODS,
                     DIFFUSION_LAW_METHODS as DIFFUSION_TREE_METHODS,
                     PREFIX_CORE_SPUR_METHODS,
                     PREFIX_RESCORED_TREE_METHODS, PREFIX_TREE_METHODS)
from .sampling import (block_verify_batched, block_verify_sparse,
                       block_verify_sparse_lazy, greedy_max_distribution,
                       matching_verify, probabilities, reweight_selected_path,
                       sample, select_and_reweight, select_greedy_path,
                       tree_block_verify_recycle,
                       tree_block_verify_terminal_mass,
                       tree_verify_internal_ancestral_batched,
                       tree_verify_ancestral_lazy_projection,
                       tree_verify_ancestral_batched,
                       token_verify)
from .fused_tree_sampling import (tree_verify_ancestral_fused,
                                  tree_verify_ancestral_fused_parallel,
                                  tree_verify_ancestral_fused_scan,
                                  tree_verify_ancestral_sparse_exit_fused_scan,
                                  tree_verify_ancestral_same_draw_fused,
                                  tree_verify_ancestral_logits_fused_scan,
                                  tree_verify_ancestral_lazy_projection_fused_scan,
                                  tree_verify_ancestral_lazy_softmax_fused_scan)
from .tree import (adaptive_path_proposal, adaptive_prefix_proposal,
                   budgeted_prefix_proposal, compact_cache, probability_tree,
                   sampled_tree, block_aligned_spine_tree,
                   confidence_spine_tree,
                   embedded_prefix_proposal, online_rank_calibrated_tree,
                   update_online_rank_bias, Tree)
from .tree_block_skip import (child_for_edge, cost_aware_tree_block,
                              latency_curve_from_anchors, local_tree_block,
                              risk_bounded_prefix_cap)


SPARSE_FULL_TREE_METHODS = {
    "tree_gbv_full", "tree_gbv_prefix", "tree_gbv_full_sparse_ref",
    "tree_gbv_full_sparse_lazy",
}
FINITE_TREE_METHODS = SPARSE_FULL_TREE_METHODS | {"tree_gbv_full_dense_ref"}
RECYCLE_TREE_METHODS = {
    "tree_gbv_recycle", "tree_gbv_prefix_recycle",
    "tree_gbv_budgeted_prefix_recycle",
    "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch",
    "tree_gbv_budgeted_prefix_recycle_sparse_lazy",
    "tree_gbv_budgeted_prefix_recycle_host",
    "tree_gbv_slot_mixer_recycle",
    "tree_gbv_ratio_transport_recycle",
    "tree_gbv_embedded_prefix_recycle",
    "tree_gbv_prefix_recycle_packed",
}
PACKED_TREE_METHODS = {"tree_gbv_prefix_recycle_packed", "tree_gbv_packed"}
HOST_RECYCLE_METHODS = {"tree_gbv_budgeted_prefix_recycle_host"}
FINITE_TREE_METHODS.update(RECYCLE_TREE_METHODS)
TERMINAL_TREE_METHODS = {
    "ddtree_terminal_block", "ddtree_terminal_serial", "ddtree_terminal_dense",
}
FUSED_TREE_METHODS = {
    "ddtree_fused", "ddtree_fused_parallel", "ddtree_fused_scan",
    "ddtree_sparse_exit_fused_scan", "ddtree_compiled_batched",
    "ddtree_same_draw_fused",
}
LAZY_HEAD_TREE_METHODS = {
    "ddtree_lazy_projection", "ddtree_lazy_projection_fused_scan",
}
LAZY_SOFTMAX_TREE_METHODS = {"ddtree_lazy_softmax_fused_scan"}
DIRECT_LOGITS_TREE_METHODS = {"ddtree_direct_logits_fused_scan"}
LAZY_TARGET_TREE_PREFETCH = {
    "ddtree_lazy_target":0,
    "ddtree_lazy_target_deferred_leaf":0,
    "ddtree_lazy_target_prefetch1":1,
    "ddtree_lazy_target_prefetch2":2,
    "ddtree_lazy_target_aligned32":-32,
    "ddtree_lazy_target_aligned40":-40,
    "ddtree_lazy_target_terminal_aligned32":-32,
    "ddtree_lazy_target_terminal_aligned40":-40,
}
LAZY_TARGET_TREE_METHODS = set(LAZY_TARGET_TREE_PREFETCH)
PARTIAL_TERMINAL_TREE_METHODS = {
    "ddtree_lazy_target_terminal_aligned32",
    "ddtree_lazy_target_terminal_aligned40",
}
TREE_BLOCK_SKIP_ROWS = {
    "ddtree_tree_block_skip8": 8,
    "ddtree_tree_block_skip16": 16,
    "ddtree_tree_block_skip24": 24,
    "ddtree_tree_block_skip32": 32,
    "ddtree_tree_block_skip40": 40,
}
CONDITIONAL_TREE_BLOCK_SKIP_METHODS = {
    "ddtree_tree_block_skip_conditional005",
}
COST_AWARE_TREE_BLOCK_SKIP_METHODS = {
    "ddtree_tree_block_skip_cost_dp",
}
TREE_BLOCK_SKIP_METHODS = (
    set(TREE_BLOCK_SKIP_ROWS) | CONDITIONAL_TREE_BLOCK_SKIP_METHODS
    | COST_AWARE_TREE_BLOCK_SKIP_METHODS
)
# Direct full-Target medians measured on the experiment's Qwen3-4B/T=1/H20
# path.  The DP consumes the complete curve, including its fixed call cost;
# there is no target row-count or probability threshold in the policy.
H20_QWEN3_4B_T1_TARGET_LATENCY_MS = latency_curve_from_anchors(46, (
    (8, 26.17144012451172),
    (12, 25.459247589111328),
    (16, 26.01968002319336),
    (24, 26.148144721984863),
    (32, 26.128864288330078),
    (40, 26.515711784362793),
    (46, 26.423328399658203),
))
DEFERRED_LEAF_TREE_METHODS = {
    "ddtree_lazy_target_deferred_leaf",
    "ddtree_lazy_target_aligned32", "ddtree_lazy_target_aligned40",
}
TARGET_TREE_BACKEND_SPECS = {
    "eager": None,
    "torch_compile_inductor_dynamic": {
        "scope":"decoder", "mode":"reduce-overhead", "dynamic":True,
    },
    "torch_compile_inductor_default_dynamic": {
        "scope":"decoder", "mode":"default", "dynamic":True,
    },
    "torch_compile_inductor_reduce_overhead_static": {
        "scope":"decoder", "mode":"reduce-overhead", "dynamic":False,
    },
    "torch_compile_inductor_full_target_dynamic": {
        "scope":"full_target", "mode":"reduce-overhead", "dynamic":True,
    },
}
TARGET_TREE_BACKENDS = frozenset(TARGET_TREE_BACKEND_SPECS)


class StageMeter:
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.host = defaultdict(float)
        self.events = []

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        pair = None
        if self.device.type == "cuda":
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
        yield
        if pair:
            pair[1].record()
            self.events.append((name, pair))
        self.host[name] += 1000 * (time.perf_counter() - start)

    def result(self):
        gpu = defaultdict(float)
        for name, pair in self.events:
            gpu[name] += pair[0].elapsed_time(pair[1])
        return {"host_ms": dict(self.host), "cuda_event_ms": dict(gpu)}


def summarize_stage_profile(
        stages: dict, *, e2e_ms: float, decode_ms: float,
        official_scope_decode_ms: float) -> dict:
    """Rank diagnostic stages without presenting them as primary wall time.

    Host intervals measure Python submission time on CUDA and CUDA events
    measure stream time. Neither is silently substituted for the synchronized
    end-to-end timer. The residuals make uninstrumented orchestration visible.
    """
    excluded_from_decode = {"prefill"}
    excluded_from_official = {
        "prefill", "draft_prefill", "first_draft_boundary_sync",
    }

    def view(values: dict[str, float], *, excluded: set[str]) -> dict:
        selected = {
            name: float(milliseconds)
            for name, milliseconds in values.items()
            if name not in excluded
        }
        total = sum(selected.values())
        ranked = sorted(selected.items(), key=lambda item: item[1], reverse=True)
        return {
            "measured_stage_ms": selected,
            "measured_stage_total_ms": total,
            "ranked_stages": [
                {
                    "stage": name,
                    "milliseconds": milliseconds,
                    "percent_of_measured_stages": (
                        100 * milliseconds / total if total else None
                    ),
                }
                for name, milliseconds in ranked
            ],
            "longest_stage": ranked[0][0] if ranked else None,
        }

    host = {name: float(value) for name, value in stages["host_ms"].items()}
    cuda = {
        name: float(value) for name, value in stages["cuda_event_ms"].items()
    }
    host_decode = view(host, excluded=excluded_from_decode)
    host_official = view(host, excluded=excluded_from_official)
    cuda_decode = view(cuda, excluded=excluded_from_decode)
    cuda_official = view(cuda, excluded=excluded_from_official)
    return {
        "diagnostic_only": True,
        "eligible_for_primary_timing": False,
        "profile_instrumentation_enabled": True,
        "interpretation": {
            "host_ms": "Python/CPU interval; on CUDA this is mainly launch and orchestration time",
            "cuda_event_ms": "GPU stream interval; use this ranking for GPU-kernel hotspots",
            "unattributed_ms": "synchronized wall time not enclosed by a named stage",
        },
        "synchronized_wall_ms": {
            "e2e": float(e2e_ms),
            "decode_including_first_draft": float(decode_ms),
            "official_scope_decode": float(official_scope_decode_ms),
        },
        "host_decode_scope": {
            **host_decode,
            "unattributed_ms": max(
                0.0, float(decode_ms) - host_decode["measured_stage_total_ms"]
            ),
        },
        "host_official_scope": {
            **host_official,
            "unattributed_ms": max(
                0.0,
                float(official_scope_decode_ms)
                - host_official["measured_stage_total_ms"],
            ),
        },
        "cuda_decode_scope": cuda_decode,
        "cuda_official_scope": cuda_official,
    }


def draft_model_class():
    # Load the vendored package under a private name to avoid 'model' collisions.
    path = ROOT / "third_party/ddtree_official/model/__init__.py"
    spec = importlib.util.spec_from_file_location("_gbv_dflash_model", path,
                                                 submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DFlashDraftModel


def load_models(cfg: dict, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal model benchmarks require an available CUDA GPU")
    torch.cuda.set_device(device)
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"], revision=cfg.get("target_revision"), torch_dtype=dtype,
        attn_implementation=cfg.get("target_attention", "sdpa"),
    ).to(device).eval()
    draft = draft_model_class().from_pretrained(
        cfg["draft"], revision=cfg.get("draft_revision"), torch_dtype=dtype,
        attn_implementation=cfg.get("draft_attention", "sdpa"),
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["target"], revision=cfg.get("target_revision"))
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("Target and draft vocabularies differ")
    if target.config.hidden_size != draft.config.hidden_size:
        raise ValueError("Target/draft hidden sizes differ")
    if draft.mask_token_id is None or not (0 <= draft.mask_token_id < target.config.vocab_size):
        raise ValueError("Invalid draft mask token")
    if any(i < 0 or i >= target.config.num_hidden_layers for i in draft.target_layer_ids):
        raise ValueError("Invalid target feature layer IDs")
    target.requires_grad_(False)
    draft.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", False))
    return Engine(target, draft), tokenizer


class Engine:
    def __init__(self, target, draft, cache_factory=None):
        self.target, self.draft = target, draft
        self.device = next(target.parameters()).device
        if cache_factory is None:
            from transformers import DynamicCache
            cache_factory = DynamicCache
        self.cache_factory = cache_factory
        self.proposal_adapter = None
        self._tree_proposal_bias_cache = {}
        self._eager_target_decoder = getattr(target, "model", None)
        self._compiled_target_decoder = None
        self._compiled_target_decoders = {}
        self._compiled_target_models = {}
        self._target_verification_backend = "eager"

    def set_target_verification_backend(self, backend: str) -> None:
        """Select the executor used only by the Target tree forward."""
        if backend not in TARGET_TREE_BACKENDS:
            raise ValueError(f"Unknown Target verification backend: {backend}")
        spec = TARGET_TREE_BACKEND_SPECS[backend]
        if spec is not None:
            if self.device.type != "cuda" or self._eager_target_decoder is None:
                raise RuntimeError(
                    "Compiled Target tree verification requires a CUDA decoder"
                )
            if backend == "torch_compile_inductor_reduce_overhead_static":
                # Past-cache length changes every decode round.  A static
                # backend must fail on the first new shape rather than silently
                # exhaust Dynamo's cache and fall back to eager execution.
                torch._dynamo.config.error_on_recompile = True
            if spec["scope"] == "decoder" and backend not in self._compiled_target_decoders:
                compiled = torch.compile(
                    self._eager_target_decoder, mode=spec["mode"],
                    fullgraph=False,
                    dynamic=spec["dynamic"],
                )
                self._compiled_target_decoders[backend] = compiled
                if backend == "torch_compile_inductor_dynamic":
                    self._compiled_target_decoder = compiled
            elif (spec["scope"] == "full_target"
                  and backend not in self._compiled_target_models):
                self._compiled_target_models[backend] = torch.compile(
                    self.target, mode=spec["mode"], fullgraph=False,
                    dynamic=spec["dynamic"],
                )
        self._target_verification_backend = backend

    def tree_proposal_bias(self, path: str | None):
        if path is None:
            return None
        if path not in self._tree_proposal_bias_cache:
            with open(path) as stream:
                payload = json.load(stream)
            values = payload.get("vocab_bias")
            if not isinstance(values, list):
                raise ValueError("Tree calibration artifact has no vocabulary bias")
            bias = torch.tensor(
                values, dtype=torch.float32, device=self.device,
            )
            if bias.shape != (self.target.config.vocab_size,):
                raise ValueError("Tree proposal bias has the wrong vocabulary size")
            self._tree_proposal_bias_cache[path] = bias
        return self._tree_proposal_bias_cache[path]

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _runtime_verifier_identity(
            verifier, parents, tree_tokens, all_p, nodes, output_tokens, bonus,
            *, generator_before, validate) -> dict:
        """Describe one callable and its actual inputs/outputs after it returned."""
        # Keep source inspection and hashing entirely off the timed production
        # path.  A verifier observer is used only by preflight diagnostics.
        import hashlib
        import inspect
        from pathlib import Path

        from .common import digest, file_hash

        source = inspect.getsourcefile(verifier)
        if source is None:
            raise RuntimeError("Cannot resolve the executed verifier source")
        source_path = Path(source).resolve()
        try:
            source_file = str(source_path.relative_to(ROOT.resolve()))
        except ValueError:
            source_file = str(source_path)
        module = getattr(verifier, "__module__", None)
        qualname = getattr(verifier, "__qualname__", None)
        if not isinstance(module, str) or not isinstance(qualname, str):
            raise RuntimeError("Executed verifier lacks a stable Python identity")
        if not isinstance(bonus, int) or isinstance(bonus, bool):
            raise RuntimeError("Executed verifier returned a non-integral bonus token")
        probability_tensor = all_p.detach().cpu().contiguous()
        probability_payload = hashlib.sha256()
        probability_payload.update(str(probability_tensor.dtype).encode())
        probability_payload.update(str(tuple(probability_tensor.shape)).encode())
        probability_payload.update(probability_tensor.numpy().tobytes())
        verifier_input = {
            "tree_sha256":digest([list(parents), list(tree_tokens)]),
            "probabilities_sha256":probability_payload.hexdigest(),
            "probability_shape":list(probability_tensor.shape),
            "probability_dtype":str(probability_tensor.dtype),
            "generator_before":generator_before,
            "validate":validate,
        }
        verifier_output = {
            "accepted_nodes":list(nodes),
            "accepted_tokens":list(output_tokens),
            "bonus_token":bonus,
        }
        return {
            "module":module,
            "qualname":qualname,
            "callable":f"{module}.{qualname}",
            "source_file":source_file,
            "source_sha256":file_hash(source_path),
            "observed_call_index":1,
            "input":verifier_input,
            "input_sha256":digest(verifier_input),
            "output":verifier_output,
            "output_sha256":digest(verifier_output),
        }

    @staticmethod
    def _runtime_generator_identity(generator) -> dict:
        """Hash the verifier's actual generator state before diagnostic dispatch."""
        import hashlib

        state = generator.get_state().detach().cpu().contiguous()
        payload = hashlib.sha256()
        payload.update(str(state.dtype).encode())
        payload.update(str(tuple(state.shape)).encode())
        payload.update(state.numpy().tobytes())
        return {
            "initial_seed":int(generator.initial_seed()),
            "device":str(generator.device),
            "state_sha256":payload.hexdigest(),
        }

    def features(self, hidden_states, rows=None):
        selected = [hidden_states[i + 1] for i in self.draft.target_layer_ids]
        if rows is not None:
            selected = [x.index_select(1, rows) for x in selected]
        return torch.cat(selected, dim=-1)

    def target_forward(self, ids, cache, *, hidden, positions=None, mask=None, last_only=False):
        kwargs = dict(input_ids=ids, past_key_values=cache, use_cache=True,
                      output_hidden_states=hidden, return_dict=True)
        if positions is not None:
            kwargs["position_ids"] = positions
        if mask is not None:
            kwargs["attention_mask"] = mask
        if last_only:
            kwargs["logits_to_keep"] = 1
        return self.target(**kwargs)

    def target_tree_forward(
            self, ids, cache, *, hidden, positions=None, mask=None):
        """Run one Target tree block through the selected verification backend."""
        if self._target_verification_backend == "eager":
            return self.target_forward(
                ids, cache, hidden=hidden, positions=positions, mask=mask,
            )
        spec = TARGET_TREE_BACKEND_SPECS[self._target_verification_backend]
        if spec["scope"] == "full_target":
            kwargs = dict(
                input_ids=ids, past_key_values=cache, use_cache=True,
                output_hidden_states=hidden, return_dict=True,
            )
            if positions is not None:
                kwargs["position_ids"] = positions
            if mask is not None:
                kwargs["attention_mask"] = mask
            return self._compiled_target_models[
                self._target_verification_backend
            ](**kwargs)
        eager_decoder = self.target.model
        decoder = self._compiled_target_decoders.get(
            self._target_verification_backend
        )
        if (decoder is None and self._target_verification_backend
                == "torch_compile_inductor_dynamic"):
            decoder = self._compiled_target_decoder
        if decoder is None:
            raise RuntimeError("Selected Target tree backend was not compiled")
        self.target.model = decoder
        try:
            return self.target_forward(
                ids, cache, hidden=hidden, positions=positions, mask=mask,
            )
        finally:
            self.target.model = eager_decoder

    def target_hidden_forward(self, ids, cache, *, positions=None, mask=None):
        """Run the Target transformer without eagerly applying its LM head."""
        kwargs = dict(
            input_ids=ids, past_key_values=cache, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        if positions is not None:
            kwargs["position_ids"] = positions
        if mask is not None:
            kwargs["attention_mask"] = mask
        spec = TARGET_TREE_BACKEND_SPECS[self._target_verification_backend]
        if spec is None or spec["scope"] != "decoder":
            return self.target.model(**kwargs)
        decoder = self._compiled_target_decoders.get(
            self._target_verification_backend
        )
        if decoder is None:
            raise RuntimeError("Selected Target decoder backend was not compiled")
        return decoder(**kwargs)

    @staticmethod
    def greedy_trace_point(logits, generated_index):
        """Compact evidence for a greedy decision without serializing the vocabulary logits."""
        values = logits.float()
        token_id = int(values.argmax())
        runner_values = values.clone()
        runner_values[token_id] = float("-inf")
        runner_up_token_id = int(runner_values.argmax())
        return {
            "generated_index": generated_index,
            # torch.argmax is also the T=0 decoder's deterministic tie-break.
            # torch.topk may return another tied index first, so it must not be
            # used as the audited token identity.
            "token_id": token_id,
            "runner_up_token_id": runner_up_token_id,
            "top1_margin": float(values[token_id] - values[runner_up_token_id]),
        }

    @torch.inference_mode()
    def generate(self, input_ids, variant: Variant, max_new_tokens: int, stop_ids,
                 seed=0, profile=False, audit_greedy=False, tree_observer=None,
                 shared_suffix_observer=None, atom_observer=None, diffusion_observer=None,
                 ar_observer=None, scaffold_observer=None, verifier_observer=None,
                 lazy_target_observer=None):
        variant.validate()
        if (self._target_verification_backend != "eager"
                and variant.method not in FUSED_TREE_METHODS):
            raise ValueError(
                "Compiled Target verification is gated to fused tree-block methods"
            )
        proposal_bias = self.tree_proposal_bias(
            variant.tree_proposal_bias_path
        )
        if input_ids.shape[0] != 1 or max_new_tokens < 1:
            raise ValueError("Expected one prompt and max_new_tokens >= 1")
        if input_ids.shape[1] + max_new_tokens + self.draft.block_size > self.target.config.max_position_embeddings:
            raise ValueError("Prompt plus generation/verification exceeds model context length; no silent truncation")
        if variant.length >= self.draft.block_size and variant.method != "target":
            raise ValueError("Candidate length exceeds the checkpoint's future slots")
        if audit_greedy and variant.temperature != 0:
            raise ValueError("Greedy logit auditing requires temperature=0")
        if audit_greedy and variant.method in LAZY_TARGET_TREE_METHODS:
            raise ValueError("Lazy Target verification is defined only for T>0")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        host_generator = torch.Generator(device="cpu").manual_seed(
            seed ^ 0x5A17_2026
        )
        dtype = getattr(torch, variant.probability_dtype)
        stops = set(stop_ids)
        target_cache = self.cache_factory()
        draft_cache = self.cache_factory()
        meter = StageMeter(self.device, profile)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sync()
        started = time.perf_counter()
        with meter.measure("prefill"):
            initial = self.target_forward(input_ids, target_cache,
                                          hidden=variant.method != "target", last_only=True)
            if ar_observer is not None:
                # Diagnostic only, before sampling so invalid rows fail closed.
                ar_observer(0, initial.logits[0, -1:])
            anchor = int(sample(probabilities(initial.logits[0, -1], variant.temperature, dtype), generator))
            target_greedy_trace = []
            if audit_greedy and variant.method == "target":
                target_greedy_trace.append(self.greedy_trace_point(initial.logits[0, -1], 0))
            update = self.features(initial.hidden_states) if variant.method != "target" else None
            full_features = update if not variant.reuse_draft_cache else None
            del initial
        self.sync()
        prefill_end = time.perf_counter()
        # The vendored DDTree/DFlash timer resets after the first Draft forward
        # and divides by every returned token, including the anchor sampled by
        # Target prefill. Preserve our stricter end-to-end decode fields below,
        # while recording that published scope independently.
        official_scope_decode_start = (
            prefill_end if variant.method == "target" else None
        )
        generated = [anchor]
        rounds = []
        greedy_audit = []
        target_calls, draft_calls = 1, 0
        target_tokens = input_ids.shape[1]
        verifier_observed = False
        online_rank_bias = (
            torch.zeros(
                (variant.length, min(variant.tree_budget,
                                     self.target.config.vocab_size)),
                dtype=torch.float32, device=self.device,
            )
            if variant.method == "ddtree_online_rank" else None
        )
        history_commit_ewma = None
        while len(generated) < max_new_tokens and generated[-1] not in stops:
            if variant.method == "target":
                with meter.measure("target_decode"):
                    output = self.target_forward(torch.tensor([[generated[-1]]], device=self.device),
                                                 target_cache, hidden=False, last_only=True)
                    if ar_observer is not None:
                        ar_observer(len(generated), output.logits[0, -1:])
                    token = int(sample(probabilities(output.logits[0, -1], variant.temperature, dtype), generator))
                    if audit_greedy:
                        target_greedy_trace.append(
                            self.greedy_trace_point(output.logits[0, -1], len(generated))
                        )
                generated.append(token)
                target_calls += 1
                target_tokens += 1
                continue
            prefix_len = int(target_cache.get_seq_length())
            block_width = int(self.draft.block_size)
            first_draft = draft_calls == 0
            with meter.measure("draft_prefill" if first_draft else "draft"):
                if not variant.reuse_draft_cache:
                    draft_cache = self.cache_factory()
                    context = full_features
                else:
                    context = update
                if int(draft_cache.get_seq_length()) + context.shape[1] != prefix_len:
                    raise RuntimeError("Draft cache and feature-update lengths disagree")
                if variant.condition_features == "zero":
                    context = torch.zeros_like(context)
                noise_ids = torch.full((1, block_width), int(self.draft.mask_token_id),
                                       dtype=torch.long, device=self.device)
                noise_ids[0, 0] = generated[-1]
                noise = self.target.get_input_embeddings()(noise_ids)
                positions = torch.arange(draft_cache.get_seq_length(), prefix_len + block_width,
                                         device=self.device)[None]
                draft_mask = None
                if variant.draft_attention == "causal":
                    draft_mask = torch.zeros((block_width, prefix_len + block_width),
                                             dtype=noise.dtype, device=self.device)
                    forbidden = torch.ones((block_width, block_width), device=self.device,
                                           dtype=torch.bool).triu(1)
                    draft_mask[:, prefix_len:].masked_fill_(forbidden, float("-inf"))
                    draft_mask = draft_mask[None, None]
                hidden = self.draft(target_hidden=context, noise_embedding=noise,
                                    position_ids=positions, attention_mask=draft_mask,
                                    past_key_values=draft_cache, use_cache=True, is_causal=False)
                if not isinstance(hidden, torch.Tensor):
                    hidden = hidden.last_hidden_state
                proposal_hidden = hidden[:, 1:variant.length + 1]
                if variant.method in {
                        "tree_gbv_slot_mixer_recycle", "ddtree_slot_mixer"}:
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            f"{variant.method} requires a loaded proposal adapter"
                        )
                    proposal_hidden = self.proposal_adapter(proposal_hidden)
                if (variant.method == "prefix_beam_tree"
                        and self.proposal_adapter is not None
                        and getattr(
                            self.proposal_adapter,
                            "applies_before_projection", False,
                        )):
                    proposal_hidden = (
                        self.proposal_adapter.correct_proposal_hidden(
                            proposal_hidden, variant.prefix_strength,
                        )
                    )
                logits = self.target.get_output_embeddings()(proposal_hidden)[0]
                if (variant.method == "prefix_beam_tree"
                        and self.proposal_adapter is not None
                        and getattr(
                            self.proposal_adapter,
                            "refines_after_projection", False,
                        ) and variant.prefix_strength > 0):
                    proposal_hidden = (
                        self.proposal_adapter.refine_proposal_hidden(
                            proposal_hidden, logits[None],
                            self.target.get_input_embeddings().weight,
                            variant.prefix_strength,
                        )
                    )
                    logits = self.target.get_output_embeddings()(
                        proposal_hidden,
                    )[0]
                if variant.method == "ddtree_refine":
                    # Parallel prefix-Jacobi refinement: expose only the first
                    # k tentative tokens to the second frozen-DFlash pass and
                    # leave all later slots masked.  Limiting k avoids the
                    # severe all-clean-block distribution shift while still
                    # giving every later slot a coherent tentative prefix in
                    # one additional (non-autoregressive) model forward.
                    draft_cache.crop(prefix_len)
                    refined_ids = noise_ids.clone()
                    refine_length = min(
                        variant.length, variant.diffusion_spur_length,
                    )
                    refined_ids[0, 1:refine_length + 1] = (
                        logits[:refine_length].argmax(-1)
                    )
                    refined_noise = self.target.get_input_embeddings()(
                        refined_ids
                    )
                    refined = self.draft(
                        target_hidden=context,
                        noise_embedding=refined_noise,
                        position_ids=positions,
                        attention_mask=draft_mask,
                        past_key_values=draft_cache,
                        use_cache=True, is_causal=False,
                    )
                    if not isinstance(refined, torch.Tensor):
                        refined = refined.last_hidden_state
                    proposal_hidden = refined[:, 1:variant.length + 1]
                    logits = self.target.get_output_embeddings()(
                        proposal_hidden
                    )[0]
                if variant.method == "tree_gbv_ratio_transport_recycle":
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "tree_gbv_ratio_transport_recycle requires a loaded ratio head"
                        )
                    logits = self.proposal_adapter.correct_logits(
                        proposal_hidden, logits[None],
                        self.target.get_output_embeddings().weight,
                    )[0]
                if proposal_bias is not None:
                    logits = logits.float() + proposal_bias[None]
                draft_cache.crop(prefix_len)
                draft_temp = variant.draft_temperature or variant.temperature or 1.0
                tree_proposal_temp = (
                    variant.tree_proposal_temperature or draft_temp
                )
                if (variant.method in {"dflash", "rank_calibrated_tree"}
                        or variant.method in DIFFUSION_TREE_METHODS):
                    q = None
                elif variant.tree_proposal_temperature_schedule is not None:
                    tree_temperatures = torch.tensor(
                        variant.tree_proposal_temperature_schedule,
                        dtype=dtype, device=logits.device,
                    )
                    q = torch.softmax(
                        logits.to(dtype) / tree_temperatures[:, None], dim=-1,
                    )
                elif variant.tree_proposal_temperature_end is not None:
                    tree_temperatures = torch.linspace(
                        tree_proposal_temp,
                        variant.tree_proposal_temperature_end,
                        logits.shape[0], dtype=dtype, device=logits.device,
                    )
                    q = torch.softmax(
                        logits.to(dtype) / tree_temperatures[:, None], dim=-1,
                    )
                else:
                    q = probabilities(logits, tree_proposal_temp, dtype)
                if variant.method in DIFFUSION_TREE_METHODS:
                    law_length = (
                        variant.diffusion_spur_length
                        if variant.method == "diffusion_core_spur_bv"
                        else variant.length
                    )
                    diffusion_law = diffusion_tree_bv.DiffusionBlockLaw.from_logits(
                        logits[:law_length], noise_ids[0, :law_length + 1],
                        draft_temp, int(self.draft.mask_token_id),
                        support_size=variant.diffusion_support_size)
                    if variant.method == "diffusion_core_spur_bv":
                        q = probabilities(logits, draft_temp, dtype)
                draft_calls += 2 if variant.method == "ddtree_refine" else 1
            if first_draft:
                # This synchronization and reset reproduce the start boundary
                # at ddtree.py:374-377 and dflash.py:80-83. It is deliberately
                # outside the published numerator but remains in decode_ms.
                with meter.measure("first_draft_boundary_sync"):
                    self.sync()
                official_scope_decode_start = time.perf_counter()
            with meter.measure("tree_build"):
                if variant.method == "ddtree_online_rank":
                    tree = online_rank_calibrated_tree(
                        q, variant.tree_budget, online_rank_bias,
                    )
                    paths = None
                    tree_proposal = None
                elif variant.method == "ddtree_history_budget":
                    if history_commit_ewma is None:
                        history_budget = variant.tree_history_warm_budget
                        history_depth = variant.length
                    elif history_commit_ewma >= variant.tree_history_threshold:
                        history_budget = variant.tree_budget
                        history_depth = variant.tree_history_high_depth
                    else:
                        history_budget = variant.tree_history_low_budget
                        history_depth = variant.tree_history_low_depth
                    tree = probability_tree(q[:history_depth], history_budget)
                    paths = None
                    tree_proposal = None
                elif variant.method == "ddtree_confidence_spine":
                    tree = confidence_spine_tree(
                        q, variant.tree_budget,
                        variant.tree_spine_confidence_threshold,
                    )
                    paths = None
                    tree_proposal = None
                elif (variant.method in {
                        "ddtree", "ddtree_refine", "ddtree_slot_mixer"}
                        or variant.method in
                        TERMINAL_TREE_METHODS | FUSED_TREE_METHODS
                        | LAZY_HEAD_TREE_METHODS | LAZY_SOFTMAX_TREE_METHODS
                        | DIRECT_LOGITS_TREE_METHODS | LAZY_TARGET_TREE_METHODS
                        | TREE_BLOCK_SKIP_METHODS):
                    tree = probability_tree(
                        q, variant.tree_budget,
                        depth_reward=variant.tree_depth_reward,
                        depth_reward_schedule=(
                            variant.tree_depth_reward_schedule
                        ),
                        rank_caps=variant.tree_rank_caps,
                        adaptive_min_budget=variant.tree_adaptive_min_budget,
                        confidence_threshold=(
                            variant.tree_adaptive_confidence_threshold
                        ),
                    )
                    paths = None
                    tree_proposal = None
                elif variant.method == "dflash":
                    paths = logits.argmax(-1)[None]
                    tree = sampled_tree(paths)
                    tree_proposal = None
                elif variant.method == "rank_calibrated_tree":
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "rank_calibrated_tree requires a trained rank head"
                        )
                    tree = self.proposal_adapter.build_tree(
                        proposal_hidden, logits, variant.tree_budget,
                        draft_temp,
                    )
                    paths = None
                    tree_proposal = None
                elif variant.method == "ddtree_markov_branch":
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "ddtree_markov_branch requires a trained branch head"
                        )
                    branch_context = {}
                    if getattr(
                            self.proposal_adapter,
                            "requires_anchor_token", False):
                        branch_context["anchor_token_id"] = generated[-1]
                    tree = self.proposal_adapter.build_tree(
                        proposal_hidden, q,
                        self.target.get_output_embeddings().weight,
                        variant.tree_budget, tree_proposal_temp,
                        variant.prefix_strength,
                        **branch_context,
                    )
                    paths = None
                    tree_proposal = None
                elif variant.method == "ddtree_budget_policy":
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "ddtree_budget_policy requires a trained policy"
                        )
                    policy_context = {}
                    if getattr(
                            self.proposal_adapter,
                            "requires_token_embeddings", False):
                        policy_context["token_embeddings"] = (
                            self.target.get_output_embeddings().weight
                        )
                    if getattr(
                            self.proposal_adapter,
                            "requires_input_embeddings", False):
                        policy_context["input_embeddings"] = (
                            self.target.get_input_embeddings().weight
                        )
                    if getattr(
                            self.proposal_adapter,
                            "requires_root_hidden", False):
                        policy_context["root_hidden"] = (
                            self.draft.hidden_norm(self.draft.fc(
                                context[:, -1:],
                            ))[:, 0]
                        )
                    tree = self.proposal_adapter.build_tree(
                        proposal_hidden, q, **policy_context,
                    )
                    paths = None
                    tree_proposal = None
                elif variant.method == "block_aligned_tree":
                    tree = block_aligned_spine_tree(
                        q, variant.tree_budget, variant.paths,
                    )
                    paths = None
                    tree_proposal = None
                elif variant.method in SHARED_SUFFIX_METHODS:
                    paths = root_marginal.propose(q, variant.paths, generator)
                    tree = sampled_tree(paths, variant.share_prefixes)
                    tree_proposal = None
                elif variant.method in PREFIX_RESCORED_TREE_METHODS:
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "prefix_rescored_tree requires a trained prefix head"
                        )
                    tree_builder = (
                        self.proposal_adapter.build_beam_tree
                        if variant.method == "prefix_beam_tree"
                        else self.proposal_adapter.build_tree
                    )
                    architecture_context = {}
                    if getattr(
                            self.proposal_adapter,
                            "requires_target_context", False):
                        architecture_context = {
                            "target_context": self.draft.hidden_norm(
                                self.draft.fc(context)
                            )[:, -1],
                            "anchor_token_id": generated[-1],
                        }
                    if getattr(
                            self.proposal_adapter,
                            "uses_position_probabilities", False):
                        architecture_context["position_probabilities"] = q
                    tree = tree_builder(
                        proposal_hidden, logits,
                        self.target.get_output_embeddings().weight,
                        **({"position_probabilities": q}
                           if variant.method in {
                               "prefix_rescored_tree",
                               "prefix_rescored_tree_fused_scan",
                               "prefix_hybrid_tree",
                           } else {}),
                        budget=variant.tree_budget,
                        temperature=draft_temp,
                        prefix_embeddings=(
                            self.target.get_input_embeddings().weight
                        ),
                        support_size=variant.diffusion_support_size,
                        strength=variant.prefix_strength,
                        **({"pool_factor": variant.prefix_pool_factor}
                           if variant.method in {
                               "prefix_rescored_tree",
                               "prefix_rescored_tree_fused_scan",
                           } else {}),
                        **({"pool_factor": variant.prefix_pool_factor,
                            "core_budget": variant.prefix_core_budget}
                           if variant.method == "prefix_hybrid_tree" else {}),
                        **architecture_context,
                    )
                    paths = None
                    tree_proposal = None
                elif variant.method in PREFIX_CORE_SPUR_METHODS:
                    if self.proposal_adapter is None:
                        raise RuntimeError(
                            "prefix_core_spur_bv requires a trained prefix head"
                        )
                    tree_proposal = self.proposal_adapter.propose(
                        proposal_hidden, logits,
                        self.target.get_output_embeddings().weight,
                        noise_ids[0], variant.diffusion_spur_length,
                        draft_temp, generator,
                        greedy=variant.method == "prefix_core_spur_tree",
                        prefix_embeddings=(
                            self.target.get_input_embeddings().weight
                        ),
                        support_size=variant.diffusion_support_size,
                        strength=variant.prefix_strength,
                    )
                    paths = tree_proposal.paths()
                    tree = diffusion_tree_bv.core_spur_tree(
                        tree_proposal, q, variant.tree_budget,
                    )
                elif variant.method in DIFFUSION_TREE_METHODS:
                    coupling = (
                        "entropy_coalesced"
                        if variant.method in {
                            "diffusion_coalesced_block_skip",
                            "diffusion_entropy_coalesced_bv",
                        }
                        else "aligned"
                        if variant.method == "diffusion_tree_bv_aligned"
                        else "depth_permuted"
                    )
                    tree_proposal = (
                        coalesced_block_skip.propose(
                            diffusion_law, variant.paths, generator,
                        )
                        if variant.method in {
                            "diffusion_coalesced_block_skip",
                            "diffusion_entropy_coalesced_bv",
                        }
                        else diffusion_tree_bv.propose(
                            diffusion_law, variant.paths, generator,
                            coupling=coupling,
                        )
                    )
                    paths = tree_proposal.paths()
                    if variant.method == "diffusion_core_spur_bv":
                        tree = diffusion_tree_bv.core_spur_tree(
                            tree_proposal, q, variant.tree_budget,
                        )
                    elif variant.method in DIFFUSION_SCAFFOLD_METHODS:
                        tree = diffusion_tree_bv.scaffold_tree(
                            tree_proposal, logits.argmax(-1), variant.tree_budget,
                            fill=variant.method != "diffusion_scaffold_no_fill")
                    else:
                        tree = sampled_tree(paths, variant.share_prefixes and variant.method != "diffusion_tree_bv_unmerged")
                elif variant.method in ATOM_TREE_METHODS:
                    coupling = (
                        "aligned" if variant.method == "atom_tree_bv_aligned"
                        else "fixed" if variant.method == "atom_tree_bv_fixed"
                        else "depth_permuted"
                    )
                    tree_proposal = atom_tree_bv.propose(
                        q, variant.paths, generator,
                        coupling=coupling,
                    )
                    paths = tree_proposal.paths()
                    tree = sampled_tree(paths, variant.share_prefixes)
                elif variant.method in {"tree_gbv", "tree_gbv_packed"}:
                    tree_proposal = adaptive_path_proposal(q, variant.tree_budget)
                    paths, proposal_token_probabilities = tree_proposal.sample(variant.paths, generator)
                    tree = sampled_tree(paths, variant.share_prefixes)
                elif variant.method == "tree_gbv_embedded_prefix_recycle":
                    tree = probability_tree(q, variant.tree_budget)
                    tree_proposal = embedded_prefix_proposal(
                        q, tree, variant.diffusion_spur_length,
                    )
                    paths = tree_proposal.paths
                    proposal_token_probabilities = (
                        tree_proposal.token_probabilities
                    )
                elif variant.method in FINITE_TREE_METHODS:
                    if variant.method in {
                        "tree_gbv_budgeted_prefix_recycle",
                        "tree_gbv_budgeted_prefix_recycle_sparse_lazy",
                        "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch",
                        "tree_gbv_budgeted_prefix_recycle_host",
                        "tree_gbv_slot_mixer_recycle",
                        "tree_gbv_ratio_transport_recycle",
                    }:
                        constructor = budgeted_prefix_proposal
                    elif variant.method in {
                        "tree_gbv_prefix", "tree_gbv_prefix_recycle",
                        "tree_gbv_prefix_recycle_packed",
                    }:
                        constructor = adaptive_prefix_proposal
                    else:
                        constructor = adaptive_path_proposal
                    tree_proposal = constructor(q, variant.tree_budget)
                    paths = tree_proposal.paths
                    proposal_token_probabilities = tree_proposal.token_probabilities
                    tree = sampled_tree(paths, variant.share_prefixes)
                else:
                    paths = sample(q[None].expand(variant.paths, -1, -1), generator)
                    tree = sampled_tree(paths, variant.share_prefixes)
                    tree_proposal = None
            with meter.measure("tree_compile"):
                if variant.method in TREE_BLOCK_SKIP_METHODS:
                    cost_aware_decision = None
                    if variant.method in COST_AWARE_TREE_BLOCK_SKIP_METHODS:
                        cost_aware_decision = cost_aware_tree_block(
                            tree.parents, tree.tokens, tree.depths,
                            tree.draft_prefix_log_masses,
                            H20_QWEN3_4B_T1_TARGET_LATENCY_MS,
                        )
                        tree_block_row_cap = None
                        first_block = cost_aware_decision.block
                    else:
                        tree_block_row_cap = (
                            risk_bounded_prefix_cap(
                                tree.parents, tree.draft_prefix_log_masses,
                                risk_budget=.05, minimum_rows=16,
                            )
                            if variant.method in CONDITIONAL_TREE_BLOCK_SKIP_METHODS
                            else TREE_BLOCK_SKIP_ROWS[variant.method]
                        )
                        first_block = local_tree_block(
                            tree.parents, tree.tokens, tree.depths, 0,
                            tree_block_row_cap,
                        )
                    verify_tree = Tree(
                        tokens=list(first_block.tokens),
                        parents=list(first_block.parents),
                        depths=[depth - first_block.depths[0]
                                for depth in first_block.depths],
                        path_nodes=[],
                    )
                elif variant.method in LAZY_TARGET_TREE_METHODS:
                    # A posterior row is needed to choose an outgoing edge.
                    # Verify the ancestor-closed internal subtree plus a small
                    # fixed set of high-Draft-mass leaves; defer only an
                    # unprefetched leaf that the realized walk reaches.
                    internal_nodes = sorted(set(tree.parents[1:]))
                    internal_node_set = set(internal_nodes)
                    leaf_nodes = [
                        node for node in range(1, len(tree.parents))
                        if node not in internal_node_set
                    ]
                    # probability_tree assigns node ids in descending Draft
                    # prefix-mass order, so the earliest leaves are the safest
                    # no-synchronization prefetch choices.
                    prefetch_control = LAZY_TARGET_TREE_PREFETCH[
                        variant.method
                    ]
                    prefetch_count = (
                        prefetch_control
                        if prefetch_control >= 0
                        else max(0, -prefetch_control - len(internal_nodes))
                    )
                    prefetched_leaf_nodes = leaf_nodes[:prefetch_count]
                    verified_nodes = sorted(
                        internal_nodes + prefetched_leaf_nodes
                    )
                    verified_node_to_compact = {
                        node: row for row, node in enumerate(verified_nodes)
                    }
                    sparse_target_tree = Tree(
                        tokens=[tree.tokens[node - 1]
                                for node in verified_nodes[1:]],
                        parents=[-1] + [
                            verified_node_to_compact[tree.parents[node]]
                            for node in verified_nodes[1:]
                        ],
                        depths=[tree.depths[node] for node in verified_nodes],
                        path_nodes=[],
                    )
                    verify_tree = sparse_target_tree
                else:
                    verify_tree = tree
                ids = torch.tensor(
                    [[generated[-1]] + verify_tree.tokens], device=self.device
                )
                positions = (
                    torch.tensor(verify_tree.depths, device=self.device)
                    + prefix_len
                )[None]
                mask = verify_tree.mask(
                    prefix_len, next(self.target.parameters()).dtype, self.device
                )
                lazy_lm_head = (
                    variant.method in LAZY_HEAD_TREE_METHODS
                    or variant.method == "prefix_core_spur_bv_lazy"
                    or (
                        variant.method in DIFFUSION_SCAFFOLD_METHODS
                        and variant.method != "diffusion_core_spur_bv"
                    )
                )
            with meter.measure("verify"):
                packed_cache = None
                cascade_hidden_states = None
                cascade_block_roots = []
                cascade_block_rows = []
                if variant.method in TREE_BLOCK_SKIP_METHODS:
                    root_node = 0
                    cascade_nodes = []
                    cascade_tokens = []
                    hidden_chunks = None
                    while True:
                        if root_node == 0:
                            block = first_block
                        elif variant.method in COST_AWARE_TREE_BLOCK_SKIP_METHODS:
                            # A skipped edge pays at most one rescue call: once
                            # reached, verify its complete remaining subtree.
                            block = local_tree_block(
                                tree.parents, tree.tokens, tree.depths,
                                root_node, len(tree.parents),
                            )
                        else:
                            block = local_tree_block(
                                tree.parents, tree.tokens, tree.depths,
                                root_node, tree_block_row_cap,
                            )
                        cascade_block_roots.append(root_node)
                        cascade_block_rows.append(len(block.nodes))
                        local_tree = Tree(
                            tokens=list(block.tokens),
                            parents=list(block.parents),
                            depths=[depth - block.depths[0]
                                    for depth in block.depths],
                            path_nodes=[],
                        )
                        cache_start = int(target_cache.get_seq_length())
                        block_ids = torch.tensor(
                            [[generated[-1] if root_node == 0
                              else tree.tokens[root_node - 1]]
                             + local_tree.tokens],
                            dtype=torch.long, device=self.device,
                        )
                        block_positions = torch.tensor(
                            block.depths, dtype=torch.long, device=self.device,
                        )[None] + prefix_len
                        block_mask = local_tree.mask(
                            cache_start,
                            next(self.target.parameters()).dtype,
                            self.device,
                        )
                        block_output = self.target_hidden_forward(
                            block_ids, target_cache,
                            positions=block_positions, mask=block_mask,
                        )
                        block_logits = self.target.get_output_embeddings()(
                            block_output.last_hidden_state[0]
                        )
                        block_p = probabilities(
                            block_logits, variant.temperature, dtype,
                        )
                        local_nodes, local_tokens, bonus = (
                            tree_block_verify_terminal_mass(
                                local_tree.parents, local_tree.tokens,
                                block_p, generator, validate=False,
                                prefix_mode="batched", exit_mode="internal",
                            )
                        )
                        mapped_nodes = [
                            block.nodes[node] for node in local_nodes
                        ]
                        local_keep = [0] + local_nodes
                        hidden_index = torch.tensor(
                            local_keep, dtype=torch.long, device=self.device,
                        )
                        chosen_hidden = tuple(
                            layer.index_select(1, hidden_index)
                            for layer in block_output.hidden_states
                        )
                        if hidden_chunks is None:
                            hidden_chunks = [
                                [layer] for layer in chosen_hidden
                            ]
                        else:
                            for chunks, layer in zip(
                                    hidden_chunks, chosen_hidden):
                                chunks.append(layer)
                        compact_cache(
                            target_cache, cache_start, local_keep, self.device,
                        )
                        target_calls += 1
                        target_tokens += block_ids.shape[1]
                        if root_node:
                            cascade_nodes.append(root_node)
                            cascade_tokens.append(tree.tokens[root_node - 1])
                        cascade_nodes.extend(mapped_nodes)
                        cascade_tokens.extend(local_tokens)
                        terminal_node = (
                            mapped_nodes[-1] if mapped_nodes else root_node
                        )
                        continuation = child_for_edge(
                            tree.parents, tree.tokens,
                            terminal_node, int(bonus),
                        )
                        del (block_output, block_logits, block_p,
                             block_ids, block_positions, block_mask,
                             chosen_hidden, hidden_index, local_tree)
                        if continuation is None:
                            break
                        if continuation in block.nodes:
                            raise AssertionError(
                                "Terminal block exited through a verified edge"
                            )
                        root_node = continuation
                    if hidden_chunks is None:
                        raise AssertionError("Cascade produced no Target block")
                    cascade_hidden_states = tuple(
                        torch.cat(chunks, dim=1) for chunks in hidden_chunks
                    )
                    nodes, tokens = cascade_nodes, cascade_tokens
                    accepted = len(nodes)
                    output = None
                    all_p = None
                elif variant.method in LAZY_TARGET_TREE_METHODS:
                    output = self.target_hidden_forward(
                        ids, target_cache, positions=positions, mask=mask
                    )
                    verified_logits = self.target.get_output_embeddings()(
                        output.last_hidden_state[0]
                    )
                    verified_p = probabilities(
                        verified_logits, variant.temperature, dtype
                    )
                    internal_compact_indices = torch.tensor(
                        [verified_node_to_compact[node]
                         for node in internal_nodes],
                        device=self.device,
                    )
                    internal_p = verified_p.index_select(
                        0, internal_compact_indices
                    )
                    all_p = None
                    target_tokens += ids.shape[1]
                elif variant.method in PACKED_TREE_METHODS:
                    # Verify complete support paths as a regular causal batch.
                    # This duplicates shared tree nodes, but avoids the arbitrary
                    # 4-D tree mask and lets SDPA select its fast causal kernel.
                    packed_cache = self.cache_factory()
                    for layer_index, cached_layer in enumerate(target_cache.layers):
                        packed_cache.update(
                            cached_layer.keys.expand(paths.shape[0], -1, -1, -1),
                            cached_layer.values.expand(paths.shape[0], -1, -1, -1),
                            layer_index,
                        )
                    packed_ids = torch.cat((
                        torch.full((paths.shape[0], 1), generated[-1],
                                   dtype=torch.long, device=self.device),
                        paths,
                    ), dim=1)
                    packed_positions = torch.arange(
                        prefix_len, prefix_len + packed_ids.shape[1],
                        device=self.device,
                    )[None].expand(paths.shape[0], -1)
                    output = self.target_forward(
                        packed_ids, packed_cache, hidden=True,
                        positions=packed_positions,
                    )
                    packed_p = probabilities(output.logits, variant.temperature, dtype)
                    node_sources = [None] * len(tree.parents)
                    node_sources[0] = (0, 0)
                    for leaf, leaf_nodes in enumerate(tree.path_nodes):
                        for depth, node in enumerate(leaf_nodes, start=1):
                            if node_sources[node] is None:
                                node_sources[node] = (leaf, depth)
                    all_p = torch.stack([
                        packed_p[leaf, depth]
                        for leaf, depth in node_sources
                    ])
                    target_tokens += packed_ids.numel()
                else:
                    if lazy_lm_head:
                        output = self.target_hidden_forward(
                            ids, target_cache, positions=positions, mask=mask
                        )
                        all_p = None
                    else:
                        output = self.target_tree_forward(
                            ids, target_cache, hidden=True,
                            positions=positions, mask=mask,
                        )
                        # Diffusion BV only needs the labelled proposal rows up
                        # front.  Its scaffold continuation normalizes the
                        # reached subtree lazily, so materializing FP64
                        # full-vocabulary probabilities for every tree row here
                        # wastes bandwidth without changing the sampling law.
                        all_p = (None if variant.method in (ATOM_TREE_METHODS - {"atom_tree_ancestral"})
                                 | (DIFFUSION_TREE_METHODS - {"diffusion_tree_ancestral"})
                                 | {"prefix_core_spur_bv"}
                                 | LAZY_SOFTMAX_TREE_METHODS | DIRECT_LOGITS_TREE_METHODS
                                 else probabilities(output.logits[0], variant.temperature, dtype))
                    target_tokens += ids.shape[1]
                if variant.method not in TREE_BLOCK_SKIP_METHODS:
                    target_calls += 1
            if tree_observer is not None:
                # Only diagnostic runs attach an observer. Captures and their
                # synchronization must never contaminate primary throughput.
                if (variant.method in LAZY_TARGET_TREE_METHODS
                        or variant.method in TREE_BLOCK_SKIP_METHODS):
                    raise ValueError(
                        "Tree observer does not support partial Target schedules"
                    )
                tree_observer(tree.parents, tree.tokens,
                              all_p if all_p is not None else probabilities(
                                  self.target.get_output_embeddings()(
                                      output.last_hidden_state[0]
                                  ) if lazy_lm_head
                                  else output.logits[0],
                                  variant.temperature, dtype,
                              ))
            if shared_suffix_observer is not None and variant.method in SHARED_SUFFIX_METHODS:
                # Diagnostic-only capture includes the ACTUAL proposal Q;
                # marginal verifier replay cannot be reconstructed from p alone.
                shared_suffix_observer(tree.parents, tree.tokens, paths, all_p, q)
            if atom_observer is not None and variant.method in ATOM_TREE_METHODS:
                # Diagnostic-only: preserve the latent law, not only its marginals.
                atom_observer(tree.parents, tree.tokens, tree_proposal, output.logits[0], variant.temperature)
            if diffusion_observer is not None and variant.method in DIFFUSION_TREE_METHODS:
                diffusion_observer(
                    tree, tree_proposal,
                    (self.target.get_output_embeddings()(output.last_hidden_state[0])
                     if lazy_lm_head else output.logits[0]),
                    variant.temperature,
                )
            executed_verifier = None
            terminal_leaf = -1
            prefetched_leaf_hit = False
            fallback_leaf_forward = False
            deferred_leaf_stop = False
            append_bonus = True
            target_cache_precompacted = variant.method in TREE_BLOCK_SKIP_METHODS
            with meter.measure("select_and_correct"):
                if variant.method in TREE_BLOCK_SKIP_METHODS:
                    # The cascade already sampled and mapped every reached
                    # block.  This stage is intentionally empty so its host
                    # synchronization cost remains attributed to verification.
                    pass
                elif variant.method in RECYCLE_TREE_METHODS:
                    node_paths = torch.tensor(
                        tree.path_nodes, dtype=torch.long, device=self.device
                    )
                    recycle_children = {
                        (tree.parents[i], tree.tokens[i - 1]): i
                        for i in range(1, len(tree.parents))
                    }
                    nodes, tokens, bonus, recycle_stats = tree_block_verify_recycle(
                        paths, node_paths, recycle_children, all_p,
                        tree_proposal.leaf_probabilities,
                        proposal_token_probabilities, variant.paths, generator,
                        # All recycle verifier variants share the same exact
                        # block-verification law; this switch chooses whether
                        # to use dense per-segment BV rows or deferred-tails
                        # lazy sampling.
                        segment_verifier=(
                            "sparse_lazy"
                            if variant.method in {
                                "tree_gbv_budgeted_prefix_recycle_sparse_lazy",
                                "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch",
                            }
                            else "sparse" if variant.method in {
                                "tree_gbv_prefix_recycle",
                                "tree_gbv_budgeted_prefix_recycle",
                                "tree_gbv_budgeted_prefix_recycle_host",
                                "tree_gbv_slot_mixer_recycle",
                                "tree_gbv_ratio_transport_recycle",
                                "tree_gbv_prefix_recycle_packed",
                            } else "sparse_lazy"
                        ),
                        control_device=(
                            "cpu" if variant.method in HOST_RECYCLE_METHODS else "device"
                        ),
                        host_generator=host_generator,
                        precompute_subtrees=(
                            variant.method == "tree_gbv_budgeted_prefix_recycle_sparse_lazy_prefetch"
                        ),
                    )
                    accepted = len(nodes)
                elif variant.method in TERMINAL_TREE_METHODS:
                    nodes, tokens, bonus = tree_block_verify_terminal_mass(
                        # The shared FP64 softmax constructs these probability
                        # rows; topology checks remain enabled in the kernel.
                        tree.parents, tree.tokens, all_p, generator,
                        validate=False,
                        prefix_mode=("serial" if variant.method == "ddtree_terminal_serial"
                                     else "batched"),
                        exit_mode=("dense" if variant.method == "ddtree_terminal_dense"
                                   else "internal"),
                    )
                    accepted = len(nodes)
                elif variant.method == "prefix_rescored_tree_fused_scan":
                    # The scan kernel is CUDA-only.  Keep CPU reference tests
                    # exact by falling back to the batched verifier; production
                    # GPU benchmarks still exercise the fused implementation.
                    verifier = (
                        tree_verify_ancestral_fused_scan
                        if all_p.is_cuda
                        else tree_verify_ancestral_batched
                    )
                    generator_before = (
                        self._runtime_generator_identity(generator)
                        if verifier_observer is not None and not verifier_observed
                        else None
                    )
                    nodes, tokens, bonus = verifier(
                        tree.parents, tree.tokens, all_p,
                        generator, validate=False,
                    )
                    accepted = len(nodes)
                    if generator_before is not None:
                        executed_verifier = (verifier, generator_before)
                elif variant.method in FUSED_TREE_METHODS:
                    verifier = {
                        "ddtree_compiled_batched": tree_verify_ancestral_batched,
                        "ddtree_fused": tree_verify_ancestral_fused,
                        "ddtree_fused_parallel": tree_verify_ancestral_fused_parallel,
                        "ddtree_fused_scan": tree_verify_ancestral_fused_scan,
                        "ddtree_sparse_exit_fused_scan": (
                            tree_verify_ancestral_sparse_exit_fused_scan
                        ),
                        "ddtree_same_draw_fused": tree_verify_ancestral_same_draw_fused,
                    }[variant.method]
                    generator_before = (
                        self._runtime_generator_identity(generator)
                        if verifier_observer is not None and not verifier_observed
                        else None
                    )
                    nodes, tokens, bonus = verifier(
                        tree.parents, tree.tokens, all_p, generator,
                        validate=False,
                    )
                    accepted = len(nodes)
                    if generator_before is not None:
                        executed_verifier = (verifier, generator_before)
                elif variant.method in LAZY_HEAD_TREE_METHODS:
                    lazy_verifier = (
                        tree_verify_ancestral_lazy_projection_fused_scan
                        if variant.method == "ddtree_lazy_projection_fused_scan"
                        else tree_verify_ancestral_lazy_projection
                    )
                    nodes, tokens, bonus, lazy_projection_stats = lazy_verifier(
                        tree.parents, tree.tokens,
                        output.last_hidden_state[0],
                        self.target.get_output_embeddings(),
                        variant.temperature, dtype, generator,
                        validate=False,
                    )
                    accepted = len(nodes)
                elif variant.method in LAZY_SOFTMAX_TREE_METHODS:
                    nodes, tokens, bonus, lazy_projection_stats = (
                        tree_verify_ancestral_lazy_softmax_fused_scan(
                            tree.parents, tree.tokens, output.logits[0],
                            variant.temperature, dtype, generator,
                            validate=False,
                        )
                    )
                    accepted = len(nodes)
                elif variant.method in DIRECT_LOGITS_TREE_METHODS:
                    nodes, tokens, bonus, direct_logits_stats = (
                        tree_verify_ancestral_logits_fused_scan(
                            tree.parents, tree.tokens, output.logits[0].contiguous(),
                            variant.temperature, dtype, generator,
                            validate=False,
                        )
                    )
                    accepted = len(nodes)
                elif variant.method in PARTIAL_TERMINAL_TREE_METHODS:
                    # The verified rows form an ancestor-closed ticket tree.
                    # A terminal block draw samples its first Target exit in
                    # one operation.  An edge omitted from the ticket tree is
                    # emitted as the bonus and ends the round; descendants of
                    # an unverified node are never used.
                    (compact_nodes, tokens,
                     bonus) = tree_block_verify_terminal_mass(
                        sparse_target_tree.parents,
                        sparse_target_tree.tokens,
                        verified_p, generator, validate=False,
                        prefix_mode="batched", exit_mode="internal",
                    )
                    nodes = [verified_nodes[node] for node in compact_nodes]
                    terminal_leaf = -1
                    lazy_target_stats = {
                        "internal_probability_rows": len(verified_nodes),
                        "leaf_probability_rows": 0,
                    }
                    accepted = len(nodes)
                elif variant.method in LAZY_TARGET_TREE_METHODS:
                    (nodes, tokens, bonus, terminal_leaf,
                     lazy_target_stats) = tree_verify_internal_ancestral_batched(
                        tree.parents, tree.tokens, internal_p, generator,
                        validate=False,
                    )
                    accepted = len(nodes)
                elif variant.method in {
                        "ddtree", "ddtree_refine", "ddtree_online_rank",
                        "root_shared_ddtree",
                        "atom_tree_ancestral",
                        "diffusion_tree_ancestral", "prefix_core_spur_tree",
                        "rank_calibrated_tree",
                        "block_aligned_tree",
                        "ddtree_slot_mixer",
                        "ddtree_markov_branch",
                        "ddtree_budget_policy",
                        "ddtree_history_budget",
                        "ddtree_confidence_spine",
                        "prefix_sampled_spur_tree"} | PREFIX_RESCORED_TREE_METHODS:
                    verifier = tree_verify_ancestral_batched
                    generator_before = (
                        self._runtime_generator_identity(generator)
                        if verifier_observer is not None and not verifier_observed
                        else None
                    )
                    nodes, tokens, bonus = verifier(
                        tree.parents, tree.tokens, all_p, generator,
                        validate=False,
                    )
                    accepted = len(nodes)
                    if generator_before is not None:
                        executed_verifier = (verifier, generator_before)
                elif variant.method in (
                        {"diffusion_core_spur_bv"} | PREFIX_CORE_SPUR_METHODS):
                    if variant.method == "prefix_core_spur_bv_lazy":
                        nodes, tokens, bonus = (
                            diffusion_tree_bv.verify_scaffold_hidden(
                                output.last_hidden_state[0],
                                self.target.get_output_embeddings(),
                                tree, tree_proposal, variant.temperature,
                                generator, recycle=True,
                                continuation="ancestral", validate=False,
                            )
                        )
                    else:
                        nodes, tokens, bonus = (
                            diffusion_tree_bv.verify_scaffold_logits(
                                output.logits[0], tree, tree_proposal,
                                variant.temperature, generator,
                                continuation="ancestral", validate=False,
                                node_probabilities=all_p,
                            )
                        )
                    accepted = len(nodes)
                elif variant.method in DIFFUSION_SCAFFOLD_METHODS:
                    nodes, tokens, bonus = diffusion_tree_bv.verify_scaffold_hidden(
                        output.last_hidden_state[0],
                        self.target.get_output_embeddings(),
                        tree, tree_proposal, variant.temperature, generator,
                        recycle=variant.method != "diffusion_scaffold_no_recycle",
                        continuation=("ancestral"
                                      if variant.method in {
                                          "diffusion_scaffold_ancestral",
                                          "diffusion_core_spur_bv",
                                      }
                                      else "terminal"),
                        validate=False,
                    )
                    accepted = len(nodes)
                elif variant.method in DIFFUSION_TREE_METHODS:
                    if variant.method in {
                        "diffusion_coalesced_block_skip",
                        "diffusion_coalesced_flow_bv",
                    }:
                        nodes, tokens, bonus, coalesced_state = (
                            coalesced_block_skip.verify_logits(
                                output.logits[0], tree, tree_proposal,
                                variant.temperature, generator, validate=False,
                            )
                        )
                    else:
                        nodes, tokens, bonus = diffusion_tree_bv.verify_logits(
                            output.logits[0], tree, tree_proposal, variant.temperature, generator,
                            pool=variant.method != "diffusion_tree_bv_no_pool", validate=False)
                    accepted = len(nodes)
                elif variant.method in ATOM_TREE_METHODS:
                    # Distinct first tokens keep complete branches contiguous.
                    # Do not materialize all branch/depth probability or residual rows.
                    chosen, accepted_suffix, bonus = atom_tree_bv.verify_logits(
                        output.logits[0, 0],
                        output.logits[0, 1:].view(variant.paths, variant.length, -1),
                        tree_proposal, variant.temperature, generator,
                        pool=variant.method != "atom_tree_bv_no_pool", validate=False,
                    )
                    nodes = [] if chosen == -1 else tree.path_nodes[chosen][:accepted_suffix + 1]
                    tokens = [tree.tokens[node - 1] for node in nodes]
                    accepted = len(nodes)
                elif variant.method in SHARED_SUFFIX_METHODS:
                    # Distinct roots imply no cross-branch prefix sharing.
                    # sampled_tree stores complete branches consecutively, so
                    # use a view instead of copying a K x L x V probability tensor.
                    branch_p = all_p[1:].view(variant.paths, variant.length, -1)
                    verifier = (
                        protected_tree_bv.verify if variant.method == "root_protected_bv"
                        else root_marginal.verify if variant.method == "root_marginal_bv_ref"
                        else root_marginal.verify_early_root if variant.method == "root_early_bv"
                        else root_marginal.verify_tensorized
                    )
                    options = {"rule": "token"} if variant.method == "root_marginal_token" else {}
                    chosen, accepted_suffix, bonus = verifier(
                        paths[:, 0], all_p[0], paths[0, 1:], branch_p, q[1:],
                        generator, validate=False, **options,
                    )
                    nodes = [] if chosen == -1 else tree.path_nodes[chosen][:accepted_suffix + 1]
                    tokens = [tree.tokens[node - 1] for node in nodes]
                    accepted = len(nodes)
                    del branch_p
                else:
                    if variant.method in SPARSE_FULL_TREE_METHODS:
                        # Selecting a finite-tree leaf needs only the Target
                        # probability of each proposed token.  Avoid building a
                        # leaves x length x vocabulary tensor before one leaf is
                        # selected; materialize dense Target rows only for it.
                        target_row_nodes = torch.tensor(
                            [[0] + nodes[:-1] for nodes in tree.path_nodes],
                            dtype=torch.long, device=self.device,
                        )
                        target_token_probabilities = all_p[
                            target_row_nodes, paths
                        ]
                        selected_leaf_probabilities = greedy_max_distribution(
                            paths, target_token_probabilities, proposal_token_probabilities,
                            tree_proposal.leaf_probabilities, variant.paths,
                        )
                        chosen = int(sample(selected_leaf_probabilities, generator).item())
                        chosen_nodes = tree.path_nodes[chosen]
                        chosen_p = all_p[[0] + chosen_nodes]
                        proposal_tokens, proposal_probabilities = tree_proposal.conditional_sparse(
                            paths[chosen], selected_leaf_probabilities
                        )
                        sparse_verifier = (
                            block_verify_sparse_lazy
                            if variant.method == "tree_gbv_full_sparse_lazy"
                            else block_verify_sparse
                        )
                        accepted, bonus = sparse_verifier(
                            paths[chosen], chosen_p, proposal_tokens,
                            proposal_probabilities, generator,
                        )
                    else:
                        p_by_path = torch.stack([all_p[[0] + nodes]
                                                 for nodes in tree.path_nodes])
                    if variant.method == "dflash":
                        chosen, r = 0, None
                        accepted, bonus = matching_verify(paths[0], p_by_path[0], generator)
                    elif variant.method == "gbv":
                        chosen, r = select_and_reweight(paths, p_by_path, q)
                    elif variant.method in {"tree_gbv", "tree_gbv_packed"}:
                        target_token_probabilities = p_by_path[:, :-1].gather(
                            2, paths[:, :, None]
                        )[:, :, 0]
                        chosen = select_greedy_path(
                            paths, target_token_probabilities, proposal_token_probabilities
                        )
                        proposal_rows = tree_proposal.conditional_rows(paths[chosen])
                        r = reweight_selected_path(
                            paths[chosen], p_by_path[chosen], proposal_rows, variant.paths
                        )
                    elif variant.method == "tree_gbv_full_dense_ref":
                        target_token_probabilities = p_by_path[:, :-1].gather(
                            2, paths[:, :, None]
                        )[:, :, 0]
                        selected_leaf_probabilities = greedy_max_distribution(
                            paths, target_token_probabilities, proposal_token_probabilities,
                            tree_proposal.leaf_probabilities, variant.paths,
                        )
                        chosen = int(sample(selected_leaf_probabilities, generator).item())
                        r = tree_proposal.conditional_rows(
                            paths[chosen], selected_leaf_probabilities
                        )
                    elif variant.method not in SPARSE_FULL_TREE_METHODS:
                        chosen, r = 0, q
                    if variant.method not in {"dflash", *SPARSE_FULL_TREE_METHODS}:
                        verifier = token_verify if variant.method == "token" else block_verify_batched
                        accepted, bonus = verifier(paths[chosen], p_by_path[chosen], r, generator)
                    nodes = tree.path_nodes[chosen][:accepted]
                    tokens = paths[chosen, :accepted].tolist()
            if (terminal_leaf >= 0
                    and terminal_leaf in verified_node_to_compact):
                # A prefetched leaf already has its posterior row in this
                # round's aligned Target block, so retain DDTree's exact bonus
                # without another Target invocation.
                with meter.measure("prefetched_leaf_sample"):
                    leaf_p = verified_p[
                        verified_node_to_compact[terminal_leaf]
                    ]
                    bonus = int(sample(leaf_p, generator))
                prefetched_leaf_hit = True
            elif (terminal_leaf >= 0
                    and variant.method in DEFERRED_LEAF_TREE_METHODS):
                # Do not spend a separate Target call only to sample the leaf
                # continuation.  Commit the leaf as this round's last token;
                # the next round verifies that leaf as its root and samples
                # exactly the same autoregressive continuation distribution.
                bonus = -1
                append_bonus = False
                deferred_leaf_stop = True
            elif terminal_leaf >= 0:
                # The reached leaf is the only omitted tree row that can affect
                # the output.  Commit the already accepted internal path to the
                # cache, then verify that one leaf token as a regular cached
                # Target step and sample its exact continuation distribution.
                internal_path_nodes = nodes[:-1]
                compact_verified_keep = [verified_node_to_compact[0]] + [
                    verified_node_to_compact[node]
                    for node in internal_path_nodes
                ]
                with meter.measure("lazy_leaf_cache_compact"):
                    compact_cache(
                        target_cache, prefix_len, compact_verified_keep,
                        self.device,
                    )
                leaf_token = tree.tokens[terminal_leaf - 1]
                with meter.measure("lazy_leaf_verify"):
                    leaf_output = self.target_forward(
                        torch.tensor([[leaf_token]], device=self.device),
                        target_cache, hidden=True,
                        positions=torch.tensor(
                            [[prefix_len + tree.depths[terminal_leaf]]],
                            device=self.device,
                        ),
                        last_only=True,
                    )
                    target_calls += 1
                    target_tokens += 1
                with meter.measure("lazy_leaf_sample"):
                    leaf_p = probabilities(
                        leaf_output.logits[0, -1], variant.temperature, dtype
                    )
                    bonus = int(sample(leaf_p, generator))
                target_cache_precompacted = True
                fallback_leaf_forward = True
            if (lazy_target_observer is not None
                    and variant.method in LAZY_TARGET_TREE_METHODS):
                # Diagnostic-only capture.  Tensor copies/synchronization are
                # deliberately absent from primary throughput runs.
                lazy_target_observer(
                    tree.parents, tree.tokens, internal_nodes, internal_p,
                    terminal_leaf,
                    leaf_p if terminal_leaf >= 0 and not deferred_leaf_stop
                    else None,
                )
            if executed_verifier is not None:
                # This is deliberately after the selected callable returned.
                # The normal timing path has no observer, performs no source
                # inspection, and introduces no additional CUDA synchronization.
                verifier, generator_before = executed_verifier
                verifier_observer(self._runtime_verifier_identity(
                    verifier, tree.parents, tree.tokens, all_p,
                    nodes, tokens, bonus,
                    generator_before=generator_before, validate=False,
                ))
                verifier_observed = True
            if scaffold_observer is not None and variant.method in DIFFUSION_SCAFFOLD_METHODS:
                # Diagnostic-only witness of the actual draft and selected exit.
                # Tree logits alone cannot certify fixed greedy coverage or
                # maximal continuation, the premises of the DFlash tail bound.
                scaffold_observer(tree, tree_proposal, logits, nodes, tokens, bonus)
            if variant.method == "ddtree_online_rank":
                with meter.measure("tree_adapt"):
                    online_rank_bias = update_online_rank_bias(
                        q, tree, all_p, online_rank_bias,
                        variant.tree_online_ewma, variant.tree_online_clip,
                    )
            if audit_greedy and variant.method != "target":
                row_index = torch.tensor([0] + nodes, device=self.device)
                tree_logits = (
                    self.target.get_output_embeddings()(
                        output.last_hidden_state[0].index_select(0, row_index)
                    )
                    if lazy_lm_head
                    else output.logits[0].index_select(0, row_index)
                )
                # Match the production AR baseline exactly: prompt prefill followed
                # by one cached token at a time. A full-sequence forward may use a
                # different SDPA kernel and is not the baseline being audited.
                sequential_cache = self.cache_factory()
                self.target_forward(input_ids, sequential_cache, hidden=False, last_only=True)
                sequential_rows = []
                for audit_token in generated + tokens:
                    audit_output = self.target_forward(
                        torch.tensor([[audit_token]], device=self.device), sequential_cache,
                        hidden=False, last_only=True
                    )
                    sequential_rows.append(audit_output.logits[0, -1])
                sequential_logits = torch.stack(sequential_rows[-(len(tokens) + 1):])
                tree_argmax = tree_logits.argmax(-1)
                sequential_argmax = sequential_logits.argmax(-1)
                tree_top2 = tree_logits.float().topk(2, dim=-1)
                sequential_top2 = sequential_logits.float().topk(2, dim=-1)
                greedy_audit.append({
                    "generated_start_index": len(generated),
                    "tree_token_ids": tree_argmax.tolist(),
                    "sequential_token_ids": sequential_argmax.tolist(),
                    "argmax_equal": tree_argmax.eq(sequential_argmax).tolist(),
                    "max_absolute_logit_error": (tree_logits.float() - sequential_logits.float()).abs().amax(-1).tolist(),
                    "tree_top1_margin": (tree_top2.values[:, 0] - tree_top2.values[:, 1]).tolist(),
                    "sequential_top1_margin": (sequential_top2.values[:, 0] - sequential_top2.values[:, 1]).tolist(),
                })
            with meter.measure("stop_check"):
                appended = tokens + ([bonus] if append_bonus else [])
                committed = appended[:max_new_tokens - len(generated)]
                for i, token in enumerate(committed):
                    if token in stops:
                        committed = committed[:i + 1]
                        break
            with meter.measure("commit"):
                generated.extend(committed)
                keep = [0] + nodes
                index = torch.tensor(keep, device=self.device)
                if variant.method in TREE_BLOCK_SKIP_METHODS:
                    update = self.features(cascade_hidden_states)
                elif variant.method in LAZY_TARGET_TREE_METHODS:
                    accepted_verified_nodes = (
                        nodes[:-1]
                        if fallback_leaf_forward or deferred_leaf_stop
                        else nodes
                    )
                    compact_verified_keep = [verified_node_to_compact[0]] + [
                        verified_node_to_compact[node]
                        for node in accepted_verified_nodes
                    ]
                    feature_index = torch.tensor(
                        compact_verified_keep, device=self.device
                    )
                    if fallback_leaf_forward:
                        selected_hidden = tuple(
                            torch.cat((
                                layer.index_select(1, feature_index),
                                leaf_layer,
                            ), dim=1)
                            for layer, leaf_layer in zip(
                                output.hidden_states,
                                leaf_output.hidden_states,
                            )
                        )
                        update = self.features(selected_hidden)
                    else:
                        update = self.features(
                            output.hidden_states, feature_index
                        )
                elif variant.method in PACKED_TREE_METHODS:
                    cache_leaf = next(
                        leaf for leaf, leaf_nodes in enumerate(tree.path_nodes)
                        if leaf_nodes[:len(nodes)] == nodes
                    ) if nodes else 0
                    selected_hidden = tuple(
                        layer[cache_leaf:cache_leaf + 1]
                        for layer in output.hidden_states
                    )
                    packed_rows = torch.arange(
                        len(keep), dtype=torch.long, device=self.device
                    )
                    update = self.features(selected_hidden, packed_rows)
                    for layer_index, packed_layer in enumerate(packed_cache.layers):
                        new_keys = packed_layer.keys[
                            cache_leaf:cache_leaf + 1, :, prefix_len:prefix_len + len(keep), :
                        ]
                        new_values = packed_layer.values[
                            cache_leaf:cache_leaf + 1, :, prefix_len:prefix_len + len(keep), :
                        ]
                        target_cache.update(new_keys, new_values, layer_index)
                else:
                    update = self.features(output.hidden_states, index)
                if full_features is not None:
                    full_features = torch.cat((full_features, update), dim=1)
                if (variant.method not in PACKED_TREE_METHODS
                        and not target_cache_precompacted):
                    cache_keep = (
                        compact_verified_keep
                        if variant.method in LAZY_TARGET_TREE_METHODS
                        else keep
                    )
                    compact_cache(
                        target_cache, prefix_len, cache_keep, self.device
                    )
                round_stats = {
                    "accepted_draft_tokens": accepted,
                    "committed_tokens": len(committed),
                    "committed_draft_tokens": min(accepted, len(committed)),
                    "proposed_tokens": (
                        variant.paths * variant.length
                        if paths is not None else len(tree.tokens)
                    ),
                    "tree_nodes": len(tree.tokens),
                    "verify_tokens": (
                        sum(cascade_block_rows)
                        if variant.method in TREE_BLOCK_SKIP_METHODS
                        else len(tree.parents)
                    ),
                }
                if variant.method in TREE_BLOCK_SKIP_METHODS:
                    round_stats.update({
                        "tree_block_skip": True,
                        "tree_block_row_cap": tree_block_row_cap,
                        "conditional_tree_block_skip": (
                            variant.method in CONDITIONAL_TREE_BLOCK_SKIP_METHODS
                            or variant.method in COST_AWARE_TREE_BLOCK_SKIP_METHODS
                        ),
                        "cost_aware_tree_block_skip": (
                            variant.method in COST_AWARE_TREE_BLOCK_SKIP_METHODS
                        ),
                        "tree_block_calls": len(cascade_block_rows),
                        "tree_block_rows": list(cascade_block_rows),
                        "tree_block_roots": list(cascade_block_roots),
                        "target_verified_rows": sum(cascade_block_rows),
                        "full_target_tree_rows": len(tree.parents),
                        "skipped_target_tree_rows": (
                            len(tree.parents) - sum(cascade_block_rows)
                        ),
                    })
                    if cost_aware_decision is not None:
                        round_stats.update({
                            "tree_block_estimated_first_call_ms": (
                                cost_aware_decision.first_call_latency_ms
                            ),
                            "tree_block_estimated_rescue_ms": (
                                cost_aware_decision.expected_rescue_latency_ms
                            ),
                            "tree_block_estimated_objective_ms": (
                                cost_aware_decision.objective_ms
                            ),
                            "tree_block_frontier_nodes": list(
                                cost_aware_decision.frontier_nodes
                            ),
                            "tree_block_first_nodes": list(
                                cost_aware_decision.block.nodes
                            ),
                            "tree_block_noncanonical": (
                                cost_aware_decision.block.nodes
                                != tuple(range(len(
                                    cost_aware_decision.block.nodes
                                )))
                            ),
                        })
                if variant.method == "ddtree_history_budget":
                    observed_commit = float(len(committed))
                    history_commit_ewma = (
                        observed_commit if history_commit_ewma is None else
                        variant.tree_history_ewma * observed_commit
                        + (1.0 - variant.tree_history_ewma)
                        * history_commit_ewma
                    )
                    round_stats["history_commit_ewma"] = history_commit_ewma
                if variant.method in LAZY_TARGET_TREE_METHODS:
                    round_stats.update({
                        "full_target_tree_rows": len(tree.parents),
                        "internal_target_tree_rows": len(internal_nodes),
                        "prefetched_leaf_rows": len(prefetched_leaf_nodes),
                        "target_verified_rows": (
                            len(verified_nodes) + int(fallback_leaf_forward)
                        ),
                        "prefetched_leaf_hit": prefetched_leaf_hit,
                        "lazy_leaf_target_forward": fallback_leaf_forward,
                        "deferred_leaf_stop": deferred_leaf_stop,
                        "posterior_probability_rows": (
                            lazy_target_stats["internal_probability_rows"]
                            + lazy_target_stats["leaf_probability_rows"]
                        ),
                        "partial_terminal_block": (
                            variant.method in PARTIAL_TERMINAL_TREE_METHODS
                        ),
                    })
                if variant.method in RECYCLE_TREE_METHODS:
                    round_stats.update({
                        "tree_bv_segments": recycle_stats["segments"],
                        "recycled_corrections": recycle_stats["recycled_corrections"],
                    })
                if variant.method in ATOM_TREE_METHODS:
                    round_stats.update({"proposal_kind": "correlated_latent_atoms",
                                        "proposal_support_size": tree_proposal.tokens.shape[-1],
                                        "proposal_atom_columns": tree_proposal.source.shape[-1],
                                        "coupling": coupling,
                                        "joint_block_verification": variant.method != "atom_tree_ancestral"})
                if variant.method in DIFFUSION_TREE_METHODS:
                    round_stats.update({"proposal_kind": "one_step_masked_block_diffusion",
                                        "denoising_steps": 1, "stochastic_positions": variant.length,
                                        "proposal_support_size": tree_proposal.tokens.shape[-1],
                                        "proposal_atom_columns": tree_proposal.source.shape[-1],
                                        "labelled_paths": variant.paths, "coupling": coupling,
                                        "joint_block_verification": variant.method != "diffusion_tree_ancestral"})
                    if variant.method in {
                        "diffusion_coalesced_block_skip",
                        "diffusion_coalesced_flow_bv",
                    }:
                        round_stats.update({
                            "entropy_adaptive_coalescence": True,
                            "coalesced_label_atom_fraction": float(
                                coalesced_state.coalesced_label_atom_fraction.mean()
                            ),
                            "expected_accepted_tokens": float(
                                coalesced_block_skip.expected_accepted_tokens(
                                    coalesced_state
                                )
                            ),
                        })
                    if variant.method in DIFFUSION_SCAFFOLD_METHODS:
                        round_stats.update({"fixed_greedy_scaffold": variant.method != "diffusion_core_spur_bv",
                                            "scaffold_fill": variant.method not in {"diffusion_scaffold_no_fill", "diffusion_core_spur_bv"},
                                            "correction_recycling": variant.method != "diffusion_scaffold_no_recycle",
                                            "continuation_backend": "ancestral" if variant.method == "diffusion_scaffold_ancestral" else "terminal",
                                            "core_spur_length": variant.diffusion_spur_length if variant.method == "diffusion_core_spur_bv" else None})
                if variant.method in PREFIX_TREE_METHODS:
                    round_stats.update({
                        "prefix_conditioned_proposal": True,
                        "proposal_support_size": variant.diffusion_support_size,
                        "prefix_strength": variant.prefix_strength,
                        "core_spur_length": (
                            variant.diffusion_spur_length
                            if variant.method in PREFIX_CORE_SPUR_METHODS
                            else None
                        ),
                        "joint_block_verification": (
                            variant.method == "prefix_core_spur_bv"
                        ),
                        "correction_recycling": (
                            variant.method == "prefix_core_spur_bv"
                        ),
                        "continuation_backend": "ancestral",
                    })
                if variant.method in LAZY_HEAD_TREE_METHODS | LAZY_SOFTMAX_TREE_METHODS:
                    round_stats.update({
                        "posterior_probability_rows": lazy_projection_stats.get(
                            "probability_rows", lazy_projection_stats["projected_rows"]
                        ),
                        "lm_head_projection_rows": lazy_projection_stats.get(
                            "lm_head_rows", lazy_projection_stats["projected_rows"]
                        ),
                        "full_vocabulary_projection_rows": lazy_projection_stats["total_tree_rows"],
                        "internal_projection_rows": lazy_projection_stats["internal_projected_rows"],
                        "leaf_projection_rows": lazy_projection_stats["leaf_projected_rows"],
                    })
                if variant.method in DIRECT_LOGITS_TREE_METHODS:
                    round_stats.update({
                        "posterior_probability_rows": direct_logits_stats[
                            "visited_probability_rows"
                        ],
                        "lm_head_projection_rows": direct_logits_stats["lm_head_rows"],
                        "full_vocabulary_projection_rows": direct_logits_stats[
                            "total_tree_rows"
                        ],
                        "direct_logits_persistent_kernel": True,
                    })
                rounds.append(round_stats)
                del output, all_p, hidden, logits
                if variant.method in TREE_BLOCK_SKIP_METHODS:
                    del (cascade_hidden_states, cascade_nodes,
                         cascade_tokens, cascade_block_roots,
                         cascade_block_rows, hidden_chunks,
                         first_block, verify_tree)
                    if cost_aware_decision is not None:
                        del cost_aware_decision
                if variant.method in LAZY_TARGET_TREE_METHODS:
                    del (verified_logits, verified_p, internal_p,
                         sparse_target_tree, internal_nodes,
                         internal_node_set, leaf_nodes,
                         prefetched_leaf_nodes, verified_nodes,
                         verified_node_to_compact, verify_tree,
                         internal_compact_indices, feature_index,
                         compact_verified_keep, lazy_target_stats)
                    if terminal_leaf >= 0 and not deferred_leaf_stop:
                        del leaf_p
                    if fallback_leaf_forward:
                        del (leaf_output, selected_hidden, internal_path_nodes)
                if variant.method in PACKED_TREE_METHODS:
                    del packed_cache, packed_ids, packed_positions, packed_p, selected_hidden
                if variant.method in RECYCLE_TREE_METHODS:
                    del node_paths, recycle_children
                elif variant.method in SPARSE_FULL_TREE_METHODS:
                    del (chosen_p, proposal_tokens, proposal_probabilities,
                         selected_leaf_probabilities, target_token_probabilities)
                elif (paths is not None and variant.method not in
                      SHARED_SUFFIX_METHODS | ATOM_TREE_METHODS
                      | DIFFUSION_TREE_METHODS | PREFIX_TREE_METHODS):
                    del p_by_path, r
                if variant.method in DIFFUSION_TREE_METHODS:
                    del diffusion_law
                del paths, q, tree, ids, mask, noise, noise_ids
        self.sync()
        ended = time.perf_counter()
        prefill_ms = (prefill_end - started) * 1000
        decode_ms = (ended - prefill_end) * 1000
        decode_tokens = len(generated) - 1
        if official_scope_decode_start is None:
            # Only reachable when max_new_tokens=1 (no speculative round).
            # The production gates use >=32 tokens, but keep the record total.
            official_scope_decode_start = prefill_end
        official_scope_decode_ms = (
            ended - official_scope_decode_start
        ) * 1000
        official_scope_output_tokens = len(generated)
        stages = meter.result()
        stage_profile = (
            summarize_stage_profile(
                stages,
                e2e_ms=(ended - started) * 1000,
                decode_ms=decode_ms,
                official_scope_decode_ms=official_scope_decode_ms,
            )
            if profile else None
        )
        return {
            "generated_token_ids": generated, "generated_tokens": len(generated),
            "decode_tokens": decode_tokens, "prefill_ms": prefill_ms, "decode_ms": decode_ms,
            "e2e_ms": (ended - started) * 1000,
            "decode_tokens_per_second": 1000 * decode_tokens / decode_ms if decode_tokens else None,
            "e2e_tokens_per_second": 1000 * len(generated) / ((ended - started) * 1000),
            "official_scope_decode_ms": official_scope_decode_ms,
            "official_scope_output_tokens": official_scope_output_tokens,
            "official_scope_time_per_output_token_ms": (
                official_scope_decode_ms / official_scope_output_tokens
            ),
            "official_scope_tokens_per_second": (
                1000 * official_scope_output_tokens / official_scope_decode_ms
                if official_scope_decode_ms else None
            ),
            "timing_contract": {
                "decode_ms": (
                    "after Target prefill through completion; includes first Draft forward"
                ),
                "decode_tokens": "returned tokens after the prefill-sampled anchor",
                "official_scope_decode_ms": (
                    "vendored DDTree/DFlash start/end scope; excludes Target prefill "
                    "and first Draft forward for speculative methods"
                ),
                "official_scope_output_tokens": (
                    "all returned tokens, including the prefill-sampled anchor"
                ),
                "internal_official_stage_synchronizations_reproduced": False,
                "upstream_scope_applicable": variant.method != "target",
            },
            "finish_reason": "eos" if generated[-1] in stops else "length",
            "target_forward_calls": target_calls, "draft_forward_calls": draft_calls,
            "target_verification_backend":self._target_verification_backend,
            "target_tokens_processed": target_tokens, "rounds": rounds,
            "greedy_audit": greedy_audit,
            "target_greedy_trace": target_greedy_trace,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
            "stages": stages,
            "stage_profile": stage_profile,
        }
