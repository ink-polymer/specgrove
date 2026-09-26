"""Fail-closed runtime identity checks for timed model execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


IDENTITY_SCHEMA = 2
MODEL_PARAMETERS_SCHEMA = 1
RUNTIME_READY_GATE_SCHEMA = 1
_DTYPE_NAMES = {
    "bfloat16":"torch.bfloat16",
    "float16":"torch.float16",
    "float32":"torch.float32",
}
_TARGET_CLASS = "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM"
_DRAFT_CLASS = "_gbv_dflash_model.dflash.DFlashDraftModel"
_TOKENIZER_CLASS = (
    "transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast"
)
_TOKENIZER_VOCAB_SHA256 = (
    "f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d"
)
_TOKENIZER_BACKEND_SHA256 = (
    "91a5d4d92157b8d3cdb2599dca8ff445ecec6fdaeab57904c229ece438493bca"
)
_TOKENIZER_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
_REGISTERED_MODEL_PAIRS = {
    ("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"):{
        "target":{
            "class":_TARGET_CLASS, "model_type":"qwen3", "vocab_size":151936,
            "hidden_size":2560, "num_hidden_layers":36,
            "parameters":4022468096, "parameter_tensors":398,
        },
        "draft":{
            "class":_DRAFT_CLASS, "model_type":"qwen3", "vocab_size":151936,
            "hidden_size":2560, "num_hidden_layers":5,
            "parameters":537427200, "parameter_tensors":58,
        },
        "draft_runtime":{
            "block_size":16,
            "target_layer_ids":[1, 9, 17, 25, 33],
            "mask_token_id":151669,
        },
    },
    ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"):{
        "target":{
            "class":_TARGET_CLASS, "model_type":"qwen3", "vocab_size":151936,
            "hidden_size":4096, "num_hidden_layers":36,
            "parameters":8190735360, "parameter_tensors":399,
        },
        "draft":{
            "class":_DRAFT_CLASS, "model_type":"qwen3", "vocab_size":151936,
            "hidden_size":4096, "num_hidden_layers":5,
            "parameters":1048626432, "parameter_tensors":58,
        },
        "draft_runtime":{
            "block_size":16,
            "target_layer_ids":[1, 9, 17, 25, 33],
            "mask_token_id":151669,
        },
    },
}
_EXPECTATION_KEYS = {
    "schema", "strict_same_tree_t1", "floating_parameter_dtype",
    "target_attention", "draft_attention", "parameter_device", "allow_tf32",
    "require_eval", "require_frozen", "require_no_proposal_adapter",
    "require_no_embedded_adapters", "target_config", "draft_config",
    "draft_runtime", "tokenizer",
}
_CONFIG_EXPECTATION_KEYS = {
    "class", "name_or_path", "commit_hash", "model_type", "vocab_size",
    "hidden_size", "num_hidden_layers", "parameters", "parameter_tensors",
}
_TOKENIZER_EXPECTATION_KEYS = {
    "class", "name_or_path", "source_revision", "vocab_size", "length",
    "vocab_sha256", "backend_sha256", "chat_template_sha256", "bos_token_id",
    "eos_token_id", "pad_token_id", "special_token_ids", "stop_token_ids",
}
_DRAFT_RUNTIME_KEYS = {"block_size", "target_layer_ids", "mask_token_id"}
_MODEL_IDENTITY_KEYS = {
    "class", "parameter_tensors", "parameters", "floating_parameter_tensors",
    "floating_parameter_dtypes", "parameter_devices", "training",
    "training_modules", "trainable_parameter_tensors", "trainable_parameters",
    "attention_implementation", "embedded_adapter_modules",
    "adapter_parameter_names", "config",
}
_CONFIG_IDENTITY_KEYS = {
    "_name_or_path", "_commit_hash", "model_type", "vocab_size",
    "hidden_size", "num_hidden_layers",
}
_IDENTITY_KEYS = {
    "schema", "target", "draft", "draft_runtime", "tokenizer", "tf32",
    "proposal_adapter_attribute_present", "proposal_adapter_attached",
    "proposal_adapter_class",
}
_TOKENIZER_IDENTITY_KEYS = {
    "class", "name_or_path", "source_revision", "vocab_size", "length",
    "vocab_sha256", "backend_sha256", "chat_template_sha256", "bos_token_id",
    "eos_token_id", "pad_token_id", "special_token_ids", "stop_token_ids",
    "minimum_token_id", "maximum_token_id",
}
_READY_GATE_KEYS = {
    "schema", "status", "expectations", "identity_after_load",
    "identity_before_timing", "passed_after_load", "passed_before_timing",
    "identities_match",
}
_MODEL_PARAMETERS_KEYS = {
    "schema", "run_id", "target_parameters", "draft_parameters",
    "trainable_parameters", "draft_block_size", "target_feature_layers",
    "runtime_model_gate",
}


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_commit(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_indexed_cuda_device(value) -> bool:
    if not isinstance(value, str):
        return False
    prefix, separator, index = value.partition(":")
    return prefix == "cuda" and separator == ":" and index.isdigit()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _config_expectation(model: Mapping[str, Any], role: str,
                        registered: Mapping[str, Any] | None) -> dict:
    details = dict(registered or {})
    return {
        "class":details.get("class"),
        "name_or_path":model.get(role),
        "commit_hash":model.get(role + "_revision"),
        "model_type":details.get("model_type"),
        "vocab_size":details.get("vocab_size"),
        "hidden_size":details.get("hidden_size"),
        "num_hidden_layers":details.get("num_hidden_layers"),
        "parameters":details.get("parameters"),
        "parameter_tensors":details.get("parameter_tensors"),
    }


def _tokenizer_expectation(model: Mapping[str, Any], registered: bool) -> dict:
    if not registered:
        return {
            "class":None, "name_or_path":None,
            "source_revision":None, "vocab_size":None,
            "length":None, "vocab_sha256":None, "backend_sha256":None,
            "chat_template_sha256":None, "bos_token_id":None,
            "eos_token_id":None, "pad_token_id":None,
            "special_token_ids":None, "stop_token_ids":None,
        }
    return {
        "class":_TOKENIZER_CLASS,
        "name_or_path":model["target"],
        "source_revision":model["target_revision"],
        "vocab_size":151643,
        "length":151669,
        "vocab_sha256":_TOKENIZER_VOCAB_SHA256,
        "backend_sha256":_TOKENIZER_BACKEND_SHA256,
        "chat_template_sha256":_TOKENIZER_CHAT_TEMPLATE_SHA256,
        "bos_token_id":None,
        "eos_token_id":151645,
        "pad_token_id":151643,
        "special_token_ids":list(range(151643, 151657)),
        "stop_token_ids":[151645, 151643],
    }


def _draft_runtime_expectation(registered: Mapping[str, Any] | None) -> dict:
    values = (registered or {}).get("draft_runtime", {})
    return {
        "block_size":values.get("block_size"),
        "target_layer_ids":values.get("target_layer_ids"),
        "mask_token_id":values.get("mask_token_id"),
    }


def runtime_model_expectations(
        model: Mapping[str, Any], *, strict_same_tree_t1=False,
        device: str | None = None) -> dict:
    """Build a runtime contract from a validated model configuration."""
    dtype = model.get("dtype", "bfloat16")
    target_attention = model.get("target_attention", "sdpa")
    draft_attention = model.get("draft_attention", "sdpa")
    allow_tf32 = model.get("allow_tf32", False)
    if dtype not in _DTYPE_NAMES:
        raise RuntimeError(f"Unsupported registered model dtype: {dtype!r}")
    if target_attention not in {"sdpa", "eager"}:
        raise RuntimeError(f"Unsupported Target attention backend: {target_attention!r}")
    if draft_attention not in {"sdpa", "eager"}:
        raise RuntimeError(f"Unsupported Draft attention backend: {draft_attention!r}")
    if not isinstance(allow_tf32, bool):
        raise RuntimeError("allow_tf32 must be a boolean")
    if device is not None and (not isinstance(device, str) or not device):
        raise RuntimeError("parameter device must be a non-empty string")
    pair = (model.get("target"), model.get("draft"))
    registered = _REGISTERED_MODEL_PAIRS.get(pair)
    if strict_same_tree_t1:
        if (dtype != "bfloat16" or target_attention != "sdpa"
                or draft_attention != "sdpa" or allow_tf32 is not False
                or not _is_indexed_cuda_device(device)):
            raise RuntimeError(
                "Registered same-tree T=1 requires an indexed CUDA device, "
                "bfloat16, SDPA, and TF32 disabled"
            )
        if registered is None or not all(
                _is_commit(model.get(role + "_revision"))
                for role in ("target", "draft")):
            raise RuntimeError(
                "Registered same-tree T=1 requires a known model pair and pinned commits"
            )
    expected = {
        "schema":IDENTITY_SCHEMA,
        "strict_same_tree_t1":bool(strict_same_tree_t1),
        "floating_parameter_dtype":_DTYPE_NAMES[dtype],
        "target_attention":target_attention,
        "draft_attention":draft_attention,
        "parameter_device":device,
        "allow_tf32":allow_tf32,
        "require_eval":True,
        "require_frozen":True,
        "require_no_proposal_adapter":True,
        "require_no_embedded_adapters":True,
        "target_config":_config_expectation(
            model, "target", registered and registered["target"],
        ),
        "draft_config":_config_expectation(
            model, "draft", registered and registered["draft"],
        ),
        "draft_runtime":_draft_runtime_expectation(registered),
        "tokenizer":_tokenizer_expectation(
            model, registered is not None and strict_same_tree_t1,
        ),
    }
    return expected


def strict_runtime_model_expectations() -> dict:
    """Return the state-only BF16/SDPA contract used by legacy callers."""
    return runtime_model_expectations({})


def is_registered_same_tree_t1(cfg: dict) -> bool:
    """Identify a registered same-tree T=1 protocol, not filenames."""
    entries = cfg.get("explicit_variants", [])
    if not isinstance(entries, list):
        return False
    for entry in entries:
        variant = entry.get("variant") if isinstance(entry, dict) else None
        if not isinstance(variant, dict):
            continue
        temperature = variant.get("temperature")
        if (variant.get("method") in {
                    "ddtree_fused_scan", "ddtree_same_draw_fused",
                    "ddtree_compiled_batched",
                }
                and isinstance(temperature, (int, float))
                and not isinstance(temperature, bool)
                and temperature == 1.0):
            return True
    return False


def _embedded_adapter_evidence(model) -> tuple[list[str], list[str]]:
    modules = []
    for name, module in model.named_modules():
        qualified = f"{type(module).__module__}.{type(module).__qualname__}".lower()
        peft_config = getattr(module, "peft_config", None)
        adapter_config = getattr(module, "adapter_config", None)
        if ("peftmodel" in qualified or "loramodel" in qualified
                or (peft_config is not None and bool(peft_config))
                or (adapter_config is not None and bool(adapter_config))):
            modules.append(name or "<root>")
    parameters = [
        name for name, _ in model.named_parameters()
        if any(marker in name.lower() for marker in (
            "lora_", ".adapter_", "prompt_encoder", "prefix_encoder",
        ))
    ]
    return sorted(set(modules)), sorted(parameters)


def _model_identity(model) -> dict:
    parameters = list(model.parameters())
    floating = [parameter for parameter in parameters if parameter.is_floating_point()]
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    training_modules = [
        name or "<root>" for name, module in model.named_modules()
        if module.training
    ]
    adapter_modules, adapter_parameters = _embedded_adapter_evidence(model)
    config = getattr(model, "config", None)
    return {
        "class":f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_tensors":len(parameters),
        "parameters":sum(parameter.numel() for parameter in parameters),
        "floating_parameter_tensors":len(floating),
        "floating_parameter_dtypes":sorted({str(parameter.dtype) for parameter in floating}),
        "parameter_devices":sorted({str(parameter.device) for parameter in parameters}),
        "training":bool(model.training),
        "training_modules":training_modules,
        "trainable_parameter_tensors":len(trainable),
        "trainable_parameters":sum(parameter.numel() for parameter in trainable),
        "attention_implementation":getattr(config, "_attn_implementation", None),
        "embedded_adapter_modules":adapter_modules,
        "adapter_parameter_names":adapter_parameters,
        "config":{
            field:getattr(config, field, None) for field in _CONFIG_IDENTITY_KEYS
        },
    }


def runtime_stop_token_ids(engine, tokenizer) -> list[int]:
    """Return the exact stop IDs used by generation and runtime evidence."""
    generation = getattr(engine.target, "generation_config", None)
    value = getattr(generation, "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if _plain_int(value):
        return [value]
    return list(value or [])


def _tokenizer_identity(tokenizer, stop_ids: list[int], source_revision) -> dict:
    vocabulary = tokenizer.get_vocab()
    token_ids = list(vocabulary.values())
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    return {
        "class":f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path":getattr(tokenizer, "name_or_path", None),
        "source_revision":source_revision,
        "vocab_size":getattr(tokenizer, "vocab_size", None),
        "length":len(tokenizer),
        "vocab_sha256":_canonical_sha256(vocabulary),
        "backend_sha256":_canonical_sha256(backend),
        "chat_template_sha256":hashlib.sha256(chat_template.encode()).hexdigest(),
        "bos_token_id":getattr(tokenizer, "bos_token_id", None),
        "eos_token_id":getattr(tokenizer, "eos_token_id", None),
        "pad_token_id":getattr(tokenizer, "pad_token_id", None),
        "special_token_ids":sorted(set(tokenizer.all_special_ids)),
        "stop_token_ids":list(stop_ids),
        "minimum_token_id":min(token_ids) if token_ids else None,
        "maximum_token_id":max(token_ids) if token_ids else None,
    }


def runtime_model_identity(engine, tokenizer=None) -> dict:
    """Capture only stable, JSON-serializable state relevant to formal timing."""
    import torch

    adapter_present = hasattr(engine, "proposal_adapter")
    adapter = getattr(engine, "proposal_adapter", None)
    tokenizer_identity = None
    if tokenizer is not None:
        tokenizer_identity = _tokenizer_identity(
            tokenizer, runtime_stop_token_ids(engine, tokenizer),
            getattr(getattr(engine.target, "config", None), "_commit_hash", None),
        )
    return {
        "schema":IDENTITY_SCHEMA,
        "target":_model_identity(engine.target),
        "draft":_model_identity(engine.draft),
        "draft_runtime":{
            "block_size":getattr(engine.draft, "block_size", None),
            "target_layer_ids":list(
                getattr(engine.draft, "target_layer_ids", []) or []
            ),
            "mask_token_id":getattr(engine.draft, "mask_token_id", None),
        },
        "tokenizer":tokenizer_identity,
        "tf32":{
            "cuda_matmul":bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn":bool(torch.backends.cudnn.allow_tf32),
        },
        "proposal_adapter_attribute_present":adapter_present,
        "proposal_adapter_attached":adapter is not None,
        "proposal_adapter_class":(
            f"{type(adapter).__module__}.{type(adapter).__qualname__}"
            if adapter is not None else None
        ),
    }


def _validate_expectations(expected: dict) -> None:
    if not isinstance(expected, dict) or set(expected) != _EXPECTATION_KEYS:
        raise RuntimeError("runtime expectations have an invalid schema")
    if (not _plain_int(expected.get("schema"))
            or expected["schema"] != IDENTITY_SCHEMA):
        raise RuntimeError("runtime expectations have an unsupported schema")
    for field in (
        "strict_same_tree_t1", "allow_tf32", "require_eval",
        "require_frozen", "require_no_proposal_adapter",
        "require_no_embedded_adapters",
    ):
        if not isinstance(expected.get(field), bool):
            raise RuntimeError(f"runtime expectation {field} must be boolean")
    if expected["floating_parameter_dtype"] not in set(_DTYPE_NAMES.values()):
        raise RuntimeError("runtime expectation has an unsupported dtype")
    for field in ("target_attention", "draft_attention"):
        if expected[field] not in {"sdpa", "eager"}:
            raise RuntimeError("runtime expectation has an unsupported attention backend")
    if (expected["parameter_device"] is not None
            and (not isinstance(expected["parameter_device"], str)
                 or not expected["parameter_device"])):
        raise RuntimeError("runtime expectation has an invalid device")
    for role in ("target", "draft"):
        contract = expected.get(role + "_config")
        if not isinstance(contract, Mapping) or set(contract) != _CONFIG_EXPECTATION_KEYS:
            raise RuntimeError(f"{role} config expectations have an invalid schema")
        for field in ("class", "name_or_path", "model_type"):
            value = contract[field]
            if value is not None and (not isinstance(value, str) or not value):
                raise RuntimeError(f"{role} config expectation {field} is invalid")
        revision = contract["commit_hash"]
        if revision is not None and not _is_commit(revision):
            raise RuntimeError(f"{role} config revision expectation is invalid")
        for field in (
            "vocab_size", "hidden_size", "num_hidden_layers", "parameters",
            "parameter_tensors",
        ):
            value = contract[field]
            if value is not None and (not _plain_int(value) or value < 1):
                raise RuntimeError(f"{role} config expectation {field} is invalid")
    tokenizer = expected.get("tokenizer")
    if (not isinstance(tokenizer, Mapping)
            or set(tokenizer) != _TOKENIZER_EXPECTATION_KEYS):
        raise RuntimeError("tokenizer expectations have an invalid schema")
    draft_runtime = expected.get("draft_runtime")
    if (not isinstance(draft_runtime, Mapping)
            or set(draft_runtime) != _DRAFT_RUNTIME_KEYS):
        raise RuntimeError("Draft runtime expectations have an invalid schema")
    if expected["strict_same_tree_t1"]:
        for role in ("target", "draft"):
            contract = expected[role + "_config"]
            if (not all(contract.get(field) is not None
                        for field in _CONFIG_EXPECTATION_KEYS)
                    or not _is_commit(contract["commit_hash"])):
                raise RuntimeError(f"strict {role} checkpoint expectations are incomplete")
        required_tokenizer_fields = _TOKENIZER_EXPECTATION_KEYS - {
            "bos_token_id",
        }
        tokenizer_vocab_limit = expected["target_config"]["vocab_size"]
        tokenizer_contract = expected["tokenizer"]
        if (expected["floating_parameter_dtype"] != "torch.bfloat16"
                or expected["target_attention"] != "sdpa"
                or expected["draft_attention"] != "sdpa"
                or not _is_indexed_cuda_device(expected["parameter_device"])
                or expected["allow_tf32"] is not False
                or not _plain_int(draft_runtime["block_size"])
                or draft_runtime["block_size"] < 1
                or not _plain_int(draft_runtime["mask_token_id"])
                or not isinstance(draft_runtime["target_layer_ids"], list)
                or not draft_runtime["target_layer_ids"]
                or any(not _plain_int(layer) or layer < 0
                       for layer in draft_runtime["target_layer_ids"])
                or draft_runtime["target_layer_ids"]
                   != sorted(set(draft_runtime["target_layer_ids"]))
                or any(layer >= expected["target_config"]["num_hidden_layers"]
                       for layer in draft_runtime["target_layer_ids"])
                or not 0 <= draft_runtime["mask_token_id"] < tokenizer_vocab_limit
                or any(expected["tokenizer"].get(field) is None
                       for field in required_tokenizer_fields)
                or any(not isinstance(tokenizer_contract[field], str)
                       or not tokenizer_contract[field]
                       for field in ("class", "name_or_path"))
                or not _is_commit(tokenizer_contract["source_revision"])
                or any(not _is_sha256(tokenizer_contract[field]) for field in (
                    "vocab_sha256", "backend_sha256", "chat_template_sha256",
                ))
                or not _plain_int(tokenizer_contract["vocab_size"])
                or not _plain_int(tokenizer_contract["length"])
                or not (1 <= tokenizer_contract["vocab_size"]
                        <= tokenizer_contract["length"] <= tokenizer_vocab_limit)
                or not _valid_token_ids(
                    tokenizer_contract["special_token_ids"],
                    tokenizer_vocab_limit, nonempty=True,
                )
                or not _valid_token_ids(
                    tokenizer_contract["stop_token_ids"],
                    tokenizer_vocab_limit, nonempty=True,
                )
                or not set(tokenizer_contract["stop_token_ids"]) <= set(
                    tokenizer_contract["special_token_ids"]
                )
                or tokenizer_contract["bos_token_id"] is not None
                or any(not _plain_int(tokenizer_contract[field])
                       or not 0 <= tokenizer_contract[field] < tokenizer_vocab_limit
                       for field in ("eos_token_id", "pad_token_id"))):
            raise RuntimeError("strict same-tree T=1 expectations were relaxed")


def _validate_model_identity(identity: dict, expected: dict, role: str) -> None:
    if not isinstance(identity, dict) or set(identity) != _MODEL_IDENTITY_KEYS:
        raise RuntimeError(f"{role} runtime identity has an invalid schema")
    if not isinstance(identity["class"], str) or not identity["class"]:
        raise RuntimeError(f"{role} runtime class is missing")
    contract = expected[f"{role.lower()}_config"]
    if contract["class"] is not None and identity["class"] != contract["class"]:
        raise RuntimeError(f"{role} runtime class changed")
    counts = (
        "parameter_tensors", "parameters", "floating_parameter_tensors",
        "trainable_parameter_tensors", "trainable_parameters",
    )
    if any(not _plain_int(identity[field]) or identity[field] < 0 for field in counts):
        raise RuntimeError(f"{role} runtime parameter counts are invalid")
    if (identity["floating_parameter_tensors"] < 1
            or not identity["floating_parameter_dtypes"]):
        raise RuntimeError(f"{role} has no floating-point parameters")
    if identity["parameter_tensors"] < 1 or identity["parameters"] < 1:
        raise RuntimeError(f"{role} has no parameters")
    if identity["floating_parameter_tensors"] > identity["parameter_tensors"]:
        raise RuntimeError(f"{role} floating parameter count is impossible")
    if (expected["strict_same_tree_t1"]
            and identity["floating_parameter_tensors"]
            != identity["parameter_tensors"]):
        raise RuntimeError(f"{role} contains non-floating parameters")
    if identity["trainable_parameter_tensors"] > identity["parameter_tensors"]:
        raise RuntimeError(f"{role} trainable parameter count is impossible")
    if identity["trainable_parameters"] > identity["parameters"]:
        raise RuntimeError(f"{role} trainable parameter size is impossible")
    for field in ("parameter_tensors", "parameters"):
        if contract[field] is not None and identity[field] != contract[field]:
            raise RuntimeError(f"{role} registered {field} changed")
    if (not isinstance(identity["floating_parameter_dtypes"], list)
            or identity["floating_parameter_dtypes"]
               != [expected["floating_parameter_dtype"]]):
        raise RuntimeError(f"{role} floating parameter dtype changed")
    devices = identity["parameter_devices"]
    if (not isinstance(devices, list) or not devices
            or any(not isinstance(device, str) or not device for device in devices)
            or devices != sorted(set(devices))):
        raise RuntimeError(f"{role} parameter devices are invalid")
    if (expected["parameter_device"] is not None
            and devices != [expected["parameter_device"]]):
        raise RuntimeError(f"{role} parameter device changed")
    if not isinstance(identity["training"], bool):
        raise RuntimeError(f"{role} training flag is invalid")
    training_modules = identity["training_modules"]
    if (not isinstance(training_modules, list)
            or any(not isinstance(name, str) for name in training_modules)):
        raise RuntimeError(f"{role} eval state evidence is invalid")
    if expected["require_eval"] and (identity["training"] or training_modules):
        raise RuntimeError(f"{role} eval state changed")
    if expected["require_frozen"] and (
            identity["trainable_parameter_tensors"] != 0
            or identity["trainable_parameters"] != 0):
        raise RuntimeError(f"{role} has trainable parameters")
    attention = expected[f"{role.lower()}_attention"]
    if identity["attention_implementation"] != attention:
        raise RuntimeError(f"{role} attention implementation changed")
    for field in ("embedded_adapter_modules", "adapter_parameter_names"):
        values = identity[field]
        if (not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)):
            raise RuntimeError(f"{role} adapter evidence is invalid")
        if expected["require_no_embedded_adapters"] and values:
            raise RuntimeError(f"{role} contains an unregistered embedded adapter")
    config = identity["config"]
    if not isinstance(config, dict) or set(config) != _CONFIG_IDENTITY_KEYS:
        raise RuntimeError(f"{role} config identity has an invalid schema")
    if not isinstance(config["_name_or_path"], str) or not config["_name_or_path"]:
        raise RuntimeError(f"{role} checkpoint name is missing")
    if config["_commit_hash"] is not None and not _is_commit(config["_commit_hash"]):
        raise RuntimeError(f"{role} checkpoint revision is invalid")
    if not isinstance(config["model_type"], str) or not config["model_type"]:
        raise RuntimeError(f"{role} model type is missing")
    for field in ("vocab_size", "hidden_size", "num_hidden_layers"):
        if not _plain_int(config[field]) or config[field] < 1:
            raise RuntimeError(f"{role} config field {field} is invalid")
    for expected_field, actual_field in (
        ("name_or_path", "_name_or_path"),
        ("commit_hash", "_commit_hash"),
        ("model_type", "model_type"),
        ("vocab_size", "vocab_size"),
        ("hidden_size", "hidden_size"),
        ("num_hidden_layers", "num_hidden_layers"),
    ):
        registered = contract[expected_field]
        if registered is not None and config[actual_field] != registered:
            raise RuntimeError(f"{role} registered {expected_field} changed")


def _valid_token_ids(values, vocab_size: int, *, nonempty: bool) -> bool:
    return (
        isinstance(values, list)
        and (bool(values) or not nonempty)
        and len(values) == len(set(values))
        and all(_plain_int(value) and 0 <= value < vocab_size for value in values)
    )


def _validate_draft_runtime_identity(identity: Any, expected: Mapping[str, Any],
                                     target_layers: int,
                                     target_vocab_size: int) -> None:
    if not isinstance(identity, Mapping) or set(identity) != _DRAFT_RUNTIME_KEYS:
        raise RuntimeError("Draft runtime identity has an invalid schema")
    block_size = identity["block_size"]
    layer_ids = identity["target_layer_ids"]
    mask_token_id = identity["mask_token_id"]
    if (not _plain_int(block_size) or block_size < 1
            or not isinstance(layer_ids, list) or not layer_ids
            or any(not _plain_int(layer) or not 0 <= layer < target_layers
                   for layer in layer_ids)
            or layer_ids != sorted(set(layer_ids))
            or not _plain_int(mask_token_id)
            or not 0 <= mask_token_id < target_vocab_size):
        raise RuntimeError("Draft runtime identity is internally invalid")
    for field in _DRAFT_RUNTIME_KEYS:
        registered = expected[field]
        if registered is not None and identity[field] != registered:
            raise RuntimeError(f"Draft registered runtime field {field} changed")


def _validate_tokenizer_identity(identity: Any, expected: Mapping[str, Any],
                                 target_vocab_size: int) -> None:
    required = expected["name_or_path"] is not None
    if identity is None and not required:
        return
    if not isinstance(identity, Mapping) or set(identity) != _TOKENIZER_IDENTITY_KEYS:
        raise RuntimeError("tokenizer runtime identity has an invalid schema")
    registered_fields = (
        "class", "name_or_path", "source_revision", "vocab_size", "length",
        "vocab_sha256", "backend_sha256", "chat_template_sha256", "bos_token_id",
        "eos_token_id", "pad_token_id", "special_token_ids", "stop_token_ids",
    )
    for field in registered_fields:
        registered = expected[field]
        if ((required and identity[field] != registered)
                or (not required and registered is not None
                    and identity[field] != registered)):
            raise RuntimeError(f"tokenizer registered {field} changed")
    if (not isinstance(identity["class"], str) or not identity["class"]
            or not isinstance(identity["name_or_path"], str)
            or not identity["name_or_path"]
            or not _plain_int(identity["vocab_size"])
            or not _plain_int(identity["length"])
            or identity["vocab_size"] < 1
            or identity["length"] < identity["vocab_size"]
            or identity["length"] > target_vocab_size
            or not all(_is_sha256(identity[field]) for field in (
                "vocab_sha256", "backend_sha256", "chat_template_sha256",
            ))
            or not _valid_token_ids(
                identity["special_token_ids"], target_vocab_size, nonempty=True,
            )
            or not _valid_token_ids(
                identity["stop_token_ids"], target_vocab_size, nonempty=True,
            )
            or not set(identity["stop_token_ids"]) <= set(identity["special_token_ids"])
            or any(value is not None
                   and (not _plain_int(value) or not 0 <= value < target_vocab_size)
                   for value in (
                       identity["bos_token_id"], identity["eos_token_id"],
                       identity["pad_token_id"],
                   ))
            or not _plain_int(identity["minimum_token_id"])
            or identity["minimum_token_id"] != 0
            or not _plain_int(identity["maximum_token_id"])
            or not 0 <= identity["maximum_token_id"] < target_vocab_size):
        raise RuntimeError("tokenizer runtime identity is internally invalid")


def validate_runtime_model_identity(identity: dict, expected: dict) -> None:
    """Validate persisted runtime evidence without trusting its pass booleans."""
    _validate_expectations(expected)
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise RuntimeError("runtime model identity has an invalid schema")
    if (not _plain_int(identity.get("schema"))
            or identity["schema"] != IDENTITY_SCHEMA):
        raise RuntimeError("runtime model identity has an unsupported schema")
    _validate_model_identity(identity.get("target"), expected, "Target")
    _validate_model_identity(identity.get("draft"), expected, "Draft")
    target_vocab_size = identity["target"]["config"]["vocab_size"]
    if identity["draft"]["config"]["vocab_size"] != target_vocab_size:
        raise RuntimeError("Target and Draft vocabularies differ")
    _validate_draft_runtime_identity(
        identity["draft_runtime"], expected["draft_runtime"],
        identity["target"]["config"]["num_hidden_layers"], target_vocab_size,
    )
    _validate_tokenizer_identity(
        identity["tokenizer"], expected["tokenizer"], target_vocab_size,
    )
    tf32 = identity.get("tf32")
    if (not isinstance(tf32, dict) or set(tf32) != {"cuda_matmul", "cudnn"}
            or any(not isinstance(value, bool) for value in tf32.values())):
        raise RuntimeError("TF32 runtime evidence has an invalid schema")
    if tf32 != {
        "cuda_matmul":expected["allow_tf32"],
        "cudnn":expected["allow_tf32"],
    }:
        raise RuntimeError("TF32 runtime state changed")
    if identity.get("proposal_adapter_attribute_present") is not True:
        raise RuntimeError("proposal adapter contract is missing")
    if not isinstance(identity.get("proposal_adapter_attached"), bool):
        raise RuntimeError("proposal adapter attachment state is invalid")
    if expected["require_no_proposal_adapter"] and (
            identity["proposal_adapter_attached"]
            or identity.get("proposal_adapter_class") is not None):
        raise RuntimeError("an unregistered proposal adapter is attached")


def validate_model_parameters_evidence(
        parameters: Any, *, run_id: str, expected: dict,
        require_ready: bool) -> bool:
    """Validate the persisted live-model gate used to authorize resume."""
    if not isinstance(parameters, Mapping) or set(parameters) != _MODEL_PARAMETERS_KEYS:
        raise RuntimeError("model parameter evidence has an invalid schema")
    if (not _plain_int(parameters.get("schema"))
            or parameters["schema"] != MODEL_PARAMETERS_SCHEMA
            or parameters.get("run_id") != run_id):
        raise RuntimeError("model parameter evidence belongs to another run")
    for field in (
        "target_parameters", "draft_parameters", "trainable_parameters",
        "draft_block_size",
    ):
        if not _plain_int(parameters.get(field)) or parameters[field] < 0:
            raise RuntimeError(f"model parameter field {field} is invalid")
    if (parameters["target_parameters"] < 1
            or parameters["draft_parameters"] < 1
            or parameters["trainable_parameters"] != 0
            or parameters["draft_block_size"] < 1):
        raise RuntimeError("model parameter counts or block size are invalid")
    layers = parameters.get("target_feature_layers")
    if (not isinstance(layers, list) or not layers
            or any(not _plain_int(layer) or layer < 0 for layer in layers)
            or layers != sorted(set(layers))):
        raise RuntimeError("Draft Target-feature layer evidence is invalid")

    gate = parameters.get("runtime_model_gate")
    if not isinstance(gate, Mapping) or set(gate) != _READY_GATE_KEYS:
        raise RuntimeError("runtime ready gate has an invalid schema")
    try:
        expectations_match = (
            _canonical_sha256(gate.get("expectations"))
            == _canonical_sha256(expected)
        )
    except (TypeError, ValueError):
        expectations_match = False
    if (not _plain_int(gate.get("schema"))
            or gate["schema"] != RUNTIME_READY_GATE_SCHEMA
            or not expectations_match
            or gate.get("passed_after_load") is not True):
        raise RuntimeError("runtime ready gate contract is incomplete")
    validate_runtime_model_identity(gate.get("identity_after_load"), expected)
    if gate.get("status") == "loaded_not_ready":
        ready = False
        if (gate.get("identity_before_timing") is not None
                or gate.get("passed_before_timing") is not False
                or gate.get("identities_match") is not False):
            raise RuntimeError("loaded-only runtime gate has contradictory fields")
    elif gate.get("status") == "ready_for_timing":
        ready = True
        validate_runtime_model_identity(gate.get("identity_before_timing"), expected)
        if (gate.get("passed_before_timing") is not True
                or gate.get("identities_match") is not True
                or gate["identity_after_load"] != gate["identity_before_timing"]):
            raise RuntimeError("runtime gate is not ready for timing")
    else:
        raise RuntimeError("runtime ready gate has an unknown status")
    loaded = gate["identity_after_load"]
    if (parameters["target_parameters"] != loaded["target"]["parameters"]
            or parameters["draft_parameters"] != loaded["draft"]["parameters"]
            or parameters["trainable_parameters"] != (
                loaded["target"]["trainable_parameters"]
                + loaded["draft"]["trainable_parameters"]
            )
            or parameters["draft_block_size"]
               != loaded["draft_runtime"]["block_size"]
            or parameters["target_feature_layers"]
               != loaded["draft_runtime"]["target_layer_ids"]):
        raise RuntimeError("top-level model counts differ from runtime identity")
    if require_ready and not ready:
        raise RuntimeError("existing result rows lack a ready-for-timing model gate")
    return ready


def enforce_runtime_model_gate(
        engine, expected: dict | None = None, *, tokenizer=None) -> dict:
    """Capture and validate the live engine, raising before any formal timing."""
    if expected is None:
        expected = strict_runtime_model_expectations()
    try:
        identity = runtime_model_identity(engine, tokenizer)
        validate_runtime_model_identity(identity, expected)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Runtime model gate failed: {exc}") from exc
    return identity
