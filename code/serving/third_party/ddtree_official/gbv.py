"""DFlash multi-path tree decoding with Greedy Block Verification (GBV)."""

import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from block_verification import (
    block_rejection_sample,
    gbv_select_path_and_probs,
    sample_probs,
    sampling_probs,
)
from ddtree import compile_ddtree_tree, compact_dynamic_cache
from dflash import cuda_time, empty_stage_times
from model import DFlashDraftModel, extract_context_feature, sample


GBV_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def build_sampled_path_tree(paths: torch.Tensor):
    """Merge K sampled paths into a prefix tree in parent-before-child order."""
    if paths.ndim != 2:
        raise ValueError("paths must have shape [K, L]")
    paths_cpu = paths.detach().to("cpu", torch.long)
    parents = [-1]
    child_maps = [dict()]
    node_tokens: list[int] = []
    node_depths: list[int] = []
    path_nodes: list[list[int]] = []
    for path in paths_cpu.tolist():
        parent = 0
        nodes = []
        for depth, token in enumerate(path, start=1):
            child = child_maps[parent].get(token)
            if child is None:
                child = len(parents)
                child_maps[parent][token] = child
                parents.append(parent)
                child_maps.append({})
                node_tokens.append(token)
                node_depths.append(depth)
            nodes.append(child)
            parent = child
        path_nodes.append(nodes)

    count = len(parents)
    visibility = np.zeros((count, count), dtype=np.bool_)
    visibility[0, 0] = True
    for index in range(1, count):
        visibility[index, :index] = visibility[parents[index], :index]
        visibility[index, index] = True
    return (
        torch.tensor(node_tokens, dtype=torch.long),
        torch.tensor(node_depths, dtype=torch.long),
        parents,
        child_maps,
        torch.from_numpy(visibility),
        path_nodes,
    )


@torch.inference_mode()
def gbv_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 1.0,
    path_count: int = 3,
) -> SimpleNamespace:
    """Sample K paths, merge them into one tree forward, then apply exact GBV."""
    if temperature <= 0:
        raise ValueError("GBV requires temperature > 0")
    if block_size <= 1 or path_count <= 0:
        raise ValueError("GBV requires block_size > 1 and path_count > 0")

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    horizon = block_size - 1
    max_nodes = 1 + path_count * horizon
    output_ids = torch.full(
        (1, max_length + max_nodes), mask_token_id, dtype=torch.long, device=model.device
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_ids_buffer = torch.empty((1, max_nodes), dtype=torch.long, device=model.device)
    verify_pos_buffer = torch.empty((1, max_nodes), dtype=torch.long, device=model.device)
    attention_buffer = torch.zeros(
        (1, 1, max_nodes, max_length + max_nodes), dtype=target.dtype, device=model.device
    )
    visibility_buffer = torch.empty((max_nodes, max_nodes), dtype=torch.bool, device=model.device)
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    stage_times = empty_stage_times(GBV_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=target_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    ttft = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock = cuda_time()
    start = num_input_tokens
    acceptance_lengths, round_timestamps = [], []
    previous_tree_start = previous_tree_length = 0
    first_draft = True

    while start < max_length:
        template = output_ids[:, start : start + block_size].clone()
        draft_start = cuda_time()
        noise = target.model.embed_tokens(template)
        draft_logits = target.lm_head(
            model(
                target_hidden=target_hidden,
                noise_embedding=noise,
                position_ids=position_ids[:, draft_cache.get_seq_length() : start + block_size],
                past_key_values=draft_cache,
                use_cache=True,
                is_causal=False,
            )[:, -horizon:, :]
        )[0]
        draft_cache.crop(start)
        draft_probs = sampling_probs(draft_logits, temperature)
        paths = sample_probs(draft_probs.unsqueeze(0).expand(path_count, -1, -1))
        elapsed = cuda_time() - draft_start
        if first_draft:
            first_draft = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += elapsed

        build_start = time.perf_counter()
        node_ids, node_depths, parents, child_maps, visibility, path_nodes = build_sampled_path_tree(paths)
        stage_times["tree_build"] += time.perf_counter() - build_start

        compile_start = cuda_time()
        verify_ids, verify_pos, verify_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=template[0, 0],
            start=start,
            node_token_ids=node_ids,
            node_depths=node_depths,
            visibility_cpu=visibility,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_ids_buffer,
            verify_position_ids_buffer=verify_pos_buffer,
            attention_mask_buffer=attention_buffer,
            tree_visibility_buffer=visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - compile_start

        verify_start = cuda_time()
        output = target(
            verify_ids,
            position_ids=verify_pos,
            attention_mask=verify_mask,
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_start

        commit_start = cuda_time()
        all_target_probs = sampling_probs(output.logits[0], temperature)
        path_target_rows = []
        for nodes in path_nodes:
            # Root predicts depth 1; each path node predicts the following token.
            row_indices = [0] + nodes
            path_target_rows.append(all_target_probs[row_indices])
        target_by_path = torch.stack(path_target_rows)
        selected, skewed_q = gbv_select_path_and_probs(paths, target_by_path, draft_probs)
        selected_nodes = path_nodes[selected]
        accepted, bonus = block_rejection_sample(
            paths[selected], target_by_path[selected], skewed_q
        )
        keep_indices = [0] + selected_nodes[:accepted]
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=model.device)
        committed = verify_ids.index_select(1, keep_tensor)
        output_ids[:, start : start + committed.shape[1]] = committed
        output_ids[:, start + committed.shape[1]] = bonus
        compact_dynamic_cache(target_cache, start, keep_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, keep_tensor)

        advanced = len(keep_indices)
        acceptance_lengths.append(advanced)
        start += advanced
        stage_times["commit"] += cuda_time() - commit_start
        round_timestamps.append(cuda_time() - round_clock)
        if stop_tensor is not None:
            new = output_ids[:, start - advanced : start + 1]
            if torch.isin(new[0], stop_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_tensor is not None:
        indices = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(as_tuple=True)[0]
        if indices.numel():
            output_ids = output_ids[:, : num_input_tokens + indices[0] + 1]
    output_tokens = output_ids.shape[1] - num_input_tokens
    decode_time = cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_tokens,
        time_to_first_token=ttft,
        time_per_output_token=decode_time / max(output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
    )
