"""Auditable fixed-length natural-task selection, never truncating task tokens."""
import hashlib
import json

import torch

from gbv_experiments.conversation import encode_messages


def pad_natural_ids(ids, header, newline_id, target=256):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > target:
        raise ValueError('Invalid input shape or original prompt exceeds target')
    if not torch.equal(ids[:, :header.shape[1]], header):
        raise ValueError('Chat header boundary mismatch; refusing to alter task')
    length = header.shape[1]
    filler = ids.new_full((1, target - ids.shape[1]), newline_id)
    result = torch.cat((ids[:, :length], filler, ids[:, length:]), dim=1)
    recovered = torch.cat((result[:, :length], result[:, length + filler.shape[1]:]), dim=1)
    if result.shape[1] != target or not torch.equal(recovered, ids):
        raise RuntimeError('Original natural task tokens were not preserved')
    return result


def select_input256(tokenizer, cfg, path, count=32):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header_text = '<|im_start|>user\n'
    header = torch.tensor([tokenizer.encode(header_text, add_special_tokens=False)])
    newline = tokenizer.encode('\n', add_special_tokens=False)
    if len(newline) != 1:
        raise ValueError('This fixed-length protocol requires a single newline token')
    prompts, identities, skipped, source_ids = [], [], [], set()
    for index, row in enumerate(rows):
        ids = encode_messages(tokenizer, [{'role': 'user', 'content': row['prompt']}], cfg, 'cpu')
        original = ids.shape[1]
        if original > 256:
            skipped.append({'index': index, 'source_id': row['source_id'], 'original_tokens': original})
            continue
        if str(row['source_id']) in source_ids:
            raise ValueError('Duplicate request identity')
        source_ids.add(str(row['source_id']))
        padded = pad_natural_ids(ids, header, newline[0])
        identities.append({'dataset': row['dataset'], 'source_id': row['source_id'],
            'original_index': index, 'original_prompt_sha256': row['prompt_sha256'],
            'original_tokens': original, 'input_tokens': 256, 'filler_tokens': 256 - original,
            'input_token_sha256': hashlib.sha256(json.dumps(padded.tolist()).encode()).hexdigest(),
            'prompt_preserved': True})
        prompts.append(padded)
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise ValueError(f'{path.name}: only {len(prompts)} eligible tasks; need {count}')
    return prompts, {'dataset': path.stem, 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'selection': 'first eligible in original order, original chat-token length <=256; no output-based selection',
        'filler': 'explicit newline tokens immediately after user header; attended, not masked padding',
        'identities': identities, 'skipped_before_selection_complete': skipped,
        'quality_protocol': 'modified fixed-length prompts; do not pool with original-prompt quality scores'}
