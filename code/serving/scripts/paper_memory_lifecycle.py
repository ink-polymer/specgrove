"""Release completed decode reference cycles between trials, never inside decode."""
import gc
import time
import torch

POLICY_VERSION = 'completed-decode-gc-20260917-v1'

def completed_cell_key(case):
    return tuple(case[k] for k in ('phase','dataset','temperature','budget','concurrency','max_new_tokens','prompt_tokens'))

def complete_legacy_cells(directory):
    if directory is None or not (directory/'contract.json').exists():
        return set(), []
    import json
    import hashlib
    from collections import Counter
    planned=Counter()
    for case in json.loads((directory/'contract.json').read_text())['cases']:
        planned[completed_cell_key(case)]+=(case['requests']+case['concurrency']-1)//case['concurrency']
    measured=Counter()
    for path in (directory/'groups').glob('*.json'):
        record=json.loads(path.read_text())
        name=hashlib.sha256(json.dumps(record['identity'],sort_keys=True).encode()).hexdigest()[:24]+'.json'
        if path.name!=name:
            raise RuntimeError('Legacy identity filename mismatch')
        measured[completed_cell_key(record['identity']['case'])]+=1
    if any(measured[k]>planned.get(k,0) for k in measured):
        raise RuntimeError('Legacy cell exceeds registered group count')
    complete={k for k,n in planned.items() if measured[k]==n}
    partial=[{'cell':list(k),'measured_groups':measured[k],'planned_groups':n}
             for k,n in planned.items() if 0<measured[k]<n]
    return complete, partial

def reclaim_completed_decodes(engine, *, release_cached_blocks=True):
    """GPU sync and GC before/after measured trials; do not change request RNG."""
    engine.sync()
    started=time.perf_counter()
    device=torch.device(engine.device)
    is_cuda=device.type=='cuda'
    before_allocated=torch.cuda.memory_allocated(device) if is_cuda else 0
    before_reserved=torch.cuda.memory_reserved(device) if is_cuda else 0
    objects=gc.collect()
    if is_cuda and release_cached_blocks:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    engine.sync()
    after_allocated=torch.cuda.memory_allocated(device) if is_cuda else 0
    after_reserved=torch.cuda.memory_reserved(device) if is_cuda else 0
    return {'policy':POLICY_VERSION,'collected_objects':objects,
            'allocated_before_bytes':before_allocated,'allocated_after_bytes':after_allocated,
            'reserved_before_bytes':before_reserved,'reserved_after_bytes':after_reserved,
            'allocated_freed_bytes':before_allocated-after_allocated,
            'allocator_cache_released':bool(is_cuda and release_cached_blocks),
            'maintenance_ms':1000*(time.perf_counter()-started),
            'included_in_decode_wall_ms':False}
