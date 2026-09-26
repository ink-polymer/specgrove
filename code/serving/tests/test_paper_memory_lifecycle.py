"""Memory lifecycle repair checks; no decoder arithmetic or RNG changes."""
import gc
import hashlib
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from paper_memory_lifecycle import reclaim_completed_decodes,complete_legacy_cells,completed_cell_key
from gbv_experiments.continuous_tree_block_decode import _PersistentRequestCachePool

class Engine:
    device='cpu'
    def sync(self):pass

class Cache:
    def __init__(self):
        self.layers=[SimpleNamespace(keys=torch.ones(1,2,4,3),values=torch.ones(1,2,4,3))]
    def get_seq_length(self):return 4

def test_real_pool_reference_cycle_is_reclaimed_after_decode():
    enabled=gc.isenabled()
    gc.disable()
    try:
        pool=_PersistentRequestCachePool([Cache(),Cache()],capacity=8)
        pool_ref=weakref.ref(pool);storage_ref=weakref.ref(pool.key_storage[0])
        del pool
        assert pool_ref() is not None and storage_ref() is not None
        result=reclaim_completed_decodes(Engine())
        assert pool_ref() is None and storage_ref() is None
        assert result['collected_objects']>0
        assert result['included_in_decode_wall_ms'] is False
    finally:
        if enabled:gc.enable()

def test_cleanup_never_collects_live_pool_or_changes_its_bytes():
    pool=_PersistentRequestCachePool([Cache(),Cache()],capacity=8)
    before=pool.key_storage[0].clone()
    reclaim_completed_decodes(Engine())
    assert torch.equal(pool.key_storage[0],before)
    assert [cache.get_seq_length() for cache in pool.caches]==[4,4]
    assert pool.caches[0].pool is pool

def test_cleanup_does_not_consume_global_or_request_rng():
    rng=torch.Generator().manual_seed(91)
    request=rng.get_state().clone()
    global_rng=torch.random.get_rng_state().clone()
    reclaim_completed_decodes(Engine())
    assert torch.equal(request,rng.get_state())
    assert torch.equal(global_rng,torch.random.get_rng_state())

def case(seed=17,c=1):
    return dict(phase='context',dataset='gsm8k',temperature=1,budget=193,
                concurrency=c,max_new_tokens=128,prompt_tokens=8192,
                requests=2,repeats=3,seed=seed)

def save_group(directory,c,group):
    identity={'case':c,'group':group,'first':group,'data_sha256':'hash'}
    name=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]+'.json'
    (directory/'groups'/name).write_text(json.dumps({'identity':identity}))

def test_reuse_only_whole_cell_across_seeds_not_partial_seed(tmp_path):
    (tmp_path/'groups').mkdir()
    a=case();b=case(29);partial=case(c=2)
    (tmp_path/'contract.json').write_text(json.dumps({'cases':[a,b,partial]}))
    for group in [0,1]:save_group(tmp_path,a,group)
    whole,incomplete=complete_legacy_cells(tmp_path)
    assert not whole and len(incomplete)==1
    for group in [0,1]:save_group(tmp_path,b,group)
    whole,incomplete=complete_legacy_cells(tmp_path)
    assert whole=={completed_cell_key(a)} and not incomplete

def test_reuse_rejects_misnamed_record(tmp_path):
    (tmp_path/'groups').mkdir()
    c=case()
    (tmp_path/'contract.json').write_text(json.dumps({'cases':[c]}))
    (tmp_path/'groups/bad.json').write_text(json.dumps({'identity':{'case':c}}))
    with pytest.raises(RuntimeError,match='filename mismatch'):complete_legacy_cells(tmp_path)

def test_original_core_and_both_previous_drivers_remain_frozen():
    expected={'src/gbv_experiments/continuous_tree_block_decode.py':'3f132a41b659d1306cbfc3fbb348c0f89284d097d69851f0f73d8a6e4518587f',
              'scripts/run_dp_paper_study.py':'02f9f422293f09e7a1ff0cbc38a61b52365cd9cea6c74ac8f6da7b83683325a3',
              'scripts/run_dp_paper_single_wave.py':'750cdae8a4bf375e99fdee16a18e0f7fcf397b7f26266a64600442b0a90a0468'}
    for path,digest in expected.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
