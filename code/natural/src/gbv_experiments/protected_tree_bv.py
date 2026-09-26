"""Baseline-protected shared-suffix tree block verification (research candidate).

Reserve each early-root BV flow before pooling unused proposal capacity. The
real-arithmetic prefix-survival guarantee is described in docs/PROTECTED_TREE_BV.md.
The backward flow sampler is attributed to LV; this is a new candidate forward
construction, not a claim of established novelty or measured CUDA acceleration.
"""
import torch

from . import sampling
from .root_marginalized_bv import _normalize, _validate


def _scores(alpha, path, branch_p, q):
    """Compute protected early-BV floors and actual scores along this prefix.

    Only K-vectors are sequential. Full vocabulary work is batched afterwards.
    No host scalar conversions occur in this recurrence.
    """
    k, m = alpha.numel(), path.numel()
    chosen_p = branch_p[:, :m].gather(2, path[None, :, None].expand(k, -1, 1)).squeeze(-1)
    chosen_q = q.gather(1, path[:, None]).squeeze(-1)
    log_ratio = chosen_p.log() - chosen_q.log()[None]
    cumulative = torch.cat((alpha.new_zeros((k, 1)), log_ratio.cumsum(1)), dim=1)
    floors = alpha[:, None] * (cumulative - cumulative.cummax(1).values).exp()
    scores = [alpha]
    for j in range(m):
        base = floors[:, j + 1]
        extra_cap = (scores[-1] * chosen_p[:, j] - chosen_q[j] * base).clamp_min(0)
        source_left = (1 - base.sum()).clamp_min(0)
        cap_sum = extra_cap.sum()
        # Divide by q only when total extra capacity fits the remaining source.
        # Otherwise the source-limited allocation uses normalized capacities.
        increment = torch.where(
            cap_sum > chosen_q[j] * source_left,
            source_left * _normalize(extra_cap), extra_cap / chosen_q[j],
        )
        scores.append(base + increment)
    return floors, torch.stack(scores, dim=1)


def _residuals(alpha, floors, scores, branch_p, q):
    proposal = torch.cat((q, q.new_zeros((1, branch_p.shape[-1]))), dim=0)
    capacity = scores[:, :, None] * branch_p
    reserved = torch.minimum(floors[:, :, None] * branch_p,
                             alpha[:, None, None] * proposal[None])
    remaining = (capacity - reserved).clamp_min(0)
    source_left = (proposal - reserved.sum(0)).clamp_min(0)
    cap_sum = remaining.sum(0)
    # Direct residual fraction retains small tails; avoids capacity - flow.
    safe = torch.where(cap_sum > 0, cap_sum, torch.ones_like(cap_sum))
    fraction = (cap_sum - source_left).clamp_min(0) / safe
    return remaining * fraction[None], capacity


def _prepare(alpha, shared_path, branch_p, suffix_q):
    """Pure deterministic flow stage; isolated for same-law CUDA fusion tests."""
    floors, scores = _scores(alpha, shared_path, branch_p, suffix_q)
    residual, capacity = _residuals(alpha, floors, scores, branch_p, suffix_q)
    return scores, residual, capacity


def verify(root_tokens, root_p, shared_path, branch_p, suffix_q, generator=None,
           *, validate=True):
    """Return (branch, accepted_suffix, correction), using three draws.

    Same proposal contract as RM-BV: distinct roots chosen BEFORE the shared
    suffix. Unlike RM-BV, the root is drawn from the joint RESIDUAL flow, not
    the target posterior. Swapping those distributions would be biased.
    """
    _validate(root_tokens, root_p, shared_path, branch_p, suffix_q, validate)
    covered = root_p[root_tokens]
    alpha = _normalize(covered)
    scores, residual, capacity = _prepare(alpha, shared_path, branch_p, suffix_q)
    mass = residual.sum((0, 2))
    failure_mass = (1 - scores.sum(0)).clamp_min(0)
    total = mass + failure_mass
    safe = torch.where(total > 0, total, torch.ones_like(total))
    success = torch.where(total > 0, mass / safe, torch.ones_like(total))
    failure = torch.where(total > 0, failure_mass / safe, torch.zeros_like(total))
    later_failure = torch.cat((failure[1:].flip(0).cumprod(0).flip(0), root_p.new_ones(1)))
    endpoint = _normalize(success * later_failure)
    outside = root_p.clone()
    outside[root_tokens] = 0
    selected = sampling.sample(_normalize(torch.cat((outside.sum()[None], covered.sum() * endpoint))), generator)
    inside, depth = selected > 0, (selected - 1).clamp_min(0)
    # Zero-total rows are unreachable. Complete them with their target capacity.
    joint = torch.where(total[depth] > 0, residual[:, depth], capacity[:, depth])
    token = sampling.sample(_normalize(torch.where(inside, joint.sum(0), outside)), generator)
    branch = sampling.sample(_normalize(joint[:, token]), generator)
    return tuple(torch.stack((torch.where(inside, branch, -1),
                              torch.where(inside, depth, 0), token)).tolist())
