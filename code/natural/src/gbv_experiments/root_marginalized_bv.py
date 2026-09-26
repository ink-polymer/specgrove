"""Root-marginalized BV: audited reference and tensorized inference paths.

Uses standard BV as an attributed building block. See
docs/TREE_COUPLING_RESEARCH.md for the composition proof and claim limitations.
No CUDA speedup is implied by the CPU correctness tests.
"""
import torch

from . import sampling
from .config import ROOT_MARGINAL_METHODS as METHODS


def propose(q, branches, generator=None):
    """Preselect distinct roots, then sample ONE independent product suffix.

    Identical suffix tokens under different roots must retain distinct KV rows.
    Top-k ties follow torch.topk; roots are always chosen before suffix sampling.
    """
    if q.ndim != 2 or q.shape[0] < 1 or not 1 <= branches <= q.shape[1]:
        raise ValueError("invalid shared-suffix proposal shape/branch count")
    roots = q[0].topk(branches).indices
    suffix = sampling.sample(q[1:], generator) if q.shape[0] > 1 else roots[:0]
    return torch.cat((roots[:, None], suffix[None].expand(branches, -1)), dim=1)


def _validate(root_tokens, root_p, shared_path, branch_p, suffix_q, values=True):
    if root_p.ndim != 1 or root_tokens.ndim != 1 or shared_path.ndim != 1:
        raise ValueError("expected one-dimensional root and suffix inputs")
    k, m, vocab = root_tokens.numel(), shared_path.numel(), root_p.numel()
    if (k < 1 or vocab < 1 or branch_p.shape != (k, m + 1, vocab)
            or suffix_q.shape != (m, vocab)):
        raise ValueError("root-mixture tensor shape mismatch")
    if root_tokens.dtype != torch.long or shared_path.dtype != torch.long:
        raise ValueError("token tensors must have torch.long dtype")
    if any(t.device != root_p.device for t in (root_tokens, shared_path, branch_p, suffix_q)):
        raise ValueError("all tensors must be on the same device")
    if root_p.dtype != torch.float64 or branch_p.dtype != root_p.dtype or suffix_q.dtype != root_p.dtype:
        raise ValueError("root-marginal verification requires FP64 probability tensors")
    if not values:
        return
    if (root_tokens.unique().numel() != k or bool((root_tokens < 0).any())
            or bool((root_tokens >= vocab).any()) or bool((shared_path < 0).any())
            or bool((shared_path >= vocab).any())):
        raise ValueError("duplicate or out-of-range token id")
    for rows in (root_p, branch_p, suffix_q):
        sums = rows.sum(-1)
        if not bool(torch.isfinite(rows).all() & (rows >= 0).all()
                    & torch.isclose(sums, torch.ones_like(sums), rtol=0, atol=1e-12).all()):
            raise ValueError("probability rows must be finite, nonnegative, normalized")
    if m and not bool((suffix_q.gather(1, shared_path[:, None]) > 0).all()):
        raise ValueError("shared path has zero proposal probability")


def verify(root_tokens, root_p, shared_path, branch_p, suffix_q, generator=None,
           *, validate=True):
    """Audited serial reference returning (branch, accepted suffix, correction).

    Roots must be distinct and preselected without observing shared_path.
    branch_p[a,j] is p(next | context, root_tokens[a], shared_path[:j]);
    suffix_q[j] is the ACTUAL proposal conditional for shared_path[j].
    If branch=-1 emit correction alone; otherwise emit root, accepted suffix,
    correction. Preserve only that branch's accepted KV rows.
    """
    _validate(root_tokens, root_p, shared_path, branch_p, suffix_q, validate)
    m = shared_path.numel()
    covered = root_p[root_tokens]
    outside = root_p.clone()
    outside[root_tokens] = 0
    # Explicit complement summation retains tiny outside mass when covered.sum()
    # rounds to one. This is not an arbitrary-precision guarantee.
    gate = torch.stack((outside.sum(), covered.sum()))
    if int(sampling.sample(gate / gate.sum(), generator).item()) == 0:
        bonus = int(sampling.sample(outside / outside.sum(), generator).item())
        return -1, 0, bonus

    posterior = covered / covered.sum()
    posterior_rows, mixture_rows = [], []
    for j in range(m + 1):
        posterior_rows.append(posterior)
        mixture = (posterior[:, None] * branch_p[:, j]).sum(0)
        mixture_rows.append(mixture / mixture.sum())
        if j < m:
            updated = posterior * branch_p[:, j, shared_path[j]]
            total = updated.sum()
            # Conditional rows after a target-zero prefix are immaterial to BV:
            # the prefix acceptance weight has become zero and stays zero.
            posterior = updated / total if bool(total > 0) else posterior

    accepted, bonus = sampling.block_verify(
        shared_path, torch.stack(mixture_rows), suffix_q, generator)
    posterior = posterior_rows[accepted] * branch_p[:, accepted, bonus]
    if not bool(torch.isfinite(posterior).all() & (posterior.sum() > 0)):
        raise FloatingPointError("BV emitted a target-zero suffix or numerical underflow occurred")
    branch = int(sampling.sample(posterior / posterior.sum(), generator).item())
    return branch, accepted, bonus


def _normalize(weights):
    """Normalize nonnegative rows; use a harmless uniform unreachable row."""
    total = weights.sum(-1, keepdim=True)
    zero = total == 0
    safe = torch.where(zero, torch.ones_like(total), total)
    # Only a ZERO-mass unreachable row gets a fallback. Preserve NaNs/Infs so
    # multinomial fails, even on the trusted hot path, rather than hiding them.
    return torch.where(zero, torch.ones_like(weights) / weights.shape[-1], weights / safe)


def _mixture(root_tokens, root_p, shared_path, branch_p):
    # Prefix posterior via parallel scans, not a Python loop over depths.
    k, m = root_tokens.numel(), shared_path.numel()
    likelihoods = branch_p[:, :m].gather(
        2, shared_path[None, :, None].expand(k, -1, 1)).squeeze(-1)
    prefix_log = torch.cat((root_p.new_zeros((k, 1)), likelihoods.log().cumsum(1)), dim=1)
    log_joint = root_p[root_tokens].log()[:, None] + prefix_log
    reachable = torch.isfinite(log_joint).any(0)
    # All-zero target prefixes cannot be accepted. Complete their conditionals
    # arbitrarily without creating NaNs that would poison subsequent kernels.
    safe_joint = torch.where(reachable[None], log_joint, torch.zeros_like(log_joint))
    posterior = safe_joint.softmax(dim=0)
    mixture = torch.einsum("kj,kjv->jv", posterior, branch_p)
    return posterior, _normalize(mixture)


def _terminal_rows(path, p, q, rule):
    """Marginalize verifier draws into (terminal-depth law, correction rows).

    BV chooses the deepest successful row. If its row success probabilities
    are h_i, Pr(tau=i)=h_i prod_{j>i}(1-h_j). No discarded vocabulary draws
    need to be materialized. The token ablation stops at the first rejection.
    """
    selected_p = p[:-1].gather(1, path[:, None]).squeeze(-1)
    selected_q = q.gather(1, path[:, None]).squeeze(-1)
    if rule == "bv":
        cumulative = torch.cat((p.new_zeros(1), (selected_p.log() - selected_q.log()).cumsum(0)))
        # w_i = exp(s_i - max_{0<=j<=i} s_j), equivalent to
        # w_{i+1}=min(1, w_i*p_i[z_i]/q_i[z_i]). A zero stays zero.
        w = (cumulative - cumulative.cummax(0).values).exp()
        proposal = torch.cat((q, p.new_zeros((1, p.shape[1]))), dim=0)
        residual = (w[:, None] * p - proposal).clamp_min(0)
        mass, reject = residual.sum(-1), (1 - w).clamp_min(0)
        total = mass + reject
        safe = torch.where(total > 0, total, torch.ones_like(total))
        success = torch.where(total > 0, mass / safe, torch.ones_like(total))
        failure = torch.where(total > 0, reject / safe, torch.zeros_like(total))
        # Compute failure directly, rather than 1-success, to retain small tails.
        later_failure = torch.cat((failure[1:].flip(0).cumprod(0).flip(0), p.new_ones(1)))
        endpoint = success * later_failure
        rows = torch.where((total == 0)[:, None], p, residual)
    elif rule == "token":
        accept = (selected_p / selected_q).clamp(max=1)
        survive = torch.cat((p.new_ones(1), accept.cumprod(0)))
        endpoint = survive * torch.cat(((1 - accept).clamp_min(0), p.new_ones(1)))
        rows = torch.cat(((p[:-1] - q).clamp_min(0), p[-1:]), dim=0)
    else:
        raise ValueError("unknown suffix verification rule")
    return _normalize(endpoint), rows


def _draw_terminal(root_tokens, root_p, path, p, q, generator, rule):
    endpoint, rows = _terminal_rows(path, p, q, rule)
    outside = root_p.clone()
    outside[root_tokens] = 0
    # Fuse the root-coverage gate and suffix terminal choice into one draw.
    weights = torch.cat((outside.sum().reshape(1), root_p[root_tokens].sum() * endpoint))
    selected = sampling.sample(_normalize(weights), generator)
    inside = selected > 0
    depth = (selected - 1).clamp_min(0)
    token_row = torch.where(inside, rows[depth], outside)
    token = sampling.sample(_normalize(token_row), generator)
    return inside, depth, token


def verify_tensorized(root_tokens, root_p, shared_path, branch_p, suffix_q,
                      generator=None, *, validate=True, rule="bv"):
    """Same BV emitted-block law as verify(), with three categorical draws.

    validate=False is only for trusted Engine tensors (softmax + propose).
    There is one explicit result-to-host conversion; this is NOT a claim about
    internal synchronization in PyTorch/CUDA multinomial or measured latency.
    rule='token' is the exact token-verification ablation, not a BV speed result.
    """
    _validate(root_tokens, root_p, shared_path, branch_p, suffix_q, validate)
    posterior, p = _mixture(root_tokens, root_p, shared_path, branch_p)
    inside, depth, token = _draw_terminal(
        root_tokens, root_p, shared_path, p, suffix_q, generator, rule)
    root_weights = posterior[:, depth] * branch_p[:, depth, token]
    branch = sampling.sample(_normalize(root_weights), generator)
    result = torch.stack((torch.where(inside, branch, -1),
                          torch.where(inside, depth, 0), token))
    return tuple(result.tolist())


def verify_early_root(root_tokens, root_p, shared_path, branch_p, suffix_q,
                      generator=None, *, validate=True):
    """Same tree/BV terminal implementation, but choose root BEFORE suffix BV.

    A valid ablation: shared suffix Q is independent of root, so ordinary BV on
    the chosen target branch remains exact. Three draws, as in the full method.
    """
    _validate(root_tokens, root_p, shared_path, branch_p, suffix_q, validate)
    branch = sampling.sample(_normalize(root_p[root_tokens]), generator)
    inside, depth, token = _draw_terminal(
        root_tokens, root_p, shared_path, branch_p[branch], suffix_q, generator, "bv")
    result = torch.stack((torch.where(inside, branch, -1),
                          torch.where(inside, depth, 0), token))
    return tuple(result.tolist())
