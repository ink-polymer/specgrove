"""Atom-coupled tree block verification: experimental, not a speed/novelty claim.

A shared latent atom maps to DIFFERENT tokens on different branches. The actual
joint proposal, including its correlations, is retained through verification.
Top-down protected flow and the attributed LV backward construction operate on
these sparse atoms. Only the selected exit needs a full vocabulary residual.
See docs/ATOM_TREE_BV.md for assumptions, proof and limitations.
"""
from dataclasses import dataclass
import math

import torch

from . import sampling
from .root_marginalized_bv import _normalize


SUPPORT_SIZE = 8  # Frozen development choice, not tuned on evaluation prompts.
COUPLING_MODES = frozenset({"aligned", "fixed", "depth_permuted"})


@dataclass
class AtomProposal:
    roots: torch.Tensor                 # K distinct first tokens
    tokens: torch.Tensor                # K x M x R unique support tokens per row
    slots: torch.Tensor                 # K x M x C atom -> support slot
    source: torch.Tensor                # M x C actual atom probabilities
    draws: torch.Tensor                 # M sampled atom indices

    def marginals(self):
        return self.source.new_zeros(self.tokens.shape).scatter_add_(
            -1, self.slots, self.source[None].expand_as(self.slots))

    def paths(self):
        k, m, _ = self.tokens.shape
        slot = self.slots.gather(2, self.draws[None, :, None].expand(k, m, 1))
        suffix = self.tokens.gather(2, slot).squeeze(-1)
        return torch.cat((self.roots[:, None], suffix), dim=1)

    def validate(self, vocab, *, values=True):
        if self.tokens.ndim != 3 or self.source.ndim != 2:
            raise ValueError("Expected K x M x R supports and M x C source")
        k, m, r = self.tokens.shape
        c = self.source.shape[1]
        if (min(k, r, c, vocab) < 1 or self.roots.shape != (k,)
                or self.slots.shape != (k, m, c) or self.source.shape[0] != m
                or self.draws.shape != (m,)):
            raise ValueError("Atom proposal shape mismatch")
        if (self.source.dtype != torch.float64
                or any(t.dtype != torch.long for t in (self.roots, self.tokens, self.slots, self.draws))
                or any(t.device != self.source.device for t in (self.roots, self.tokens, self.slots, self.draws))):
            raise ValueError("Atom proposal requires same-device FP64 probabilities and long indices")
        if not values:
            return
        if (self.roots.unique().numel() != k or bool((self.roots < 0).any())
                or bool((self.roots >= vocab).any()) or bool((self.tokens < 0).any())
                or bool((self.tokens >= vocab).any())
                or (r > 1 and bool((self.tokens.sort(-1).values.diff(dim=-1) == 0).any()))):
            raise ValueError("Roots/support tokens must be distinct and in range")
        if bool((self.slots < 0).any() | (self.slots >= r).any()
                | (self.draws < 0).any() | (self.draws >= c).any()):
            raise ValueError("Atom index out of range")
        _check_rows(self.source)
        if m and not bool((self.source.gather(1, self.draws[:, None]) > 0).all()):
            raise ValueError("Sampled atom has zero proposal mass")


def _check_rows(p):
    if not bool(torch.isfinite(p).all() & (p >= 0).all()
                & torch.isclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-12, rtol=0).all()):
        raise ValueError("Probability rows must be finite, nonnegative and normalized")


def couple(roots, tokens, weights, generator=None, *, shifts=None):
    """Partition [0,1) at every shifted branch CDF boundary; retain actual masses.

    Supports/weights are fixed BEFORE latent draws. Branch marginals may differ.
    Equal shifts recover aligned coupling; a/K shifts give stratified coupling.
    Duplicate token mappings across atoms are expected, unlike duplicate support
    tokens within one branch/position. Zero-width atoms are harmless padding.
    """
    if tokens.ndim != 3 or weights.shape != tokens.shape or weights.dtype != torch.float64:
        raise ValueError("Expected matching K x M x R supports/FP64 weights")
    k, m, r = tokens.shape
    if min(k, r) < 1 or roots.shape != (k,):
        raise ValueError("Invalid root/support shape")
    _check_rows(weights)
    if shifts is None:
        shifts = torch.arange(k, device=weights.device, dtype=weights.dtype) / k
    if shifts.shape == (k,):
        shifts = shifts[:, None].expand(k, m)
    if (shifts.shape != (k, m) or shifts.device != weights.device
            or shifts.dtype != weights.dtype or not bool(torch.isfinite(shifts).all())):
        raise ValueError("Invalid coupling shifts")
    shifts = shifts.remainder(1)
    cdf = weights.cumsum(-1)
    cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), -1)
    shifted = (cdf - shifts[:, :, None]).remainder(1)
    boundaries = torch.cat((weights.new_zeros((m, 1)), weights.new_ones((m, 1)),
                            shifted.permute(1, 0, 2).reshape(m, k * r)), -1).sort(-1).values
    source = boundaries.diff(dim=-1)
    source = source / source.sum(-1, keepdim=True)
    middle = (boundaries[:, :-1] + boundaries[:, 1:]) / 2
    coordinates = (middle[None] + shifts[:, :, None]).remainder(1)
    slots = torch.searchsorted(cdf.contiguous(), coordinates.contiguous(), right=True).clamp_max(r - 1)
    draws = sampling.sample(source, generator) if m else roots.new_empty(0)
    result = AtomProposal(roots, tokens, slots, source, draws)
    # Full token-range validation occurs with the actual target vocabulary.
    return result


def depth_permuted_shifts(branches, positions, *, device=None, dtype=torch.float64):
    """Assign quantile strata to roots with a different permutation by depth.

    Multiplication by a unit modulo ``branches`` is a permutation, so every
    depth uses the same low-discrepancy set of strata.  Cycling the available
    units changes which root receives which stratum and avoids locking a root
    to one quantile band for its entire proposed path.
    """
    if (type(branches) is not int or type(positions) is not int
            or branches < 1 or positions < 0):
        raise ValueError("Invalid branch/position count")
    branch = torch.arange(branches, device=device, dtype=torch.long)
    units = torch.tensor(
        [value for value in range(1, branches) if math.gcd(value, branches) == 1] or [1],
        device=device, dtype=torch.long,
    )
    multiplier = units[torch.arange(positions, device=device) % units.numel()]
    return ((branch[:, None] * multiplier[None]) % branches).to(dtype) / branches


def propose(q, branches, generator=None, *, coupling="depth_permuted",
            support_size=SUPPORT_SIZE):
    if q.ndim != 2 or q.shape[0] < 1 or not 1 <= branches <= q.shape[1]:
        raise ValueError("Invalid draft shape/branch count")
    if (type(support_size) is not int or support_size < 1 or q.dtype != torch.float64
            or coupling not in COUPLING_MODES):
        raise ValueError("Positive support size and FP64 draft probabilities required")
    roots = q[0].topk(branches).indices
    weights, tokens = q[1:].topk(min(support_size, q.shape[1]), dim=-1)
    weights = _normalize(weights)
    if coupling == "aligned":
        shifts = q.new_zeros((branches, q.shape[0] - 1))
    elif coupling == "fixed":
        shifts = torch.arange(branches, device=q.device, dtype=q.dtype) / branches
    else:
        shifts = depth_permuted_shifts(
            branches, q.shape[0] - 1, device=q.device, dtype=q.dtype,
        )
    return couple(roots, tokens[None].expand(branches, -1, -1),
                  weights[None].expand(branches, -1, -1), generator, shifts=shifts)


@dataclass
class FlowPlan:
    floors: torch.Tensor
    scores: torch.Tensor
    flow: torch.Tensor
    residual_mass: torch.Tensor
    endpoint: torch.Tensor


def plan(alpha, support_p, proposal, *, pool=True):
    """O(K M C) flow, with no K x M x vocabulary residual allocation.

    support_p[a,j,r] is target probability at branch a's OWN sampled prefix.
    Target mass outside the sparse proposal is not truncated from the output.
    """
    k, m, _ = proposal.tokens.shape
    source = proposal.source
    token_marginal = proposal.marginals()
    # Listed zero-probability slots belong to the target-only residual, too.
    support_p = torch.where(token_marginal > 0, support_p, torch.zeros_like(support_p))
    marginal = token_marginal.gather(-1, proposal.slots)
    safe = torch.where(marginal > 0, marginal, torch.ones_like(marginal))
    lifted = support_p.gather(-1, proposal.slots) * (source[None] / safe)
    outside_support = (1 - support_p.sum(-1)).clamp_min(0)
    floors, scores, flows, residuals = [alpha], [alpha], [], []
    for j in range(m):
        base = torch.minimum(floors[-1][:, None] * lifted[:, j], alpha[:, None] * source[j])
        capacity = scores[-1][:, None] * lifted[:, j]
        extra = (capacity - base).clamp_min(0)
        left = (source[j] - base.sum(0)).clamp_min(0)
        total = extra.sum(0)
        factor = (left / torch.where(total > 0, total, torch.ones_like(total))).clamp_max(1)
        flow = base + extra * factor if pool else base
        # Stable nonnegative parts; do not subtract two near-one total masses.
        residuals.append(scores[-1] * outside_support[:, j] + (capacity - flow).clamp_min(0).sum(-1))
        atom = proposal.draws[j]
        floors.append(base[:, atom] / source[j, atom])
        scores.append(flow[:, atom] / source[j, atom])
        flows.append(flow)
    residuals.append(scores[-1])  # Bonus row: source and continuation flow are zero.
    score = torch.stack(scores, 1)
    residual = torch.stack(residuals, 1)
    failure_mass = (1 - score.sum(0)).clamp_min(0)
    mass = residual.sum(0)
    total = failure_mass + mass
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    success = torch.where(total > 0, mass / denominator, torch.ones_like(total))
    failure = torch.where(total > 0, failure_mass / denominator, torch.zeros_like(total))
    later = torch.cat((failure[1:].flip(0).cumprod(0).flip(0), alpha.new_ones(1)))
    return FlowPlan(torch.stack(floors, 1), score,
                    torch.stack(flows, 1) if m else alpha.new_zeros((k, 0, source.shape[1])),
                    residual, _normalize(success * later))


def _draw(root_p, proposal, state, probability_row, generator):
    """One joint (depth, branch) draw, then ONE full-vocabulary correction draw."""
    k, m, _ = proposal.tokens.shape
    covered = root_p[proposal.roots]
    outside = root_p.clone()
    outside[proposal.roots] = 0
    # Normalize branch masses at each depth, with harmless unreachable-row completion.
    branch_at_depth = _normalize(state.residual_mass.T).T
    joint = covered.sum() * state.endpoint[None] * branch_at_depth
    selected = sampling.sample(_normalize(torch.cat((outside.sum()[None], joint.reshape(-1)))), generator)
    inside = selected > 0
    index = (selected - 1).clamp_min(0)
    branch, depth = index // (m + 1), index % (m + 1)
    row = probability_row(branch, depth)
    flow = torch.cat((state.flow, root_p.new_zeros((k, 1, proposal.source.shape[1]))), 1)[branch, depth]
    atom_tokens = proposal.tokens.gather(-1, proposal.slots)
    atom_tokens = torch.cat((atom_tokens, proposal.roots.new_zeros((k, 1, proposal.source.shape[1]))), 1)
    correction = (state.scores[branch, depth] * row).scatter_add(0, atom_tokens[branch, depth], -flow).clamp_min(0)
    token = sampling.sample(_normalize(torch.where(inside, correction, outside)), generator)
    return tuple(torch.stack((torch.where(inside, branch, -1),
                              torch.where(inside, depth, 0), token)).tolist())


def verify(root_p, branch_p, proposal, generator=None, *, pool=True, validate=True):
    """Probability-input reference/API. Returns branch, accepted suffix, correction."""
    k, m, _ = proposal.tokens.shape
    proposal.validate(root_p.numel(), values=validate)
    if (root_p.ndim != 1 or branch_p.shape != (k, m + 1, root_p.numel())
            or root_p.dtype != torch.float64 or branch_p.dtype != root_p.dtype
            or root_p.device != proposal.source.device or branch_p.device != root_p.device):
        raise ValueError("Target probability shape/dtype/device mismatch")
    if validate:
        _check_rows(root_p)
        _check_rows(branch_p)
    support_p = branch_p[:, :m].gather(-1, proposal.tokens)
    state = plan(_normalize(root_p[proposal.roots]), support_p, proposal, pool=pool)
    return _draw(root_p, proposal, state, lambda a, j: branch_p[a, j], generator)


def verify_logits(root_logits, branch_logits, proposal, temperature, generator=None,
                  *, pool=True, validate=True):
    """Production path: normalize candidate entries; materialize only one exit row.

    Full logit normalizers still read the vocabulary (O(K M V)); this does not
    eliminate the target LM head. No claim of O(K M C) total model work is made.
    """
    k, m, _ = proposal.tokens.shape
    proposal.validate(root_logits.numel(), values=validate)
    if (root_logits.ndim != 1 or branch_logits.shape != (k, m + 1, root_logits.numel())
            or root_logits.device != proposal.source.device or branch_logits.device != root_logits.device
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("Target logit shape/device/temperature mismatch")
    if validate:
        for rows in (root_logits, branch_logits):
            if bool(torch.isnan(rows).any() | torch.isposinf(rows).any()
                    | ~torch.isfinite(rows).any(-1).all()):
                raise ValueError("Invalid target logits")
    root_p = sampling.probabilities(root_logits, temperature, torch.float64)
    if temperature == 0:
        support_p = proposal.tokens.eq(branch_logits[:, :m].argmax(-1, keepdim=True)).double()
    else:
        # The FP64 scaled-logit temporary is released before flow/residual work.
        scaled = branch_logits[:, :m].double() / temperature
        normalizer = torch.logsumexp(scaled, -1, keepdim=True)
        support_p = (scaled.gather(-1, proposal.tokens) - normalizer).exp()
        del scaled
    state = plan(_normalize(root_p[proposal.roots]), support_p, proposal, pool=pool)
    return _draw(root_p, proposal, state,
                 lambda a, j: sampling.probabilities(branch_logits[a, j], temperature, torch.float64), generator)
