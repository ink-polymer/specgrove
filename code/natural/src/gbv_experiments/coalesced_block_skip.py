"""Causal prefix-coalesced block-flow verification for sampled DFlash tries.

The verifier keeps the exact latent-atom proposal used by
``diffusion_tree_bv`` but changes how scarce proposal flow is assigned.  At a
given depth and atom, labels that map the same verified prefix to the same
child token are given priority.  Remaining ties prefer the largest immediate
Target continuation capacity.  The allocation is causal: it reads only the
already-realized prefix, the current atom mapping, and the Target row at that
prefix.  It never reads a future atom or future Target row.

After the forward flow is built, one joint deepest endpoint is sampled.  A
selected endpoint commits its whole prefix block, so intermediate token-wise
verification decisions and all unrelated branches are skipped.  The Target
transformer still evaluates the sampled trie in one packed forward.

This is an experimental allocation rule, not a literature-novelty claim.  Its
losslessness follows from the same finite flow-conservation conditions used by
the diffusion trie block verifier: protected flow <= allocated flow <= Target
capacity, total outgoing flow for every proposal atom <= that atom's source
mass, and the exact full-vocabulary residual is sampled at the endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from . import atom_tree_bv as transport
from . import sampling
from .diffusion_tree_bv import DiffusionProposal
from .root_marginalized_bv import _normalize
from .tree import Tree, sampled_tree


@dataclass
class CoalescedBlockSkipPlan:
    floors: torch.Tensor
    scores: torch.Tensor
    flow: torch.Tensor
    residual_mass: torch.Tensor
    endpoint: torch.Tensor
    prioritized_extra_mass: torch.Tensor
    coalesced_label_atom_fraction: torch.Tensor


def entropy_coalesced_shifts(law, branches: int) -> torch.Tensor:
    """Continuously trade quantile coverage for shared-prefix probability.

    Every labelled sample still has exactly ``law.weights`` as its marginal:
    adding any deterministic shift to one common uniform variable preserves a
    uniform variable.  Normalized entropy is therefore free to control only
    *dependence* between labels.  A sharp denoising row collapses the shifts
    toward a common draw; a flat row recovers the depth-permuted stratification.
    """

    if type(branches) is not int or branches < 1:
        raise ValueError("Positive branch count required")
    weights = law.weights
    if weights.ndim != 2 or weights.dtype != torch.float64:
        raise ValueError("Entropy coupling requires FP64 diffusion weights")
    base = transport.depth_permuted_shifts(
        branches, weights.shape[0], device=weights.device, dtype=weights.dtype,
    )
    if weights.shape[-1] == 1:
        normalized_entropy = weights.new_zeros(weights.shape[0])
    else:
        normalized_entropy = -(
            torch.where(weights > 0, weights * weights.log(), torch.zeros_like(weights))
        ).sum(-1) / math.log(weights.shape[-1])
    return base * normalized_entropy.clamp(0, 1)[None]


def propose(law, branches: int, generator: torch.Generator | None = None):
    """Draw correlated DFlash blocks with entropy-adaptive coalescence."""

    shifts = entropy_coalesced_shifts(law, branches)
    atoms = transport.couple(
        torch.arange(branches, device=law.weights.device),
        law.tokens[None].expand(branches, -1, -1),
        law.weights[None].expand(branches, -1, -1),
        generator,
        shifts=shifts,
    )
    return DiffusionProposal(law, atoms.slots, atoms.source, atoms.draws)


def _priority_extra(
    extra: torch.Tensor,
    left: torch.Tensor,
    capacity: torch.Tensor,
    source: torch.Tensor,
    prefix_nodes: torch.Tensor,
    mapped_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate each atom's remaining source by causal coalescence priority.

    Inputs are ``K x C`` except ``left`` and ``source`` (``C``) and
    ``prefix_nodes`` (``K``).  The allocation is a greedy capped fill after a
    stable priority sort.  It therefore stays in ``[0, extra]`` and its sum is
    at most ``left`` for every atom.
    """

    if extra.ndim != 2 or capacity.shape != extra.shape:
        raise ValueError("Expected K x C extra/capacity tensors")
    branches, columns = extra.shape
    if (left.shape != (columns,) or source.shape != (columns,)
            or prefix_nodes.shape != (branches,)
            or mapped_tokens.shape != (branches, columns)):
        raise ValueError("Coalesced allocation shape mismatch")

    # Pairwise equality avoids packing node/token ids into a bounded integer
    # key and performs no result-to-host scalar copy on CUDA.
    same_prefix = prefix_nodes[:, None].eq(prefix_nodes[None, :])
    same_token = mapped_tokens[:, None, :].eq(mapped_tokens[None, :, :])
    multiplicity = (same_prefix[:, :, None] & same_token).sum(1)

    # A shared child always outranks a non-shared child.  Within the same
    # multiplicity, prefer the largest immediate feasible Target capacity.
    safe_source = torch.where(source > 0, source, torch.ones_like(source))
    continuation = (capacity / safe_source[None]).clamp(min=0, max=1)
    priority = multiplicity.to(extra.dtype) * 2 + continuation
    order = torch.argsort(priority, dim=0, descending=True, stable=True)
    ordered_extra = extra.gather(0, order)
    before = ordered_extra.cumsum(0) - ordered_extra
    ordered_allocation = torch.minimum(
        ordered_extra,
        (left[None] - before).clamp_min(0),
    )
    priority_allocation = torch.zeros_like(extra).scatter(
        0, order, ordered_allocation
    )

    # If an atom contains no genuine prefix/token coalescence, preserve the
    # established proportional transport exactly.  The new rule is activated
    # only where its claimed mechanism exists.
    total_extra = extra.sum(0)
    proportional_factor = (
        left / torch.where(total_extra > 0, total_extra,
                           torch.ones_like(total_extra))
    ).clamp_max(1)
    proportional_allocation = extra * proportional_factor[None]
    has_coalescence = multiplicity.max(0).values.gt(1)
    allocation = torch.where(
        has_coalescence[None], priority_allocation,
        proportional_allocation,
    )
    return allocation, multiplicity


def plan(
    alpha: torch.Tensor,
    support_p: torch.Tensor,
    proposal: DiffusionProposal,
    prefix_nodes: torch.Tensor,
) -> CoalescedBlockSkipPlan:
    """Build a causal coalescence-priority flow and joint block endpoint law."""

    branches, length, _ = proposal.tokens.shape
    source = proposal.source
    columns = source.shape[1]
    if (alpha.shape != (branches,)
            or support_p.shape != proposal.tokens.shape
            or prefix_nodes.shape != (branches, length + 1)
            or prefix_nodes.dtype != torch.long
            or prefix_nodes.device != source.device):
        raise ValueError("Coalesced block-skip plan shape mismatch")

    token_marginal = proposal.marginals()
    support_p = torch.where(
        token_marginal > 0, support_p, torch.zeros_like(support_p)
    )
    marginal = token_marginal.gather(-1, proposal.slots)
    safe = torch.where(marginal > 0, marginal, torch.ones_like(marginal))
    lifted = support_p.gather(-1, proposal.slots) * (source[None] / safe)
    outside_support = (1 - support_p.sum(-1)).clamp_min(0)
    mapped = proposal.tokens.gather(-1, proposal.slots)

    floors = [alpha]
    scores = [alpha]
    flows = []
    residuals = []
    prioritized = []
    shared_counts = []
    for depth in range(length):
        base = torch.minimum(
            floors[-1][:, None] * lifted[:, depth],
            alpha[:, None] * source[depth],
        )
        capacity = scores[-1][:, None] * lifted[:, depth]
        extra = (capacity - base).clamp_min(0)
        left = (source[depth] - base.sum(0)).clamp_min(0)
        extra_flow, multiplicity = _priority_extra(
            extra,
            left,
            capacity,
            source[depth],
            prefix_nodes[:, depth],
            mapped[:, depth],
        )
        flow = base + extra_flow
        residuals.append(
            scores[-1] * outside_support[:, depth]
            + (capacity - flow).clamp_min(0).sum(-1)
        )
        atom = proposal.draws[depth]
        floors.append(base[:, atom] / source[depth, atom])
        scores.append(flow[:, atom] / source[depth, atom])
        flows.append(flow)
        prioritized.append(extra_flow.sum())
        shared_counts.append(multiplicity.gt(1).to(source.dtype).mean())

    residuals.append(scores[-1])
    score = torch.stack(scores, 1)
    residual = torch.stack(residuals, 1)
    failure_mass = (1 - score.sum(0)).clamp_min(0)
    residual_total = residual.sum(0)
    total = failure_mass + residual_total
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    success = torch.where(
        total > 0, residual_total / denominator, torch.ones_like(total)
    )
    failure = torch.where(
        total > 0, failure_mass / denominator, torch.zeros_like(total)
    )
    later = torch.cat((
        failure[1:].flip(0).cumprod(0).flip(0), alpha.new_ones(1)
    ))
    return CoalescedBlockSkipPlan(
        floors=torch.stack(floors, 1),
        scores=score,
        flow=(torch.stack(flows, 1) if length else
              alpha.new_zeros((branches, 0, columns))),
        residual_mass=residual,
        endpoint=_normalize(success * later),
        prioritized_extra_mass=(torch.stack(prioritized) if prioritized else
                                alpha.new_zeros(0)),
        coalesced_label_atom_fraction=(
            torch.stack(shared_counts) if shared_counts else alpha.new_zeros(0)
        ),
    )


def expected_accepted_tokens(state: CoalescedBlockSkipPlan) -> torch.Tensor:
    depths = torch.arange(
        state.endpoint.numel(), dtype=state.endpoint.dtype,
        device=state.endpoint.device,
    )
    return (depths * state.endpoint).sum()


def verify_probabilities(
    node_probabilities: torch.Tensor,
    tree: Tree,
    proposal: DiffusionProposal,
    generator: torch.Generator | None = None,
    *,
    validate: bool = True,
):
    """Sample one exact Target block using prefix-coalesced priority flow."""

    branches, length, _ = proposal.slots.shape
    if (node_probabilities.ndim != 2
            or node_probabilities.shape[0] != len(tree.parents)
            or node_probabilities.device != proposal.source.device
            or not node_probabilities.is_floating_point()):
        raise ValueError("Target trie probabilities mismatch")
    if validate:
        proposal.validate(node_probabilities.shape[-1])
        if tree != sampled_tree(proposal.paths()):
            raise ValueError("Target tree must be the merged sampled trie")
        transport._check_rows(node_probabilities)

    row_nodes = torch.tensor(
        [[0] + nodes for nodes in tree.path_nodes],
        dtype=torch.long,
        device=node_probabilities.device,
    )
    if row_nodes.shape != (branches, length + 1):
        raise ValueError("Diffusion path index mismatch")
    prefix_nodes = row_nodes[:, :-1]
    support_p = node_probabilities[
        prefix_nodes[..., None], proposal.tokens
    ]
    alpha = proposal.source.new_full((branches,), 1 / branches)
    state = plan(alpha, support_p, proposal, row_nodes)

    by_branch = _normalize(state.residual_mass.T).T
    joint = state.endpoint[None] * by_branch
    selected = sampling.sample(_normalize(joint.reshape(-1)), generator)
    branch = selected // (length + 1)
    depth = selected % (length + 1)
    chosen_row = row_nodes[branch, depth]
    correction = state.scores[branch, depth] * node_probabilities[chosen_row]
    flows = torch.cat((
        state.flow,
        proposal.source.new_zeros((branches, 1, proposal.source.shape[1])),
    ), 1)
    mapped = proposal.tokens.gather(-1, proposal.slots)
    mapped = torch.cat((
        mapped,
        mapped.new_zeros((branches, 1, proposal.source.shape[1])),
    ), 1)
    correction.scatter_add_(0, mapped[branch, depth], -flows[branch, depth])
    token = sampling.sample(_normalize(correction.clamp_min(0)), generator)
    branch_value, depth_value, bonus = torch.stack((branch, depth, token)).tolist()
    nodes = tree.path_nodes[branch_value][:depth_value]
    return nodes, [tree.tokens[node - 1] for node in nodes], int(bonus), state


def verify_logits(
    node_logits: torch.Tensor,
    tree: Tree,
    proposal: DiffusionProposal,
    temperature: float,
    generator: torch.Generator | None = None,
    *,
    validate: bool = True,
):
    """Logit-space implementation used by the end-to-end engine."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Coalesced block skipping requires positive temperature")
    branches, length, _ = proposal.slots.shape
    if (node_logits.ndim != 2 or not node_logits.is_floating_point()
            or node_logits.shape[0] != len(tree.parents)
            or node_logits.device != proposal.source.device):
        raise ValueError("Target trie logits mismatch")
    if validate:
        proposal.validate(node_logits.shape[-1])
        if tree != sampled_tree(proposal.paths()):
            raise ValueError("Target tree must be the merged sampled trie")
        if bool(torch.isnan(node_logits).any() | torch.isposinf(node_logits).any()
                | ~torch.isfinite(node_logits).any(-1).all()):
            raise ValueError("Invalid target logits")

    row_nodes = torch.tensor(
        [[0] + nodes for nodes in tree.path_nodes],
        dtype=torch.long,
        device=node_logits.device,
    )
    if row_nodes.shape != (branches, length + 1):
        raise ValueError("Diffusion path index mismatch")
    scaled = sampling.scaled_logits(node_logits, temperature)
    normalizers = torch.logsumexp(scaled, -1)
    prefix_nodes = row_nodes[:, :-1]
    support_p = (
        scaled[prefix_nodes[..., None], proposal.tokens]
        - normalizers[prefix_nodes, None]
    ).exp()
    del scaled
    alpha = proposal.source.new_full((branches,), 1 / branches)
    state = plan(alpha, support_p, proposal, row_nodes)

    by_branch = _normalize(state.residual_mass.T).T
    joint = state.endpoint[None] * by_branch
    selected = sampling.sample(_normalize(joint.reshape(-1)), generator)
    branch = selected // (length + 1)
    depth = selected % (length + 1)
    chosen_row = row_nodes[branch, depth]
    correction = sampling.probabilities(
        node_logits[chosen_row].double(), temperature
    )
    correction = state.scores[branch, depth] * correction
    flows = torch.cat((
        state.flow,
        proposal.source.new_zeros((
            branches, 1, proposal.source.shape[1]
        )),
    ), 1)
    mapped = proposal.tokens.gather(-1, proposal.slots)
    mapped = torch.cat((
        mapped,
        mapped.new_zeros((branches, 1, proposal.source.shape[1])),
    ), 1)
    correction.scatter_add_(0, mapped[branch, depth], -flows[branch, depth])
    token = sampling.sample(_normalize(correction.clamp_min(0)), generator)
    branch_value, depth_value, bonus = torch.stack((branch, depth, token)).tolist()
    nodes = tree.path_nodes[branch_value][:depth_value]
    return nodes, [tree.tokens[node - 1] for node in nodes], int(bonus), state
