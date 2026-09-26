"""One-step masked block diffusion, coupled samples, and trie block verification.

The proposal is the actual temperature-scaled, support-truncated DFlash
transition. Every candidate position is stochastic; shared prefixes are merged
only for target evaluation. Latent sample labels survive merging. The transport
recurrence and backward sampling build on atom_tree_bv / Layer Verification;
neither that recurrence nor diffusion-specific novelty is claimed as new here.
See docs/DIFFUSION_TREE_BV.md for the complete finite-space proof and scope.
"""
from dataclasses import dataclass
import heapq
import math

import torch

from . import atom_tree_bv as transport, sampling
from .root_marginalized_bv import _normalize
from .tree import Tree, probability_tree, sampled_tree


@dataclass
class DiffusionBlockLaw:
    tokens: torch.Tensor              # L x R denoising support
    weights: torch.Tensor             # L x R actual truncated transition
    retained_mass: torch.Tensor       # L, mass under the full softmax transition
    noise_ids: torch.Tensor           # clean anchor followed by mask tokens
    draft_temperature: float
    mask_token_id: int

    @classmethod
    def from_logits(cls, logits, noise_ids, draft_temperature, mask_token_id,
                    *, support_size=8):
        if (logits.ndim != 2 or not logits.is_floating_point() or logits.shape[0] < 1
                or type(support_size) is not int or support_size < 1
                or not math.isfinite(draft_temperature) or draft_temperature <= 0):
            raise ValueError("A one-step block requires L x V logits and positive draft temperature")
        if (noise_ids.ndim != 1 or noise_ids.dtype != torch.long
                or noise_ids.device != logits.device or noise_ids.numel() <= logits.shape[0]
                or not bool(noise_ids[1:].eq(mask_token_id).all())):
            raise ValueError("DFlash transition requires a clean anchor followed by all masked slots")
        if bool(torch.isnan(logits).any() | torch.isposinf(logits).any()
                | ~torch.isfinite(logits).any(-1).all()):
            raise ValueError("Invalid denoising logits")
        scaled = sampling.scaled_logits(logits, draft_temperature)
        values, tokens = scaled.topk(min(support_size, logits.shape[-1]), dim=-1)
        support_z = torch.logsumexp(values, -1)
        retained = (support_z - torch.logsumexp(scaled, -1)).exp().clamp_max(1)
        weights = (values - support_z[:, None]).exp()
        return cls(tokens, _normalize(weights), retained, noise_ids.clone(),
                   float(draft_temperature), int(mask_token_id))

    def validate(self, vocab):
        if (self.tokens.ndim != 2 or self.weights.shape != self.tokens.shape
                or self.tokens.shape[0] < 1 or self.tokens.shape[1] < 1
                or self.retained_mass.shape != self.tokens.shape[:1]
                or self.tokens.dtype != torch.long or self.weights.dtype != torch.float64
                or self.retained_mass.dtype != torch.float64
                or any(t.device != self.weights.device for t in
                       (self.tokens, self.retained_mass, self.noise_ids))):
            raise ValueError("Malformed diffusion block law")
        transport._check_rows(self.weights)
        if (not math.isfinite(self.draft_temperature) or self.draft_temperature <= 0
                or self.noise_ids.ndim != 1 or self.noise_ids.dtype != torch.long
                or self.noise_ids.numel() <= self.tokens.shape[0]
                or not bool(self.noise_ids[1:].eq(self.mask_token_id).all())
                or bool((self.tokens < 0).any() | (self.tokens >= vocab).any())
                or bool((self.tokens.sort(-1).values.diff(dim=-1) == 0).any())
                or not bool(torch.isfinite(self.retained_mass).all()
                            & (self.retained_mass >= 0).all() & (self.retained_mass <= 1).all())):
            raise ValueError("Invalid masked diffusion transition")

    def retained_block_mass(self):
        """Q_D(S_1 x ... x S_L); TV(Q_D, truncated Q_D) = 1 - this value."""
        return self.retained_mass.prod()


@dataclass
class DiffusionProposal:
    law: DiffusionBlockLaw
    slots: torch.Tensor               # K x L x C atom -> denoising support slot
    source: torch.Tensor              # L x C; independent across positions
    draws: torch.Tensor               # L sampled atom indices

    @property
    def tokens(self):
        return self.law.tokens[None].expand(self.slots.shape[0], -1, -1)

    def marginals(self):
        return self.source.new_zeros(self.tokens.shape).scatter_add_(
            -1, self.slots, self.source[None].expand_as(self.slots))

    def paths(self):
        k, length, _ = self.slots.shape
        selected = self.slots.gather(-1, self.draws[None, :, None].expand(k, length, 1))
        return self.tokens.gather(-1, selected).squeeze(-1)

    def latent_log_probability(self):
        # This is the probability of the retained witness, NOT of its trie.
        return self.source.gather(1, self.draws[:, None]).log().sum()

    def validate(self, vocab):
        self.law.validate(vocab)
        if self.slots.ndim != 3 or self.source.ndim != 2:
            raise ValueError("Malformed diffusion atom partition")
        k, length, columns = self.slots.shape
        if (min(k, columns) < 1 or length != self.law.tokens.shape[0]
                or self.source.shape != (length, columns) or self.draws.shape != (length,)
                or self.slots.dtype != torch.long or self.draws.dtype != torch.long
                or self.source.dtype != torch.float64
                or any(t.device != self.source.device for t in
                       (self.slots, self.draws, self.law.weights))):
            raise ValueError("Diffusion atom shape/device mismatch")
        transport._check_rows(self.source)
        if (bool((self.slots < 0).any() | (self.slots >= self.tokens.shape[-1]).any())
                or bool((self.draws < 0).any() | (self.draws >= columns).any())
                or not bool((self.source.gather(1, self.draws[:, None]) > 0).all())):
            raise ValueError("Invalid diffusion atom sample")
        if not torch.allclose(self.marginals(), self.law.weights[None].expand_as(self.tokens),
                              rtol=0, atol=1e-12):
            raise ValueError("Atom mapping changed the denoising transition marginals")


def propose(law, branches, generator=None, *, coupling="depth_permuted"):
    if type(branches) is not int or branches < 1 or coupling not in {"aligned", "depth_permuted"}:
        raise ValueError("Invalid diffusion coupling")
    length = law.tokens.shape[0]
    shifts = (law.weights.new_zeros((branches, length)) if coupling == "aligned"
              else transport.depth_permuted_shifts(branches, length, device=law.weights.device))
    atoms = transport.couple(torch.arange(branches, device=law.weights.device),
                             law.tokens[None].expand(branches, -1, -1),
                             law.weights[None].expand(branches, -1, -1), generator, shifts=shifts)
    return DiffusionProposal(law, atoms.slots, atoms.source, atoms.draws)


def verify_logits(node_logits, tree, proposal, temperature, generator=None,
                  *, pool=True, validate=True):
    """Joint endpoint and residual draw on a trie; returns accepted nodes/tokens/bonus.

    No full K x L x V path tensor: normalizers are computed once per trie node,
    then only R candidate logits are gathered per labelled path and position.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Diffusion tree BV is registered only for positive target temperature")
    k, length, _ = proposal.slots.shape
    if (node_logits.ndim != 2 or not node_logits.is_floating_point()
            or node_logits.shape[-1] < 1 or node_logits.shape[0] != len(tree.parents)
            or node_logits.device != proposal.source.device):
        raise ValueError("Target trie logits mismatch")
    if validate:
        proposal.validate(node_logits.shape[-1])
        merged = sampled_tree(proposal.paths())
        unmerged = sampled_tree(proposal.paths(), False)
        if tree != merged and tree != unmerged:
            raise ValueError("Target tree must represent exactly the retained diffusion samples")
        if bool(torch.isnan(node_logits).any() | torch.isposinf(node_logits).any()
                | ~torch.isfinite(node_logits).any(-1).all()):
            raise ValueError("Invalid target logits")
    # Include the common anchor in every labelled path. Duplicate node ids are
    # intentional and carry separate transport weights, but share target rows.
    row_nodes = torch.tensor([[0] + nodes for nodes in tree.path_nodes],
                             dtype=torch.long, device=node_logits.device)
    if row_nodes.shape != (k, length + 1):
        raise ValueError("Diffusion path index mismatch")
    scaled = sampling.scaled_logits(node_logits, temperature)
    normalizers = torch.logsumexp(scaled, -1)
    prefix_nodes = row_nodes[:, :-1]
    support = (scaled[prefix_nodes[..., None], proposal.tokens]
               - normalizers[prefix_nodes, None]).exp()
    del scaled
    alpha = proposal.source.new_full((k,), 1 / k)
    state = transport.plan(alpha, support, proposal, pool=pool)
    by_branch = _normalize(state.residual_mass.T).T
    weights = state.endpoint[None] * by_branch
    selected = sampling.sample(_normalize(weights.reshape(-1)), generator)
    branch, depth = selected // (length + 1), selected % (length + 1)
    chosen_row = row_nodes[branch, depth]
    selected_logits = node_logits[chosen_row].double()
    correction = sampling.probabilities(selected_logits, temperature)
    correction = state.scores[branch, depth] * correction
    flows = torch.cat((state.flow, proposal.source.new_zeros((k, 1, proposal.source.shape[1]))), 1)
    mapped = proposal.tokens.gather(-1, proposal.slots)
    mapped = torch.cat((mapped, mapped.new_zeros((k, 1, proposal.source.shape[1]))), 1)
    correction.scatter_add_(0, mapped[branch, depth], -flows[branch, depth])
    token = sampling.sample(_normalize(correction.clamp_min(0)), generator)
    branch, depth, bonus = torch.stack((branch, depth, token)).tolist()
    nodes = tree.path_nodes[branch][:depth]
    return nodes, [tree.tokens[node - 1] for node in nodes], bonus


def verify_probabilities(node_probabilities, tree, proposal, generator=None,
                         *, pool=True, validate=True):
    """Diffusion BV using already-normalized Target rows.

    The scaffold verifier needs the same full Target rows again when an initial
    correction lands inside its deterministic continuation tree. Accepting a
    shared probability tensor avoids repeating the expensive FP64 full-vocabulary
    normalization while preserving the same transport recurrence.
    """
    k, length, _ = proposal.slots.shape
    if (node_probabilities.ndim != 2 or not node_probabilities.is_floating_point()
            or node_probabilities.shape[-1] < 1
            or node_probabilities.shape[0] != len(tree.parents)
            or node_probabilities.device != proposal.source.device):
        raise ValueError("Target trie probabilities mismatch")
    if validate:
        proposal.validate(node_probabilities.shape[-1])
        merged = sampled_tree(proposal.paths())
        unmerged = sampled_tree(proposal.paths(), False)
        if tree != merged and tree != unmerged:
            raise ValueError("Target tree must represent exactly the retained diffusion samples")
        tolerance = max(1e-10, 8 * torch.finfo(node_probabilities.dtype).eps)
        if not bool(torch.isfinite(node_probabilities).all()
                    & (node_probabilities >= 0).all()
                    & torch.isclose(
                        node_probabilities.sum(-1),
                        node_probabilities.new_ones(node_probabilities.shape[0]),
                        rtol=tolerance, atol=1e-12,
                    ).all()):
            raise ValueError("Invalid target probabilities")
    row_nodes = torch.tensor([[0] + nodes for nodes in tree.path_nodes],
                             dtype=torch.long, device=node_probabilities.device)
    if row_nodes.shape != (k, length + 1):
        raise ValueError("Diffusion path index mismatch")
    prefix_nodes = row_nodes[:, :-1]
    support = node_probabilities[prefix_nodes[..., None], proposal.tokens]
    alpha = proposal.source.new_full((k,), 1 / k)
    state = transport.plan(alpha, support, proposal, pool=pool)
    by_branch = _normalize(state.residual_mass.T).T
    weights = state.endpoint[None] * by_branch
    selected = sampling.sample(_normalize(weights.reshape(-1)), generator)
    branch, depth = selected // (length + 1), selected % (length + 1)
    chosen_row = row_nodes[branch, depth]
    correction = state.scores[branch, depth] * node_probabilities[chosen_row]
    flows = torch.cat((state.flow, proposal.source.new_zeros((
        k, 1, proposal.source.shape[1]
    ))), 1)
    mapped = proposal.tokens.gather(-1, proposal.slots)
    mapped = torch.cat((mapped, mapped.new_zeros((
        k, 1, proposal.source.shape[1]
    ))), 1)
    correction.scatter_add_(0, mapped[branch, depth], -flows[branch, depth])
    token = sampling.sample(_normalize(correction.clamp_min(0)), generator)
    branch, depth, bonus = torch.stack((branch, depth, token)).tolist()
    nodes = tree.path_nodes[branch][:depth]
    return nodes, [tree.tokens[node - 1] for node in nodes], bonus


def verify_single_path_logits(node_logits, tree, proposal, temperature,
                              generator=None, *, validate=True):
    """Specialize diffusion BV to its exact K=1 sparse-block form.

    With one labelled branch the latent coupling has the product of the
    truncated denoising marginals as its proposal law.  The general transport
    recurrence therefore reduces exactly to ordinary block verification.  This
    implementation keeps only L x R Target support probabilities plus the one
    full-vocabulary correction row that reaches the output.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Single-path diffusion BV requires positive temperature")
    k, length, _ = proposal.slots.shape
    if (k != 1 or node_logits.ndim != 2 or not node_logits.is_floating_point()
            or node_logits.shape[0] != len(tree.parents)
            or node_logits.device != proposal.source.device
            or len(tree.path_nodes) != 1
            or len(tree.path_nodes[0]) != length):
        raise ValueError("Single-path diffusion target/proposal shape mismatch")
    if validate:
        proposal.validate(node_logits.shape[-1])
        if tree != sampled_tree(proposal.paths()):
            raise ValueError("Single-path tree must represent the diffusion sample")
        if bool(torch.isnan(node_logits).any() | torch.isposinf(node_logits).any()
                | ~torch.isfinite(node_logits).any(-1).all()):
            raise ValueError("Invalid target logits")

    path = proposal.paths()[0]
    row_nodes = torch.tensor(
        [0] + tree.path_nodes[0], dtype=torch.long, device=node_logits.device,
    )
    prefix_logits = node_logits.index_select(0, row_nodes[:-1])
    scaled = sampling.scaled_logits(prefix_logits, temperature)
    normalizer = torch.logsumexp(scaled, -1, keepdim=True)
    target_support = (
        scaled.gather(1, proposal.law.tokens) - normalizer
    ).exp()
    del scaled

    proposal_support = proposal.law.weights
    matches = proposal.law.tokens.eq(path[:, None])
    selected_target = torch.where(
        matches, target_support, torch.zeros_like(target_support)
    ).sum(-1)
    selected_proposal = torch.where(
        matches, proposal_support, torch.zeros_like(proposal_support)
    ).sum(-1)
    if validate and not bool((selected_proposal > 0).all()):
        raise FloatingPointError("Sampled diffusion token has zero proposal mass")

    weight = node_logits.new_ones((), dtype=torch.float64)
    weights_by_depth = [weight]
    for depth in range(length):
        weight = torch.minimum(
            torch.ones_like(weight),
            weight * selected_target[depth] / selected_proposal[depth],
        )
        weights_by_depth.append(weight)
    prefix_weights = torch.stack(weights_by_depth)

    outside_support = (1 - target_support.sum(-1)).clamp_min(0)
    support_residual = (
        prefix_weights[:-1, None] * target_support - proposal_support
    ).clamp_min(0)
    residual_totals = torch.cat((
        prefix_weights[:-1] * outside_support + support_residual.sum(-1),
        prefix_weights[-1:],
    ))
    rejection = (1 - prefix_weights).clamp_min(0)
    totals = residual_totals + rejection
    if validate and not bool(torch.isfinite(totals).all()):
        raise FloatingPointError("Invalid single-path BV endpoint masses")
    safe_totals = torch.where(totals > 0, totals, torch.ones_like(totals))
    accept_probability = torch.where(
        totals > 0, residual_totals / safe_totals, torch.ones_like(totals)
    )
    binary = torch.stack((accept_probability, 1 - accept_probability), -1)
    accepted_rows = sampling.sample(binary, generator).eq(0)
    depth_ids = torch.arange(length + 1, device=node_logits.device)
    selected_depth = torch.where(
        accepted_rows, depth_ids, -torch.ones_like(depth_ids)
    ).max()

    correction_logits = node_logits.index_select(
        0, row_nodes.index_select(0, selected_depth.reshape(1))
    )[0]
    target_row = sampling.probabilities(
        correction_logits, temperature, torch.float64
    )
    residual_row = prefix_weights[selected_depth] * target_row
    proposal_row = selected_depth.clamp_max(length - 1)
    row_tokens = proposal.law.tokens.index_select(
        0, proposal_row.reshape(1)
    )
    row_probabilities = proposal_support.index_select(
        0, proposal_row.reshape(1)
    ) * selected_depth.lt(length)
    residual_row.scatter_add_(0, row_tokens[0], -row_probabilities[0])
    residual_row.clamp_min_(0)
    residual_total = residual_totals[selected_depth]
    safe_residual_total = torch.where(
        residual_total > 0, residual_total, torch.ones_like(residual_total)
    )
    correction = torch.where(
        residual_total > 0, residual_row / safe_residual_total, target_row
    )
    bonus = sampling.sample(correction, generator)
    depth, token = torch.stack((selected_depth, bonus)).tolist()
    if depth < 0:
        raise RuntimeError("Single-path BV failed to produce a nonempty output")
    nodes = tree.path_nodes[0][:depth]
    return nodes, [tree.tokens[node - 1] for node in nodes], int(token)


def snapshot(proposal):
    """Flat tensor/metadata format supports CPU storage and existing GPU replay."""
    result = {"diffusion_" + key: value.detach().cpu().clone()
              for key, value in vars(proposal).items() if isinstance(value, torch.Tensor)}
    result.update({"law_" + key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                   for key, value in vars(proposal.law).items()})
    return result


def restore(state):
    law = DiffusionBlockLaw(**{key: state["law_" + key] for key in DiffusionBlockLaw.__dataclass_fields__})
    return DiffusionProposal(law, **{key: state["diffusion_" + key] for key in ("slots", "source", "draws")})


def truncate_proposal(proposal, length):
    """Keep the leading positions of a sampled diffusion block and its law."""
    full_length = proposal.law.tokens.shape[0]
    if type(length) is not int or not 1 <= length < full_length:
        raise ValueError("Truncated diffusion length must be in 1..L-1")
    law = DiffusionBlockLaw(
        proposal.law.tokens[:length],
        proposal.law.weights[:length],
        proposal.law.retained_mass[:length],
        proposal.law.noise_ids[:length + 1],
        proposal.law.draft_temperature,
        proposal.law.mask_token_id,
    )
    return DiffusionProposal(
        law,
        proposal.slots[:, :length],
        proposal.source[:length],
        proposal.draws[:length],
    )


def core_spur_tree(proposal, position_probabilities, budget):
    """Union one labelled BV spur with the best DDTree prefix core.

    The sampled spur is inserted first so ``path_nodes`` remains a compact
    labelled prefix for the existing diffusion-BV kernel.  Remaining capacity
    is filled in DDTree best-first order.  The tree is fixed before seeing any
    Target row; its only randomness is the retained proposal witness.
    """
    paths = proposal.paths()
    if (paths.ndim != 2 or paths.shape[0] != 1
            or position_probabilities.ndim != 2
            or position_probabilities.shape[0] < paths.shape[1]
            or position_probabilities.shape[1] < 1
            or type(budget) is not int or budget <= paths.shape[1]):
        raise ValueError("Invalid core-spur proposal, probabilities, or budget")
    tree = sampled_tree(paths)
    children = {
        (tree.parents[node], token): node
        for node, token in enumerate(tree.tokens, 1)
    }

    def insert(parent, token):
        child = children.get((parent, token))
        if child is None:
            child = len(tree.parents)
            children[parent, token] = child
            tree.tokens.append(token)
            tree.parents.append(parent)
            tree.depths.append(tree.depths[parent] + 1)
        return child

    # Up to ``spur_length`` source nodes can overlap the random path, so ask
    # for that many extra best-first prefixes and stop at the strict B cap.
    core = probability_tree(
        position_probabilities, budget + paths.shape[1],
    )
    source_to_union = {0: 0}
    for source in range(1, len(core.parents)):
        parent = source_to_union[core.parents[source]]
        source_to_union[source] = insert(parent, core.tokens[source - 1])
        if len(tree.tokens) == budget:
            break
    if len(tree.tokens) > budget:
        raise RuntimeError("Core-spur tree exceeded its node budget")
    return tree


def scaffold_tree(proposal, greedy, budget, *, fill=True):
    """Retain the original random paths AND a fixed greedy chain within B nodes.

    Optional filler uses original retained denoising masses, not renormalized
    support probabilities. No path is pruned or resampled after seeing its draw.
    Only the random paths are transport labels; deterministic nodes are queried
    for an ordinary Target continuation after the first correction.
    """
    paths = proposal.paths()
    k, length = paths.shape
    if (greedy.shape != (length,) or greedy.dtype != torch.long
            or type(budget) is not int or budget < (k + 1) * length):
        raise ValueError("Scaffold needs a greedy chain and (K+1)*L worst-case nodes")
    tree = sampled_tree(paths)
    children = {(tree.parents[n], token): n for n, token in enumerate(tree.tokens, 1)}

    def insert(parent, token):
        child = children.get((parent, token))
        if child is None:
            child = len(tree.parents)
            children[parent, token] = child
            tree.tokens.append(token)
            tree.parents.append(parent)
            tree.depths.append(tree.depths[parent] + 1)
        return child

    parent = 0
    for token in greedy.tolist():
        parent = insert(parent, token)
    if not fill or len(tree.tokens) == budget:
        return tree
    # Only small L x R host transfers, never a full vocabulary sort. Top-R support is
    # a deliberate ceiling: this is DDTree-style filling, not DDTree's B optimum.
    weights = proposal.law.weights * proposal.law.retained_mass[:, None]
    by_token = proposal.law.tokens.argsort(dim=-1, stable=True)
    by_weight = weights.gather(1, by_token).argsort(dim=-1, descending=True, stable=True)
    order = by_token.gather(1, by_weight)
    logs = weights.gather(1, order).log().tolist()
    tokens = proposal.law.tokens.gather(1, order).tolist()
    # Canonical token-prefix tie breaks MUST NOT use random tree node ids:
    # fixed-prefix coverage in the theorem has to hold for every latent draw.
    heap = [(-logs[0][0], (tokens[0][0],), 0, 0, 0, 0.)]
    while heap and len(tree.tokens) < budget:
        neg_score, prefix, parent, depth, rank, parent_log = heapq.heappop(heap)
        if not math.isfinite(neg_score):
            continue
        child = insert(parent, tokens[depth][rank])
        if rank + 1 < len(tokens[depth]):
            heapq.heappush(heap, (-(parent_log + logs[depth][rank + 1]),
                                 prefix[:-1] + (tokens[depth][rank + 1],),
                                 parent, depth, rank + 1, parent_log))
        if depth + 1 < length:
            heapq.heappush(heap, (neg_score - logs[depth + 1][0],
                                 prefix + (tokens[depth + 1][0],), child, depth + 1, 0, -neg_score))
    return tree


def verify_scaffold_logits(node_logits, tree, proposal, temperature, generator=None,
                           *, recycle=True, continuation="terminal", validate=True,
                           node_probabilities=None):
    """Original diffusion BV, then independent ancestral Target continuation.

    Conditional on a correction landing in the verified tree, sample the
    remaining subtree through an existing terminal-mass sampler. No new model
    forward, per-token synchronization, or reuse of proposal-dependent draws.
    Full-vocabulary normalization of extra rows happens only when needed.
    """
    if continuation not in {"terminal", "ancestral"}:
        raise ValueError("Unknown scaffold continuation backend")
    count = max(n for path in tree.path_nodes for n in path) + 1
    base = Tree(tree.tokens[:count - 1], tree.parents[:count], tree.depths[:count], tree.path_nodes)
    if (node_logits.ndim != 2 or node_logits.shape[0] != len(tree.parents)
            or len(tree.tokens) + 1 != len(tree.parents) or len(tree.depths) != len(tree.parents)):
        raise ValueError("Scaffold target/topology shape mismatch")
    if validate:
        if (tree.parents[0] != -1 or tree.depths[0] != 0
                or any(parent < 0 or parent >= n or tree.depths[n] != tree.depths[parent] + 1
                       for n, parent in enumerate(tree.parents[1:], 1))
                or any(len(path) != proposal.source.shape[0]
                       for path in tree.path_nodes)
                or len(set(zip(tree.parents[1:], tree.tokens))) != len(tree.tokens)
                or any(token < 0 or token >= node_logits.shape[-1] for token in tree.tokens)
                or bool(torch.isnan(node_logits).any() | torch.isposinf(node_logits).any()
                        | ~torch.isfinite(node_logits).any(-1).all())):
            raise ValueError("Invalid scaffold target rows or topology")
    if node_probabilities is None:
        verifier = (
            verify_single_path_logits
            if proposal.slots.shape[0] == 1 else verify_logits
        )
        nodes, tokens, bonus = verifier(
            node_logits[:count], base, proposal, temperature, generator,
            validate=validate,
        )
    else:
        if (node_probabilities.ndim != 2
                or node_probabilities.shape != node_logits.shape
                or node_probabilities.device != node_logits.device):
            raise ValueError("Shared scaffold probabilities mismatch")
        nodes, tokens, bonus = verify_probabilities(
            node_probabilities[:count], base, proposal, generator,
            validate=validate,
        )
    if not recycle:
        return nodes, tokens, bonus
    if node_probabilities is not None:
        return continue_scaffold_probabilities(
            node_probabilities, tree, nodes, tokens, bonus, generator,
            continuation=continuation,
        )
    return continue_scaffold_logits(node_logits, tree, nodes, tokens, bonus,
                                    temperature, generator, continuation=continuation)


def verify_scaffold_hidden(final_hidden, lm_head, tree, proposal, temperature,
                           generator=None, *, recycle=True,
                           continuation="terminal", validate=True):
    """Verify a scaffold while projecting only rows the block can consume.

    The Target transformer still evaluates every tree node with the same mask.
    This execution path defers the vocabulary head: first project the labelled
    diffusion trie, then (only when entered) the selected continuation subtree.
    It is the same composition used by :func:`verify_scaffold_logits` and does
    not prune or alter the proposal tree after observing Target values.
    """
    count = max(n for path in tree.path_nodes for n in path) + 1
    base = Tree(
        tree.tokens[:count - 1], tree.parents[:count],
        tree.depths[:count], tree.path_nodes,
    )
    if (final_hidden.ndim != 2 or final_hidden.shape[0] != len(tree.parents)
            or final_hidden.device != proposal.source.device):
        raise ValueError("Scaffold target hidden-state/topology shape mismatch")
    if validate:
        # The trusted engine path below avoids unused LM-head rows.  Public
        # validation deliberately projects all rows once and delegates to the
        # established full-logit contract so malformed unused nodes cannot hide.
        return verify_scaffold_logits(
            lm_head(final_hidden), tree, proposal, temperature, generator,
            recycle=recycle, continuation=continuation, validate=True,
        )

    base_logits = lm_head(final_hidden[:count])
    verifier = (
        verify_single_path_logits
        if proposal.slots.shape[0] == 1 else verify_logits
    )
    nodes, tokens, bonus = verifier(
        base_logits, base, proposal, temperature, generator, validate=validate,
    )
    if not recycle:
        return nodes, tokens, bonus

    children = {
        (parent, token): node
        for node, (parent, token) in enumerate(
            zip(tree.parents[1:], tree.tokens), 1
        )
    }
    child = children.get((nodes[-1] if nodes else 0, bonus))
    if child is None:
        return nodes, tokens, bonus
    indices, local = [child], {child: 0}
    parents, proposed = [-1], []
    for node in range(child + 1, len(tree.parents)):
        if tree.parents[node] in local:
            local[node] = len(indices)
            indices.append(node)
            parents.append(local[tree.parents[node]])
            proposed.append(tree.tokens[node - 1])
    selected_hidden = final_hidden.index_select(
        0, torch.tensor(indices, dtype=torch.long, device=final_hidden.device)
    )
    probabilities = sampling.probabilities(
        lm_head(selected_hidden), temperature, torch.float64
    )
    return _continue_scaffold_probabilities(
        probabilities, indices, parents, proposed, nodes, tokens, bonus,
        generator, continuation=continuation,
    )


def continue_scaffold_logits(node_logits, tree, nodes, tokens, bonus, temperature,
                             generator=None, *, continuation="terminal"):
    """Fresh target-only continuation of an already validated block output.

    Internal composition helper: callers validate the tree, target rows, and
    initial block kernel. It does not make a biased initial output lossless.
    """
    if continuation not in {"terminal", "ancestral"}:
        raise ValueError("Unknown scaffold continuation backend")
    children = {(parent, token): n for n, (parent, token)
                in enumerate(zip(tree.parents[1:], tree.tokens), 1)}
    child = children.get((nodes[-1] if nodes else 0, bonus))
    if child is None:
        return nodes, tokens, bonus
    # Reuse established Target-only verifiers. This continuation must
    # be independent of latent/proposal-dependent endpoint decisions above.
    indices, local = [child], {child: 0}
    parents, proposed = [-1], []
    for n in range(child + 1, len(tree.parents)):
        if tree.parents[n] in local:
            local[n] = len(indices)
            indices.append(n)
            parents.append(local[tree.parents[n]])
            proposed.append(tree.tokens[n - 1])
    selected = node_logits[torch.tensor(indices, device=node_logits.device)].double()
    p = sampling.probabilities(selected, temperature)
    return _continue_scaffold_probabilities(
        p, indices, parents, proposed, nodes, tokens, bonus, generator,
        continuation=continuation,
    )


def continue_scaffold_probabilities(node_probabilities, tree, nodes, tokens, bonus,
                                    generator=None, *, continuation="terminal"):
    """Continue an initial diffusion correction using shared Target rows."""
    if continuation not in {"terminal", "ancestral"}:
        raise ValueError("Unknown scaffold continuation backend")
    children = {(parent, token): n for n, (parent, token)
                in enumerate(zip(tree.parents[1:], tree.tokens), 1)}
    child = children.get((nodes[-1] if nodes else 0, bonus))
    if child is None:
        return nodes, tokens, bonus
    indices, local = [child], {child: 0}
    parents, proposed = [-1], []
    for n in range(child + 1, len(tree.parents)):
        if tree.parents[n] in local:
            local[n] = len(indices)
            indices.append(n)
            parents.append(local[tree.parents[n]])
            proposed.append(tree.tokens[n - 1])
    p = node_probabilities.index_select(
        0, torch.tensor(indices, device=node_probabilities.device)
    )
    return _continue_scaffold_probabilities(
        p, indices, parents, proposed, nodes, tokens, bonus, generator,
        continuation=continuation,
    )


def _continue_scaffold_probabilities(p, indices, parents, proposed, nodes, tokens,
                                     bonus, generator, *, continuation):
    verifier = (sampling.tree_block_verify_terminal_mass if continuation == "terminal"
                else sampling.tree_verify_ancestral_batched)
    more_nodes, more_tokens, correction = verifier(
        parents, proposed, p, generator, validate=False)
    return (nodes + [indices[0]] + [indices[n] for n in more_nodes],
            tokens + [bonus] + more_tokens, correction)
