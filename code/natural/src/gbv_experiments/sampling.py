"""GBV/BV kernels; Thomas & Pal (2026) and Sun et al. (2024).

The CDF difference is evaluated after scaling and cancellation of its common
prefix mass, using a nonnegative polynomial instead of subtracting powers.
"""
from __future__ import annotations

import math

import torch


def scaled_logits(logits: torch.Tensor, temperature: float, dtype=torch.float64):
    """Stable softmax-equivalent logits for any finite positive temperature.

    Small temperatures require centering BEFORE division to avoid +inf.
    Large temperatures require division first: subtracting opposite large
    finite logits can overflow even when their scaled difference is modest.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected finite positive temperature")
    raw = logits.to(dtype)
    if temperature < 1:
        return (raw - raw.amax(-1, keepdim=True)) / temperature
    raw = raw / temperature
    return raw - raw.amax(-1, keepdim=True)


def probabilities(logits: torch.Tensor, temperature: float, dtype=torch.float64):
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(dtype)
    if math.isfinite(temperature) and temperature >= 1:
        # Division cannot amplify finite logits here. Softmax already centers
        # internally; avoid a redundant full-vocabulary reduction/copy.
        return torch.softmax(logits.to(dtype) / temperature, dim=-1)
    return torch.softmax(scaled_logits(logits, temperature, dtype), dim=-1)


def sample(probs: torch.Tensor, generator=None):
    shape = probs.shape[:-1]
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1,
                             generator=generator).reshape(shape)


def power_sum(upper, lower, k: int):
    # (upper**k - lower**k)/(upper-lower), including upper==lower.
    result = torch.ones_like(upper)
    lower_power = torch.ones_like(lower)
    for _ in range(1, k):
        lower_power = lower_power * lower
        result = result * upper + lower_power
    return result


def select_greedy_path(paths, target_token_probabilities, proposal_token_probabilities):
    k, length = paths.shape
    if target_token_probabilities.shape != (k, length) or proposal_token_probabilities.shape != (k, length):
        raise ValueError("GBV selected-token probability shape mismatch")
    valid = (torch.isfinite(target_token_probabilities).all()
             & torch.isfinite(proposal_token_probabilities).all()
             & (target_token_probabilities >= 0).all()
             & (proposal_token_probabilities > 0).all())
    if not bool(valid):
        raise FloatingPointError("Invalid GBV selected-token probabilities")
    scores = (target_token_probabilities.log() - proposal_token_probabilities.log()).tolist()
    cpu_paths = paths.tolist()
    return max(range(k), key=lambda j: tuple(zip(scores[j], cpu_paths[j])))


def greedy_max_distribution(paths, target_token_probabilities,
                            proposal_token_probabilities, leaf_probabilities, k: int,
                            validate: bool = True):
    """Distribution of GBV's maximum of K IID finite-tree proposals.

    This analytically marginalizes the K candidate draws after all support
    leaves have been verified.  It is distributionally identical to drawing K
    leaves with replacement and applying :func:`select_greedy_path`, while
    avoiding duplicate candidate verification and permitting a large virtual K.
    """
    leaves, length = paths.shape
    if (target_token_probabilities.shape != (leaves, length)
            or proposal_token_probabilities.shape != (leaves, length)
            or leaf_probabilities.shape != (leaves,) or k < 1):
        raise ValueError("Full-tree GBV tensor shape mismatch")
    if validate:
        valid = (torch.isfinite(target_token_probabilities).all()
                 & torch.isfinite(proposal_token_probabilities).all()
                 & torch.isfinite(leaf_probabilities).all()
                 & (target_token_probabilities >= 0).all()
                 & (proposal_token_probabilities > 0).all()
                 & (leaf_probabilities > 0).all()
                 & (leaf_probabilities.sum() > 0))
        if not bool(valid):
            raise FloatingPointError("Invalid full-tree GBV probabilities")

    normalized = leaf_probabilities / leaf_probabilities.sum()
    scores = target_token_probabilities.log() - proposal_token_probabilities.log()
    # Exact stable lexicographic ordering of
    # (score[0], token[0], score[1], token[1], ...), entirely on device.  The
    # old Python ``tolist`` implementation forced a device synchronization in
    # every recycled segment.
    order_tensor = torch.arange(leaves, dtype=torch.long, device=paths.device)
    for depth in range(length - 1, -1, -1):
        ordered_tokens = paths[:, depth].index_select(0, order_tensor)
        order_tensor = order_tensor.index_select(
            0, torch.argsort(ordered_tokens, stable=True)
        )
        ordered_scores = scores[:, depth].index_select(0, order_tensor)
        order_tensor = order_tensor.index_select(
            0, torch.argsort(ordered_scores, stable=True)
        )
    ordered = normalized.index_select(0, order_tensor)
    upper = ordered.cumsum(0)
    lower = torch.cat((ordered.new_zeros(1), upper[:-1]))
    selected_ordered = ordered * power_sum(upper, lower, k)
    selected = torch.empty_like(selected_ordered).scatter_(0, order_tensor, selected_ordered)
    total = selected.sum()
    if not bool(torch.isfinite(total) & (total > 0)):
        raise FloatingPointError("Invalid full-tree GBV maximum distribution")
    return selected / total


def reweight_selected_path(path, selected_target, proposal, k: int):
    length, vocab = proposal.shape
    if path.shape != (length,) or selected_target.shape != (length + 1, vocab) or k < 1:
        raise ValueError("GBV selected-path tensor shape mismatch")
    valid = (torch.isfinite(proposal).all() & (proposal >= 0).all()
             & (proposal.sum(-1) > 0).all())
    selected_mass = proposal.gather(1, path[:, None])[:, 0]
    if not bool(valid & torch.isfinite(selected_target).all()
                & (selected_target >= 0).all() & (selected_mass > 0).all()):
        raise FloatingPointError("Invalid autoregressive proposal or target probabilities")

    # The ordering and exclusive CDFs do not depend on the prefix recurrence.
    # Preparing all depths together removes one full-vocabulary sort, cumsum,
    # and device synchronization from every Python-loop iteration.
    target_rows = selected_target[:length]
    positive = proposal > 0
    log_ratio = torch.where(
        positive, target_rows.log() - proposal.log(),
        torch.full_like(proposal, float("inf")),
    )
    order = torch.argsort(log_ratio, dim=-1, stable=True)
    ordered = proposal.gather(1, order)
    cumulative = ordered.cumsum(-1)
    lower_ordered = torch.cat((proposal.new_zeros((length, 1)), cumulative[:, :-1]), dim=-1)
    lower_by_depth = torch.empty_like(proposal).scatter_(1, order, lower_ordered)

    # t=lambda/(lambda+Q), s=Q/(lambda+Q). Keep s separately near t=1.
    t, s = proposal.new_zeros(()), proposal.new_ones(())
    rows, totals, scales = [], [], []
    for i, token in enumerate(path.tolist()):
        row = proposal[i]
        lower = lower_by_depth[i]
        a = t + s * lower
        b = s * row
        numerator = row * power_sum(a + b, a, k)
        denominator = power_sum(torch.ones_like(t), t, k)
        conditional = numerator / denominator
        total = conditional.sum()
        conditional = conditional / total
        rows.append(conditional)
        totals.append(total)
        scale = a[token] + b[token]
        scales.append(scale)
        t, s = a[token] / scale, b[token] / scale
    result = torch.stack(rows)
    valid = (torch.isfinite(result).all()
             & (torch.stack(totals) > 0).all()
             & torch.isfinite(torch.stack(scales)).all()
             & (torch.stack(scales) > 0).all())
    if not bool(valid):
        raise FloatingPointError("Invalid GBV conditional probabilities or selected prefix")
    return result


def select_and_reweight(paths, target_by_path, q):
    k, length = paths.shape
    if q.shape[0] != length or target_by_path.shape != (k, length + 1, q.shape[-1]):
        raise ValueError("GBV tensor shape mismatch")
    if not bool((torch.isfinite(q) & (q > 0)).all()):
        raise FloatingPointError("GBV requires finite positive draft probabilities")
    chosen_p = target_by_path[:, :length].gather(2, paths[:, :, None])[:, :, 0]
    chosen_q = q[None].expand(k, -1, -1).gather(2, paths[:, :, None])[:, :, 0]
    selected = select_greedy_path(paths, chosen_p, chosen_q)
    result = reweight_selected_path(paths[selected], target_by_path[selected], q, k)
    return selected, result


def select_and_reweight_autoregressive(paths, target_by_path, q_by_path):
    """GBV for an explicitly branch-conditional autoregressive proposal.

    ``q_by_path[j, i]`` is the proposal row conditioned on candidate ``j``'s
    prefix before depth ``i``.  Candidate paths are still IID draws from one
    proposal model; their rows differ only because their realized prefixes do.
    The selected-path correction needs only the conditional rows along the
    selected prefix, which :func:`reweight_selected_path` already supports.
    """
    k, length = paths.shape
    if (q_by_path.ndim != 3
            or q_by_path.shape[:2] != (k, length)
            or target_by_path.shape != (k, length + 1, q_by_path.shape[-1])):
        raise ValueError("Autoregressive GBV tensor shape mismatch")
    if not bool((torch.isfinite(q_by_path) & (q_by_path > 0)).all()):
        raise FloatingPointError(
            "Autoregressive GBV requires finite positive draft probabilities"
        )
    chosen_p = target_by_path[:, :length].gather(
        2, paths[:, :, None]
    )[:, :, 0]
    chosen_q = q_by_path.gather(2, paths[:, :, None])[:, :, 0]
    selected = select_greedy_path(paths, chosen_p, chosen_q)
    result = reweight_selected_path(
        paths[selected], target_by_path[selected], q_by_path[selected], k
    )
    return selected, result


def block_verify(path, p, r, generator=None):
    length, vocab = path.numel(), p.shape[-1]
    w = p.new_ones(())
    best, bonus = 0, None
    for i in range(length + 1):
        proposal = r[i] if i < length else torch.zeros_like(p[i])
        residual = (w * p[i] - proposal).clamp_min(0)
        weights = torch.cat((residual, (1 - w).clamp_min(0).reshape(1)))
        total = weights.sum()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Nonfinite BV residual")
        chosen = int(sample(p[i] if total == 0 else weights / total, generator).item())
        if chosen < vocab:
            best, bonus = i, chosen
        if i < length:
            denom = r[i, path[i]]
            if not bool(denom > 0):
                raise FloatingPointError("Selected token has zero proposal mass")
            w = torch.minimum(torch.ones_like(w), w * p[i, path[i]] / denom)
    if bonus is None:
        raise RuntimeError("BV failed to produce a nonempty output")
    return best, bonus


def block_verify_batched(path, p, r, generator=None):
    """Equivalent BV draw with one batched multinomial and one result sync.

    The acceptance weights are deterministic functions of ``path``, ``p`` and
    ``r``.  Sampling their rows jointly is distributionally identical to the
    reference loop in :func:`block_verify`, but avoids a GPU synchronization
    and a multinomial launch for every draft position.
    """
    length, vocab = path.numel(), p.shape[-1]
    if p.shape != (length + 1, vocab) or r.shape != (length, vocab):
        raise ValueError("BV tensor shape mismatch")
    selected_r = r.gather(1, path[:, None])[:, 0]
    if not bool((torch.isfinite(selected_r) & (selected_r > 0)).all()):
        raise FloatingPointError("Selected token has zero proposal mass")
    selected_p = p[:-1].gather(1, path[:, None])[:, 0]

    w = p.new_ones(())
    weights_by_depth = [w]
    for i in range(length):
        w = torch.minimum(torch.ones_like(w), w * selected_p[i] / selected_r[i])
        weights_by_depth.append(w)
    prefix_weights = torch.stack(weights_by_depth)

    proposal = torch.cat((r, p.new_zeros((1, vocab))), dim=0)
    residual = (prefix_weights[:, None] * p - proposal).clamp_min(0)
    reject = (1 - prefix_weights).clamp_min(0)[:, None]
    weights = torch.cat((residual, reject), dim=-1)
    totals = weights.sum(-1)
    if not bool(torch.isfinite(totals).all()):
        raise FloatingPointError("Nonfinite BV residual")

    positive = totals > 0
    safe_totals = torch.where(positive, totals, torch.ones_like(totals))
    normalized = weights / safe_totals[:, None]
    fallback = torch.cat((p, p.new_zeros((length + 1, 1))), dim=-1)
    normalized = torch.where(positive[:, None], normalized, fallback)
    choices = sample(normalized, generator).tolist()
    accepted_rows = [(i, token) for i, token in enumerate(choices) if token < vocab]
    if not accepted_rows:
        raise RuntimeError("BV failed to produce a nonempty output")
    return accepted_rows[-1]


def block_verify_sparse(path, p, proposal_tokens, proposal_probabilities,
                        generator=None, validate: bool = True):
    """Batched BV without materializing dense proposal rows.

    A finite tree has at most one supported token per leaf at each prefix, so
    its proposal is much narrower than the target vocabulary.  The dense
    residual is still required to sample a correction token, but the proposal
    is subtracted in-place with ``scatter_add_`` instead of first expanding to
    ``length x vocab``.
    """
    length, vocab = path.numel(), p.shape[-1]
    if (p.shape != (length + 1, vocab)
            or proposal_tokens.ndim != 2
            or proposal_tokens.shape != proposal_probabilities.shape
            or proposal_tokens.shape[0] != length):
        raise ValueError("Sparse BV tensor shape mismatch")
    valid = (torch.isfinite(proposal_probabilities).all()
             & (proposal_probabilities >= 0).all()
             & (proposal_probabilities.sum(-1) > 0).all()
             & (proposal_tokens >= 0).all()
             & (proposal_tokens < vocab).all()) if validate else None
    selected_r = torch.where(
        proposal_tokens.eq(path[:, None]), proposal_probabilities,
        torch.zeros_like(proposal_probabilities),
    ).sum(-1)
    if validate and not bool(valid & (selected_r > 0).all()):
        raise FloatingPointError("Selected token has zero proposal mass")
    selected_p = p[:-1].gather(1, path[:, None])[:, 0]

    w = p.new_ones(())
    weights_by_depth = [w]
    for i in range(length):
        w = torch.minimum(torch.ones_like(w), w * selected_p[i] / selected_r[i])
        weights_by_depth.append(w)
    prefix_weights = torch.stack(weights_by_depth)

    residual = prefix_weights[:, None] * p
    residual[:-1].scatter_add_(1, proposal_tokens, -proposal_probabilities)
    residual.clamp_min_(0)
    reject = (1 - prefix_weights).clamp_min(0)[:, None]
    weights = torch.cat((residual, reject), dim=-1)
    totals = weights.sum(-1)
    if validate and not bool(torch.isfinite(totals).all()):
        raise FloatingPointError("Nonfinite BV residual")

    positive = totals > 0
    safe_totals = torch.where(positive, totals, torch.ones_like(totals))
    normalized = weights / safe_totals[:, None]
    fallback = torch.cat((p, p.new_zeros((length + 1, 1))), dim=-1)
    normalized = torch.where(positive[:, None], normalized, fallback)
    choices = sample(normalized, generator).tolist()
    accepted_rows = [(i, token) for i, token in enumerate(choices) if token < vocab]
    if not accepted_rows:
        raise RuntimeError("BV failed to produce a nonempty output")
    return accepted_rows[-1]


def _sparse_residual_totals(p, prefix_weights, proposal_tokens,
                            proposal_probabilities):
    """Compute BV residual masses using only the sparse proposal support."""
    # Multiple tree leaves can propose the same token.  Aggregate their mass,
    # then count each distinct support token exactly once in both terms below.
    equal_tokens = proposal_tokens[:, :, None].eq(proposal_tokens[:, None, :])
    previous_duplicate = torch.tril(equal_tokens, diagonal=-1).any(-1)
    first_occurrence = ~previous_duplicate
    aggregated_proposal = (
        equal_tokens * proposal_probabilities[:, None, :]
    ).sum(-1)
    support_target = p[:-1].gather(1, proposal_tokens)
    support_residual = (
        prefix_weights[:-1, None] * support_target - aggregated_proposal
    ).clamp_min(0)
    unique_support_target_mass = torch.where(
        first_occurrence, support_target, torch.zeros_like(support_target)
    ).sum(-1)
    outside_support_target_mass = (
        p[:-1].sum(-1) - unique_support_target_mass
    ).clamp_min(0)
    draft_rows = (
        prefix_weights[:-1] * outside_support_target_mass
        + torch.where(
            first_occurrence, support_residual, torch.zeros_like(support_residual)
        ).sum(-1)
    )
    final_row = prefix_weights[-1] * p[-1].sum()
    return torch.cat((draft_rows, final_row.reshape(1)))


def block_verify_sparse_lazy(path, p, proposal_tokens, proposal_probabilities,
                             generator=None, validate: bool = True):
    """Sparse BV with deferred correction-token sampling.

    Each BV row is a mixture of a rejection event and a token drawn from its
    normalized residual.  The output uses only the token from the deepest
    accepted row, so tokens from shallower rows can be marginalized exactly:
    sample every row's binary event first, then sample one token at the final
    row.  This is the same output law as :func:`block_verify_sparse`, with one
    full-vocabulary multinomial instead of ``length + 1`` of them.
    """
    length, vocab = path.numel(), p.shape[-1]
    if (p.shape != (length + 1, vocab)
            or proposal_tokens.ndim != 2
            or proposal_tokens.shape != proposal_probabilities.shape
            or proposal_tokens.shape[0] != length):
        raise ValueError("Sparse BV tensor shape mismatch")
    valid = (torch.isfinite(proposal_probabilities).all()
             & (proposal_probabilities >= 0).all()
             & (proposal_probabilities.sum(-1) > 0).all()
             & (proposal_tokens >= 0).all()
             & (proposal_tokens < vocab).all()) if validate else None
    selected_r = torch.where(
        proposal_tokens.eq(path[:, None]), proposal_probabilities,
        torch.zeros_like(proposal_probabilities),
    ).sum(-1)
    if validate and not bool(valid & (selected_r > 0).all()):
        raise FloatingPointError("Selected token has zero proposal mass")
    selected_p = p[:-1].gather(1, path[:, None])[:, 0]

    w = p.new_ones(())
    weights_by_depth = [w]
    for i in range(length):
        w = torch.minimum(torch.ones_like(w), w * selected_p[i] / selected_r[i])
        weights_by_depth.append(w)
    prefix_weights = torch.stack(weights_by_depth)

    residual_totals = _sparse_residual_totals(
        p, prefix_weights, proposal_tokens, proposal_probabilities
    )
    reject = (1 - prefix_weights).clamp_min(0)
    totals = residual_totals + reject
    if validate and not bool(torch.isfinite(totals).all()):
        raise FloatingPointError("Nonfinite BV residual")

    positive = totals > 0
    safe_totals = torch.where(positive, totals, torch.ones_like(totals))
    accept_probability = torch.where(
        positive, residual_totals / safe_totals, torch.ones_like(totals)
    )
    binary = torch.stack((accept_probability, 1 - accept_probability), dim=-1)
    accepted = sample(binary, generator).eq(0)
    depth_ids = torch.arange(length + 1, device=p.device)
    best = torch.where(accepted, depth_ids, -torch.ones_like(depth_ids)).max()
    if validate and not bool(best >= 0):
        raise RuntimeError("BV failed to produce a nonempty output")

    # Only the deepest accepted row's token reaches the output.  Construct that
    # one dense residual after the binary decisions, never length x vocabulary.
    residual_row = prefix_weights[best] * p[best]
    proposal_row = best.clamp_max(length - 1)
    row_probabilities = (
        proposal_probabilities[proposal_row] * best.lt(length)
    )
    residual_row.scatter_add_(
        0, proposal_tokens[proposal_row], -row_probabilities
    )
    residual_row.clamp_min_(0)
    residual_total = residual_totals[best]
    safe_residual_total = torch.where(
        residual_total > 0, residual_total, torch.ones_like(residual_total)
    )
    token_probability = torch.where(
        residual_total > 0, residual_row / safe_residual_total, p[best]
    )
    bonus = sample(token_probability, generator)
    best_value, bonus_value = torch.stack((best, bonus)).tolist()
    # In the trusted fast path the surrounding finite-tree kernel has already
    # checked every probability and support invariant.  Keep the logically
    # necessary terminal check on the values copied for control flow, without
    # introducing an earlier synchronization solely for validation.
    if best_value < 0:
        raise RuntimeError("BV failed to produce a nonempty output")
    return best_value, bonus_value


def _conditional_sparse_paths(paths, selected_path, leaf_probabilities,
                              validate: bool = True):
    """Sparse prefix conditionals of a weighted finite path distribution."""
    if (paths.ndim != 2 or selected_path.shape != (paths.shape[1],)
            or leaf_probabilities.shape != (paths.shape[0],)):
        raise ValueError("Finite-path conditional shape mismatch")
    matches = paths.eq(selected_path[None])
    active = torch.cat((
        torch.ones((paths.shape[0], 1), dtype=torch.bool, device=paths.device),
        matches[:, :-1].cumprod(1).bool(),
    ), dim=1)
    active_weights = leaf_probabilities[:, None] * active
    totals = active_weights.sum(0)
    probabilities_by_leaf = (
        active_weights / totals[None]
    ).transpose(0, 1).contiguous()
    if validate:
        valid = (matches.all(1).any()
                 & torch.isfinite(probabilities_by_leaf).all()
                 & (probabilities_by_leaf >= 0).all()
                 & (totals > 0).all())
        if not bool(valid):
            raise FloatingPointError("Selected path has zero finite-tree probability")
    return paths.transpose(0, 1).contiguous(), probabilities_by_leaf


def _tree_subtree_indices_and_depths(path_nodes: torch.Tensor):
    if path_nodes.ndim != 2:
        raise ValueError("Tree nodes must be [leaves, depth]")
    leaves, length = path_nodes.shape
    if leaves == 0 or length == 0:
        raise ValueError("Tree must have at least one leaf and depth")
    max_node = int(path_nodes.max().item())
    subtree = [[] for _ in range(max_node + 1)]
    node_depth = [0] * (max_node + 1)
    for leaf_idx in range(leaves):
        for depth, node in enumerate(path_nodes[leaf_idx].tolist(), start=1):
            nid = int(node)
            subtree[nid].append(leaf_idx)
            node_depth[nid] = depth
    subtree[0] = list(range(leaves))
    return subtree, node_depth, length


def _tree_subtree_csr(path_nodes: torch.Tensor):
    """Compact CSR-style representation of leaf indices for every tree node."""
    if path_nodes.ndim != 2:
        raise ValueError("Tree nodes must be [leaves, depth]")
    leaves, length = path_nodes.shape
    if leaves == 0 or length == 0:
        raise ValueError("Tree must have at least one leaf and depth")
    path_nodes_cpu = path_nodes.cpu()
    max_node = int(path_nodes_cpu.max().item())
    buckets = [[] for _ in range(max_node + 1)]
    node_depth = [0] * (max_node + 1)
    buckets[0] = list(range(leaves))
    for leaf_idx in range(leaves):
        for depth, node in enumerate(path_nodes_cpu[leaf_idx].tolist(), start=1):
            nid = int(node)
            buckets[nid].append(int(leaf_idx))
            node_depth[nid] = depth
    start = torch.full((max_node + 1,), -1, dtype=torch.long, device=path_nodes.device)
    count = torch.zeros((max_node + 1,), dtype=torch.long, device=path_nodes.device)
    flattened = []
    cursor = 0
    for node, leaf_indices in enumerate(buckets):
        if not leaf_indices:
            continue
        start[node] = cursor
        leaf_count = len(leaf_indices)
        count[node] = leaf_count
        flattened.extend(leaf_indices)
        cursor += leaf_count
    return (
        torch.tensor(flattened, dtype=torch.long, device=path_nodes.device),
        start,
        count,
        torch.tensor(node_depth, dtype=torch.long, device=path_nodes.device),
        length,
    )


def tree_block_verify_terminal_mass(parents, tokens, all_p, generator=None,
                                    validate: bool = True, *,
                                    prefix_mode: str = "batched",
                                    exit_mode: str = "internal"):
    """Collapse ancestral Target tree sampling into one terminal block draw.

    DDTree's output law can be viewed as sampling a Target token at the root
    and at every accepted child until the sample leaves the verified tree (its
    official implementation draws every node row in one batch).  The same
    random continuation can instead be partitioned by its *first exit node*.
    For a node ``v``, its terminal mass is

    ``P_target(prefix(v)) * (1 - P_target(children(v) | prefix(v)))``.

    Sampling that node and then a Target token outside its child set produces
    exactly the same variable-length output law in exact arithmetic.  Both
    categorical draws stay on device; the trusted ``validate=False`` path
    copies only the final ``(node, token)`` pair to the host.  Validation adds
    host synchronizations and is intended for callers with untrusted inputs.
    This changes execution architecture, not the DDTree proposal or its
    accepted-prefix distribution.
    """
    if (prefix_mode not in {"batched", "serial"}
            or exit_mode not in {"internal", "dense", "complement", "joint"}):
        raise ValueError("Unknown terminal-mass ablation mode")
    parents = list(parents)
    tokens = list(tokens)
    node_count = len(parents)
    if (node_count < 1 or len(tokens) != node_count - 1
            or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[-1] < 1):
        raise ValueError("Terminal-mass tree tensor shape mismatch")
    if not all_p.is_floating_point():
        raise TypeError("Terminal-mass tree probabilities must be floating point")
    if parents[0] != -1 or any(parent < 0 or parent >= node
                               for node, parent in enumerate(parents[1:], 1)):
        raise ValueError("Tree parents must precede their children")
    if len(set(zip(parents[1:], tokens))) != len(tokens):
        raise ValueError("A tree parent cannot repeat a child token")

    vocab = all_p.shape[-1]
    if any(token < 0 or token >= vocab for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & torch.isclose(
                     all_p.sum(-1), all_p.new_ones(node_count),
                     rtol=max(1e-10, 8 * torch.finfo(all_p.dtype).eps), atol=1e-12,
                 ).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for tree block")

    device = all_p.device
    if node_count == 1:
        return [], [], int(sample(all_p[0], generator))

    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    edge_probabilities = all_p[edge_parents, edge_tokens]

    # Sum the actual uncovered weights.  Subtracting covered mass from 1 can
    # erase a positive exit tail: [1., 1e-20] with child token 0 is a valid
    # FP64 softmax row, yet 1 - p[0] == 0.  Only internal nodes need a dense
    # masked copy; leaves have exit probability 1.  This trades O(I * vocab)
    # temporary storage for cancellation-free exit masses (I internal nodes).
    if exit_mode == "joint":
        # The joint first-exit draw below uses the original Target rows, so no
        # per-node exit totals are required here.
        exit_mass = internal_map = exit_rows = None
    elif exit_mode == "complement":
        # Fast model-probability path: softmax rows are normalized, so only the
        # sparse child support is needed to obtain each first-exit mass.  The
        # validation above keeps the public input contract explicit.  The
        # cancellation-safe internal mode remains the default for adversarial
        # probability rows whose uncovered tail is below one FP64 ulp.
        covered_weights = all_p.new_zeros(node_count)
        covered_weights.scatter_add_(0, edge_parents, edge_probabilities)
        exit_mass = (1 - covered_weights).clamp_min(0)
        internal_map = exit_rows = None
    else:
        internal_nodes = (list(range(node_count)) if exit_mode == "dense"
                          else sorted(set(parents[1:])))
        internal_lookup = [-1] * node_count
        for index, node in enumerate(internal_nodes):
            internal_lookup[node] = index
        internal_indices = torch.tensor(internal_nodes, dtype=torch.long, device=device)
        internal_map = torch.tensor(internal_lookup, dtype=torch.long, device=device)
        edge_rows = internal_map.index_select(0, edge_parents)
        exit_rows = all_p.index_select(0, internal_indices)
        exit_rows[edge_rows, edge_tokens] = 0
        tail_weights = exit_rows.sum(-1)
        covered_weights = all_p.new_zeros(len(internal_nodes))
        covered_weights.scatter_add_(0, edge_rows, edge_probabilities)
        row_weights = tail_weights + covered_weights
        edge_probabilities = edge_probabilities / row_weights.index_select(0, edge_rows)
        exit_mass = all_p.new_ones(node_count).index_copy_(
            0, internal_indices, tail_weights / row_weights
        )

    # Compile padded ancestor lists on the host, then gather and multiply all
    # prefixes in two tensor operations.  The old recurrence launched a tiny
    # multiply and copy for every node.  Padding uses the root's unit weight.
    if prefix_mode == "serial":
        # Law-preserving ablation: isolate batched ancestor computation.
        masses = [all_p.new_ones(())]
        for node, parent in enumerate(parents[1:], 1):
            masses.append(masses[parent] * edge_probabilities[node - 1])
        prefix_mass = torch.stack(masses)
    else:
        ancestors = [[]]
        for node, parent in enumerate(parents[1:], 1):
            ancestors.append(ancestors[parent] + [node])
        depth = max(map(len, ancestors))
        ancestor_indices = torch.tensor(
            [path + [0] * (depth - len(path)) for path in ancestors],
            dtype=torch.long, device=device,
        )
        node_weights = torch.cat((all_p.new_ones(1), edge_probabilities))
        prefix_mass = node_weights[ancestor_indices].prod(-1)
    if exit_mode == "joint":
        # Sample (terminal node, correction token) as one categorical event.
        # Its weight is Target(prefix(node)) * Target(token | prefix(node));
        # tree-child events are zero because they continue rather than exit.
        # This is the terminal block law without two dependent multinomials.
        joint_weights = prefix_mass[:, None] * all_p
        joint_weights[edge_parents, edge_tokens] = 0
        joint_total = joint_weights.sum()
        if validate and not bool(torch.isfinite(joint_total)
                                 & (joint_total > 0)
                                 & torch.isclose(
                                     joint_total, all_p.new_ones(()),
                                     rtol=max(1e-10, 8 * torch.finfo(all_p.dtype).eps),
                                     atol=1e-12,
                                 )):
            raise FloatingPointError("Joint terminal events do not partition Target law")
        joint_choice = sample(joint_weights.reshape(-1), generator)
        terminal_node_tensor = torch.div(
            joint_choice, vocab, rounding_mode="floor"
        )
        bonus_tensor = torch.remainder(joint_choice, vocab)
    else:
        terminal_mass = prefix_mass * exit_mass
        total_terminal_mass = terminal_mass.sum()
        if validate and not bool(torch.isfinite(total_terminal_mass)
                                 & (total_terminal_mass > 0)
                                 & torch.isclose(
                                     total_terminal_mass, all_p.new_ones(()),
                                     rtol=max(1e-10, 8 * torch.finfo(all_p.dtype).eps),
                                     atol=1e-12,
                                 )):
            raise FloatingPointError("Tree terminal masses do not partition Target law")
        terminal_node_tensor = sample(
            terminal_mass / total_terminal_mass, generator
        )

        target_row = all_p.index_select(
            0, terminal_node_tensor.reshape(1)
        )[0]
    if exit_mode == "complement":
        # Remove only children of the selected terminal node from one Target
        # row.  Repeated token labels below other parents contribute zero.
        selected_edges = edge_parents.eq(terminal_node_tensor).to(all_p.dtype)
        correction_weights = target_row.clone()
        correction_weights.scatter_add_(
            0, edge_tokens, -edge_probabilities * selected_edges
        )
        correction_weights.clamp_min_(0)
    elif exit_mode != "joint":
        terminal_internal = internal_map.index_select(
            0, terminal_node_tensor.reshape(1)
        )
        exit_row = exit_rows.index_select(0, terminal_internal.clamp_min(0))[0]
        correction_weights = torch.where(
            terminal_internal[0] >= 0, exit_row, target_row
        )
    if exit_mode != "joint":
        correction_total = correction_weights.sum()
        if validate and not bool(torch.isfinite(correction_total)
                                 & (correction_total > 0)):
            raise FloatingPointError("Selected terminal node has no exit token")
        bonus_tensor = sample(correction_weights / correction_total, generator)

    # One device-to-host synchronization for all control-flow values.
    terminal_node, bonus = torch.stack((
        terminal_node_tensor.to(torch.long), bonus_tensor.to(torch.long)
    )).tolist()
    nodes = []
    node = int(terminal_node)
    while node:
        nodes.append(node)
        node = parents[node]
    nodes.reverse()
    output_tokens = [tokens[node - 1] for node in nodes]
    return nodes, output_tokens, int(bonus)


def tree_verify_ancestral_lazy_projection(
        parents, tokens, final_hidden, lm_head, temperature: float,
        probability_dtype=torch.float64, generator=None, validate: bool = True):
    """Apply the vocabulary head only to rows an exact tree walk can need.

    Official DDTree projects and samples all ``B + 1`` verified tree rows.
    A leaf row cannot influence which leaf is reached; it is needed only when
    that particular leaf becomes the terminal node.  This verifier therefore
    projects all internal rows as one efficient batch, performs the same
    ancestral Target walk, and projects at most the one reached leaf on demand.

    The output law is unchanged.  For ``I`` internal nodes and ``F`` leaves,
    the vocabulary projection uses ``I + 1[terminal is a leaf]`` rows instead
    of ``I + F``.  Its expected projected-row count is at most ``I + 1`` and is
    strictly below DDTree whenever the finite tree has more than one leaf.
    """
    parents = list(parents)
    tokens = list(tokens)
    node_count = len(parents)
    if (node_count < 1 or len(tokens) != node_count - 1
            or final_hidden.ndim != 2 or final_hidden.shape[0] != node_count):
        raise ValueError("Lazy-projection tree tensor shape mismatch")
    if parents[0] != -1 or any(parent < 0 or parent >= node
                               for node, parent in enumerate(parents[1:], 1)):
        raise ValueError("Tree parents must precede their children")
    if len(set(zip(parents[1:], tokens))) != len(tokens):
        raise ValueError("A tree parent cannot repeat a child token")

    internal_nodes = sorted(set(parents[1:]))
    internal_lookup = {node: row for row, node in enumerate(internal_nodes)}
    children = {
        (parents[node], tokens[node - 1]): node
        for node in range(1, node_count)
    }
    if not internal_nodes:
        logits = lm_head(final_hidden[:1])[0]
        p = probabilities(logits, temperature, probability_dtype)
        if validate and not bool(torch.isfinite(p).all() & (p >= 0).all()
                                 & (p.sum() > 0)):
            raise FloatingPointError("Invalid lazy-projection leaf probabilities")
        return [], [], int(sample(p, generator)), {
            "internal_projected_rows": 0,
            "leaf_projected_rows": 1,
            "projected_rows": 1,
            "total_tree_rows": 1,
        }

    device = final_hidden.device
    internal_indices = torch.tensor(
        internal_nodes, dtype=torch.long, device=device
    )
    internal_logits = lm_head(final_hidden.index_select(0, internal_indices))
    internal_p = probabilities(
        internal_logits, temperature, probability_dtype
    )
    vocab = internal_p.shape[-1]
    if any(token < 0 or token >= vocab for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(internal_p).all() & (internal_p >= 0).all()
                 & (internal_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid lazy-projection internal probabilities")

    # Independent draws for unused internal rows can be discarded, exactly as
    # in official DDTree.  Keeping them batched preserves GPU utilization.
    posterior_tokens = sample(internal_p, generator).tolist()
    nodes = []
    output_tokens = []
    node = 0
    leaf_projected = 0
    while node in internal_lookup:
        bonus = int(posterior_tokens[internal_lookup[node]])
        child = children.get((node, bonus))
        if child is None:
            break
        node = child
        nodes.append(node)
        output_tokens.append(bonus)
    else:
        # Exactly one reached leaf needs a full-vocabulary Target row.
        leaf_logits = lm_head(final_hidden[node:node + 1])[0]
        leaf_p = probabilities(leaf_logits, temperature, probability_dtype)
        if validate and not bool(torch.isfinite(leaf_p).all() & (leaf_p >= 0).all()
                                 & (leaf_p.sum() > 0)):
            raise FloatingPointError("Invalid lazy-projection leaf probabilities")
        bonus = int(sample(leaf_p, generator))
        leaf_projected = 1

    return nodes, output_tokens, bonus, {
        "internal_projected_rows": len(internal_nodes),
        "leaf_projected_rows": leaf_projected,
        "projected_rows": len(internal_nodes) + leaf_projected,
        "total_tree_rows": node_count,
    }


def tree_verify_internal_ancestral_batched(
        parents, tokens, internal_p, generator=None, validate: bool = True):
    """Walk a DDTree from posterior rows for internal nodes only.

    A leaf has no proposed child, so its posterior row cannot affect whether
    the leaf is reached.  When the walk reaches a leaf this function returns
    ``bonus == -1`` and the leaf node separately; the caller must obtain and
    sample that one leaf posterior row.  Otherwise ``bonus`` is the exact exit
    token sampled at an internal node.
    """
    parents = list(parents)
    tokens = list(tokens)
    node_count = len(parents)
    internal_nodes = sorted(set(parents[1:]))
    if (node_count < 2 or len(tokens) != node_count - 1
            or not internal_nodes or internal_nodes[0] != 0
            or internal_p.ndim != 2
            or internal_p.shape[0] != len(internal_nodes)
            or internal_p.shape[-1] < 1):
        raise ValueError("Internal-only DDTree posterior tensor shape mismatch")
    if not internal_p.is_floating_point():
        raise TypeError("DDTree posterior probabilities must be floating point")
    if parents[0] != -1 or any(parent < 0 or parent >= node
                               for node, parent in enumerate(parents[1:], 1)):
        raise ValueError("Tree parents must precede their children")
    if len(set(zip(parents[1:], tokens))) != len(tokens):
        raise ValueError("A tree parent cannot repeat a child token")
    vocab = internal_p.shape[-1]
    if any(token < 0 or token >= vocab for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(internal_p).all() & (internal_p >= 0).all()
                 & (internal_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid internal Target probabilities for DDTree")

    internal_lookup = {node: row for row, node in enumerate(internal_nodes)}
    children = {
        (parents[node], tokens[node - 1]): node
        for node in range(1, node_count)
    }
    # As in official DDTree, draw the available independent posterior rows in
    # one GPU batch.  Draws for internal nodes outside the realized path are
    # discarded and do not change the ancestral distribution.
    posterior_tokens = sample(internal_p, generator).tolist()
    nodes = []
    output_tokens = []
    node = 0
    while True:
        bonus = int(posterior_tokens[internal_lookup[node]])
        child = children.get((node, bonus))
        if child is None:
            return nodes, output_tokens, bonus, -1, {
                "internal_probability_rows": len(internal_nodes),
                "leaf_probability_rows": 0,
                "total_tree_rows": node_count,
            }
        node = child
        nodes.append(node)
        output_tokens.append(bonus)
        if node not in internal_lookup:
            return nodes, output_tokens, -1, node, {
                "internal_probability_rows": len(internal_nodes),
                "leaf_probability_rows": 1,
                "total_tree_rows": node_count,
            }


def tree_verify_ancestral_batched(parents, tokens, all_p, generator=None,
                                  validate: bool = True):
    """Official DDTree posterior sampling: draw every verified row in one batch.

    The unused node draws are independent and can be discarded, so drawing all
    rows before following the root-to-exit path has exactly the same law as
    ancestral Target sampling.  This helper exists to keep architecture
    comparisons honest: the DDTree reference uses one 2-D multinomial and one
    device-to-host transfer, not a host-synchronized draw at every depth.
    """
    parents = list(parents)
    tokens = list(tokens)
    node_count = len(parents)
    if (node_count < 1 or len(tokens) != node_count - 1
            or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[-1] < 1):
        raise ValueError("DDTree posterior tensor shape mismatch")
    if not all_p.is_floating_point():
        raise TypeError("DDTree posterior probabilities must be floating point")
    if parents[0] != -1 or any(parent < 0 or parent >= node
                               for node, parent in enumerate(parents[1:], 1)):
        raise ValueError("Tree parents must precede their children")
    if len(set(zip(parents[1:], tokens))) != len(tokens):
        raise ValueError("A tree parent cannot repeat a child token")
    vocab = all_p.shape[-1]
    if any(token < 0 or token >= vocab for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for DDTree")

    posterior_tokens = sample(all_p, generator).tolist()
    children = {
        (parents[node], tokens[node - 1]): node
        for node in range(1, node_count)
    }
    nodes = []
    output_tokens = []
    node = 0
    bonus = int(posterior_tokens[node])
    while (node, bonus) in children:
        node = children[node, bonus]
        nodes.append(node)
        output_tokens.append(bonus)
        bonus = int(posterior_tokens[node])
    return nodes, output_tokens, bonus


def tree_block_verify_recycle(paths, path_nodes, children, all_p, leaf_probabilities,
                              proposal_token_probabilities, k: int,
                              generator=None, segment_verifier="sparse",
                              control_device="device", host_generator=None,
                              precompute_subtrees: bool = False):
    """Compose GBV/BV kernels when a correction lands on a tree branch.

    A regular BV call emits a Target-distributed prefix ending in a correction
    token.  If that token is already a child in the verified tree, it can serve
    as the anchor of another conditional GBV/BV call without a new Target
    forward.  Repeating this operation is distributionally the same as
    composing ordinary lossless decoding rounds, while reusing off-path tree
    nodes that the first selected path would otherwise discard.
    """
    if paths.ndim != 2:
        raise ValueError("Tree recycling requires a leaf-by-depth path matrix")
    leaves, length = paths.shape
    vocab = all_p.shape[-1]
    if (length < 1 or k < 1 or path_nodes.shape != paths.shape
            or all_p.ndim != 2
            or leaf_probabilities.shape != (leaves,)
            or proposal_token_probabilities.shape != paths.shape):
        raise ValueError("Tree recycling tensor shape mismatch")
    valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
             & (all_p.sum(-1) > 0).all()
             & torch.isfinite(leaf_probabilities).all()
             & (leaf_probabilities > 0).all()
             & (leaf_probabilities.sum() > 0)
             & torch.isfinite(proposal_token_probabilities).all()
             & (proposal_token_probabilities > 0).all()
             & (paths >= 0).all() & (paths < vocab).all()
             & (path_nodes > 0).all() & (path_nodes < all_p.shape[0]).all())
    if not bool(valid):
        raise FloatingPointError("Invalid tree recycling probabilities or nodes")

    if not isinstance(children, dict):
        raise TypeError("Tree recycling children must be an edge dictionary")
    if segment_verifier not in {"sparse", "sparse_lazy"}:
        raise ValueError("Invalid recycle segment verifier")
    if control_device not in {"device", "cpu"}:
        raise ValueError("Invalid recycle control device")
    if control_device == "cpu" and host_generator is None:
        raise ValueError("CPU recycle control requires a host generator")
    segment_verify = (
        block_verify_sparse if segment_verifier == "sparse"
        else block_verify_sparse_lazy
    )
    segment_generator = (
        host_generator if control_device == "cpu" else generator
    )
    control_paths = paths.detach().cpu() if control_device == "cpu" else paths
    control_path_nodes = (
        path_nodes.detach().cpu() if control_device == "cpu" else path_nodes
    )
    control_leaf_probabilities = (
        leaf_probabilities.detach().cpu()
        if control_device == "cpu" else leaf_probabilities
    )
    control_proposal_probabilities = (
        proposal_token_probabilities.detach().cpu()
        if control_device == "cpu" else proposal_token_probabilities
    )
    if precompute_subtrees:
        (
            subtree_flattened_leaf_indices,
            subtree_start,
            subtree_count,
            node_depth_tensor,
            tree_length,
        ) = _tree_subtree_csr(control_path_nodes)
        all_p_control = all_p.cpu() if control_device == "cpu" else all_p
    else:
        subtree_indices, node_depth, tree_length = _tree_subtree_indices_and_depths(
            control_path_nodes
        )
        all_p_control = all_p
        node_depth_tensor = None
        subtree_flattened_leaf_indices = None
        subtree_start = None
        subtree_count = None
    if tree_length != length:
        raise ValueError("Tree height mismatch with path length")

    output_nodes, output_tokens = [], []
    segments, recycled_corrections = 0, 0
    node_count = len(subtree_start) if precompute_subtrees else len(subtree_indices)
    current_node, depth = 0, 0

    def finish_with_ancestral_target(node):
        """Continue through verified nodes outside the finite proposal trie."""
        while True:
            row = (
                all_p[node].cpu()
                if control_device == "cpu" else all_p[node]
            )
            token = int(sample(row, segment_generator).item())
            child = children.get((node, token))
            if child is None:
                return output_nodes, output_tokens, token, {
                    "segments": segments,
                    "recycled_corrections": recycled_corrections,
                }
            output_nodes.append(child)
            output_tokens.append(token)
            node = child

    while depth < length:
        segments += 1
        if current_node >= node_count:
            return finish_with_ancestral_target(current_node)
        if precompute_subtrees:
            start = int(subtree_start[current_node].item())
            active_size = int(subtree_count[current_node].item())
            if start < 0 or active_size <= 0:
                return finish_with_ancestral_target(current_node)
            active_indices = subtree_flattened_leaf_indices.narrow(0, start, active_size)
            active_depth = int(node_depth_tensor[current_node].item())
        else:
            active_indices = torch.tensor(
                subtree_indices[current_node], dtype=torch.long,
                device=control_paths.device,
            )
            active_depth = node_depth[current_node]
        if active_indices.numel() == 0:
            return finish_with_ancestral_target(current_node)
        if active_depth != depth:
            # Node depth and current depth must stay synchronized.
            raise RuntimeError("Tree node depth drifted from the verified prefix")
        active_paths = control_paths.index_select(0, active_indices)[:, depth:]
        active_nodes = control_path_nodes.index_select(0, active_indices)[:, depth:]
        row_nodes = torch.cat((
            torch.full((active_indices.numel(), 1), current_node,
                       dtype=torch.long, device=control_paths.device),
            active_nodes[:, :-1],
        ), dim=1)
        device_row_nodes = row_nodes.to(paths.device)
        device_active_paths = active_paths.to(paths.device)
        target_token_probabilities = all_p_control[
            row_nodes if control_device == "cpu" else device_row_nodes,
            active_paths if control_device == "cpu" else device_active_paths
        ]
        if control_device == "cpu":
            target_token_probabilities = target_token_probabilities.cpu()
        active_proposal_tokens = control_proposal_probabilities.index_select(
            0, active_indices
        )[:, depth:]
        active_leaf_probabilities = control_leaf_probabilities.index_select(
            0, active_indices
        )
        active_leaf_probabilities = (
            active_leaf_probabilities / active_leaf_probabilities.sum()
        )
        if active_indices.numel() == 1:
            selected_active = None
            chosen_active = 0
        else:
            selected_active = greedy_max_distribution(
                active_paths, target_token_probabilities, active_proposal_tokens,
                active_leaf_probabilities, k, validate=False,
            )
            selection_generator = (
                host_generator if control_device == "cpu" else generator
            )
            chosen_active = int(sample(selected_active, selection_generator).item())
        chosen_leaf = int(active_indices[chosen_active])

        if selected_active is None:
            sparse_tokens = active_paths[chosen_active:chosen_active + 1].t().contiguous()
            # A singleton subtree is a deterministic proposal at *every*
            # remaining depth.  Keep the same [depth, support] layout as the
            # general sparse conditional path; collapsing this to one row
            # breaks recycled segments whenever more than one level remains.
            sparse_probabilities = active_leaf_probabilities.new_ones(
                sparse_tokens.shape
            )
        else:
            sparse_tokens, sparse_probabilities = _conditional_sparse_paths(
                active_paths, active_paths[chosen_active], selected_active,
                validate=False,
            )
        chosen_nodes = control_path_nodes[chosen_leaf]
        target_nodes = torch.cat((
            torch.tensor([current_node], dtype=torch.long,
                         device=control_path_nodes.device),
            chosen_nodes[depth:],
        )).to(paths.device)
        accepted, bonus = segment_verify(
            (paths[chosen_leaf, depth:] if control_device == "device" else paths[chosen_leaf, depth:].cpu()),
            (all_p_control.index_select(0, target_nodes.cpu() if control_device == "cpu" else target_nodes)),
            (sparse_tokens.to(paths.device) if control_device == "device" else sparse_tokens.cpu()),
            (sparse_probabilities.to(paths.device) if control_device == "device" else sparse_probabilities.cpu()),
            segment_generator, validate=False,
        )
        accepted = int(accepted.item()) if isinstance(accepted, torch.Tensor) else int(accepted)
        bonus = int(bonus.item()) if isinstance(bonus, torch.Tensor) else int(bonus)
        segment_nodes = chosen_nodes[depth:depth + accepted].tolist()
        segment_tokens = paths[chosen_leaf, depth:depth + accepted].tolist()
        output_nodes.extend(segment_nodes)
        output_tokens.extend(segment_tokens)
        parent = current_node if accepted == 0 else segment_nodes[-1]
        child = children.get((parent, bonus))
        if child is None:
            return output_nodes, output_tokens, bonus, {
                "segments": segments,
                "recycled_corrections": recycled_corrections,
            }

        # The correction is already a verified tree node.  Commit it and apply
        # another exact GBV/BV kernel to the remaining conditional subtree.
        output_nodes.append(child)
        output_tokens.append(bonus)
        recycled_corrections += 1
        current_node = child
        depth += accepted + 1

    bonus = int(sample(
        all_p_control[current_node] if control_device == "cpu" else all_p[current_node],
        segment_generator,
    ).item())
    return output_nodes, output_tokens, bonus, {
        "segments": segments,
        "recycled_corrections": recycled_corrections,
    }


def token_verify(path, p, q, generator=None):
    for i, token in enumerate(path):
        ratio = p[i, token] / q[i, token]
        if torch.rand((), device=q.device, dtype=q.dtype, generator=generator) >= ratio:
            residual = (p[i] - q[i]).clamp_min(0)
            total = residual.sum()
            if not bool(total > 0):
                raise FloatingPointError("Rejection with empty residual")
            return i, int(sample(residual / total, generator).item())
    return len(path), int(sample(p[len(path)], generator).item())


def matching_verify(path, p, generator=None):
    """DFlash rule: match a greedy draft against sampled Target tokens."""
    if p.shape[0] != path.numel() + 1:
        raise ValueError("Matching verifier needs one Target row per draft token plus bonus")
    posterior = sample(p, generator)
    accepted = int((path == posterior[:-1]).long().cumprod(0).sum().item())
    return accepted, int(posterior[accepted].item())
