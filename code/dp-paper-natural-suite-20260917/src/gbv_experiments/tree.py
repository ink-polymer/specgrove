from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import heapq
import math

import torch


@dataclass
class Tree:
    tokens: list[int]  # Excludes the root anchor; node i has tokens[i-1].
    parents: list[int]  # Includes root 0, whose parent is -1.
    depths: list[int]
    path_nodes: list[list[int]]
    draft_prefix_log_masses: list[float] | None = None

    def visibility(self, device=None):
        n = len(self.parents)
        visible = torch.zeros((n, n), dtype=torch.bool)
        visible[0, 0] = True
        for i in range(1, n):
            visible[i] = visible[self.parents[i]]
            visible[i, i] = True
        return visible.to(device)

    def mask(self, prefix_length: int, dtype, device):
        n = len(self.parents)
        mask = torch.full((n, prefix_length + n), float("-inf"), dtype=dtype, device=device)
        mask[:, :prefix_length] = 0
        mask[:, prefix_length:].masked_fill_(self.visibility(device), 0)
        return mask[None, None]


@dataclass
class AdaptivePathProposal:
    """Finite autoregressive proposal induced by high-mass draft paths.

    ``paths`` are the support leaves.  ``leaf_probabilities`` is their
    normalized draft mass, and ``children`` stores the corresponding sparse
    conditional distribution at every supported prefix.
    """

    paths: torch.Tensor
    leaf_probabilities: torch.Tensor
    token_probabilities: torch.Tensor
    children: dict[tuple[int, ...], tuple[list[int], list[float]]]
    vocab_size: int

    def sample(self, count: int, generator=None):
        if count < 1:
            raise ValueError("Adaptive proposal sample count must be positive")
        indices = torch.multinomial(
            self.leaf_probabilities, count, replacement=True, generator=generator
        )
        return self.paths.index_select(0, indices), self.token_probabilities.index_select(0, indices)

    def conditional_rows(self, path: torch.Tensor, leaf_probabilities: torch.Tensor | None = None):
        if path.ndim != 1 or path.numel() != self.paths.shape[1]:
            raise ValueError("Adaptive proposal path shape mismatch")
        if leaf_probabilities is not None:
            if (leaf_probabilities.shape != self.leaf_probabilities.shape
                    or not bool(torch.isfinite(leaf_probabilities).all()
                                & (leaf_probabilities >= 0).all()
                                & (leaf_probabilities.sum() > 0))):
                raise FloatingPointError("Invalid finite-tree leaf probabilities")
            rows = self.leaf_probabilities.new_zeros((path.numel(), self.vocab_size))
            active = torch.ones(self.paths.shape[0], dtype=torch.bool, device=self.paths.device)
            for depth, token in enumerate(path.tolist()):
                active_weights = leaf_probabilities[active]
                rows[depth].scatter_add_(0, self.paths[active, depth], active_weights)
                rows[depth] /= active_weights.sum()
                active &= self.paths[:, depth].eq(token)
            if not bool(active.any() & torch.isfinite(rows).all()
                        & (rows.gather(1, path[:, None])[:, 0] > 0).all()):
                raise FloatingPointError("Selected path has zero finite-tree probability")
            return rows
        rows = self.leaf_probabilities.new_zeros((path.numel(), self.vocab_size))
        prefix: tuple[int, ...] = ()
        for depth, token in enumerate(path.tolist()):
            support = self.children.get(prefix)
            if support is None or token not in support[0]:
                raise ValueError("Path is outside the adaptive proposal support")
            tokens, probabilities = support
            indices = torch.tensor(tokens, device=rows.device)
            rows[depth, indices] = rows.new_tensor(probabilities)
            prefix += (token,)
        return rows

    def conditional_sparse(self, path: torch.Tensor,
                           leaf_probabilities: torch.Tensor | None = None):
        """Return supported token ids and probabilities along ``path``.

        Unlike :meth:`conditional_rows`, this representation scales with the
        number of finite-tree leaves rather than the model vocabulary.  Each
        column represents one leaf; leaves outside the active prefix carry
        zero probability and duplicate token ids are combined by the verifier.
        """
        if path.ndim != 1 or path.numel() != self.paths.shape[1]:
            raise ValueError("Adaptive proposal path shape mismatch")
        weights = self.leaf_probabilities if leaf_probabilities is None else leaf_probabilities
        if (weights.shape != self.leaf_probabilities.shape
                or not bool(torch.isfinite(weights).all()
                            & (weights >= 0).all()
                            & (weights.sum() > 0))):
            raise FloatingPointError("Invalid finite-tree leaf probabilities")

        # Keep one sparse column per leaf. Duplicate token ids are intentional:
        # block_verify_sparse combines them with scatter_add_, which avoids a
        # unique/sort kernel at every depth.
        matches = self.paths.eq(path[None])
        active = torch.cat((
            torch.ones((self.paths.shape[0], 1), dtype=torch.bool,
                       device=self.paths.device),
            matches[:, :-1].cumprod(1).bool(),
        ), dim=1)
        active_weights = weights[:, None] * active
        totals = active_weights.sum(0)
        probabilities = (active_weights / totals[None]).transpose(0, 1).contiguous()
        token_ids = self.paths.transpose(0, 1).contiguous()
        valid = (matches.all(1).any() & torch.isfinite(probabilities).all()
                 & (probabilities >= 0).all() & (totals > 0).all())
        if not bool(valid):
            raise FloatingPointError("Selected path has zero finite-tree probability")
        return token_ids, probabilities


def sampled_tree(paths, share=True):
    tokens, parents, depths, path_nodes = [], [-1], [0], []
    children = {}
    for path in paths.tolist():
        parent, nodes = 0, []
        for depth, token in enumerate(path, 1):
            child = children.get((parent, token)) if share else None
            if child is None:
                child = len(parents)
                children[(parent, token)] = child
                tokens.append(token)
                parents.append(parent)
                depths.append(depth)
            nodes.append(child)
            parent = child
        path_nodes.append(nodes)
    return Tree(tokens, parents, depths, path_nodes)


def probability_tree(q, budget: int, depth_reward: float = 0.0,
                     depth_reward_schedule: tuple[float, ...] | None = None,
                     rank_caps: tuple[int, ...] | None = None,
                     adaptive_min_budget: int | None = None,
                     confidence_threshold: float | None = None):
    """Best-first prefix tree with an optional per-committed-token utility.

    ``depth_reward=0`` is exactly DDTree's product-probability tree.  A positive
    reward changes only which proposal prefixes receive the fixed node budget;
    exact Target verification remains independent of this ranking heuristic.
    """
    if not math.isfinite(depth_reward):
        raise ValueError("Tree depth reward must be finite")
    length, vocab = q.shape
    if (depth_reward_schedule is not None
            and (len(depth_reward_schedule) != length
                 or any(not math.isfinite(value)
                        for value in depth_reward_schedule))):
        raise ValueError("Invalid tree depth-reward schedule")
    if (rank_caps is not None
            and (len(rank_caps) != length
                 or any(not isinstance(value, int)
                        or isinstance(value, bool) or value < 1
                        for value in rank_caps))):
        raise ValueError("Invalid per-depth tree rank caps")
    rewards = (
        tuple(float(value) for value in depth_reward_schedule)
        if depth_reward_schedule is not None
        else (float(depth_reward),) * length
    )
    caps = (
        tuple(min(int(value), budget, vocab) for value in rank_caps)
        if rank_caps is not None else (min(budget, vocab),) * length
    )
    values, indices = torch.topk(q, min(budget, vocab), dim=-1, sorted=True)
    logs, ids = values.log().tolist(), indices.tolist()
    if adaptive_min_budget is not None:
        if confidence_threshold is None:
            raise ValueError("Adaptive tree budget requires a confidence threshold")
        confidence_depths = min(4, len(logs))
        greedy_log_mean = sum(
            logs[depth][0] for depth in range(confidence_depths)
        ) / confidence_depths
        if greedy_log_mean >= math.log(confidence_threshold):
            budget = adaptive_min_budget
    tokens, parents, depths = [], [-1], [0]
    prefix_log_masses = [0.0]
    # Sibling alternatives are expanded lazily; descendants retain parent mass.
    heap = [(-(logs[0][0] + rewards[0]), 0, 0, 0, 0.0)]
    while heap and len(tokens) < budget:
        neg_score, parent, depth, rank, parent_log = heapq.heappop(heap)
        node = len(parents)
        tokens.append(ids[depth][rank])
        parents.append(parent)
        depths.append(depth + 1)
        prefix_log_masses.append(-float(neg_score))
        if rank + 1 < caps[depth]:
            heapq.heappush(heap, (-(parent_log + logs[depth][rank + 1]
                                    + rewards[depth]),
                                 parent, depth, rank + 1, parent_log))
        if depth + 1 < length:
            heapq.heappush(heap, (neg_score - logs[depth + 1][0]
                                  - rewards[depth + 1],
                                 node, depth + 1, 0, -neg_score))
    return Tree(tokens, parents, depths, [], prefix_log_masses)


def confidence_spine_tree(q: torch.Tensor, budget: int,
                          confidence_threshold: float) -> Tree:
    """Keep a complete greedy spine only on confident DFlash blocks.

    The remaining node budget is filled in canonical DDTree order.  Thus this
    is an ancestry-closed proposal topology with exactly the same verifier and
    sampling law as DDTree; the confidence gate and forced spine only decide
    which prefixes receive Target rows.
    """
    if (q.ndim != 2 or q.shape[0] < 1 or budget < q.shape[0]
            or not 0. < confidence_threshold <= 1.
            or not bool(torch.isfinite(q).all())):
        raise ValueError("Invalid confidence-spine tree controls")
    confidence_depths = min(4, q.shape[0])
    confidence = q[:confidence_depths].amax(-1).double().log().mean().exp()
    if float(confidence) < confidence_threshold:
        return probability_tree(q, budget)

    greedy = q.argmax(-1).tolist()
    tokens = [int(token) for token in greedy]
    parents = [-1] + list(range(len(greedy)))
    depths = list(range(len(greedy) + 1))
    children = {
        (parents[node], tokens[node - 1]): node
        for node in range(1, len(parents))
    }
    canonical = probability_tree(q, budget)
    canonical_paths: list[tuple[int, ...]] = [()]
    for node in range(1, len(canonical.parents)):
        parent = canonical.parents[node]
        path = canonical_paths[parent] + (canonical.tokens[node - 1],)
        canonical_paths.append(path)
        tree_parent = 0
        for depth, token in enumerate(path, 1):
            child = children.get((tree_parent, token))
            if child is None:
                if len(tokens) >= budget:
                    break
                child = len(parents)
                children[(tree_parent, token)] = child
                tokens.append(token)
                parents.append(tree_parent)
                depths.append(depth)
            tree_parent = child
        if len(tokens) >= budget:
            break
    if len(tokens) != budget:
        raise RuntimeError("Confidence-spine tree did not fill its budget")
    return Tree(tokens, parents, depths, [])


def online_rank_calibrated_tree(q: torch.Tensor, budget: int,
                                rank_log_bias: torch.Tensor) -> Tree:
    """Build DDTree after history-only depth/rank calibration.

    ``rank_log_bias[d, r]`` estimates ``log(p/q)`` for Draft rank ``r`` at
    future depth ``d``.  The unobserved vocabulary tail keeps its Draft mass,
    so its normalization remains part of every prefix score.
    """
    if (q.ndim != 2 or rank_log_bias.ndim != 2
            or rank_log_bias.shape[0] != q.shape[0]
            or rank_log_bias.shape[1] < min(budget, q.shape[1])
            or budget < 1 or not bool(torch.isfinite(rank_log_bias).all())):
        raise ValueError("Invalid online rank calibration")
    length, vocab = q.shape
    width = min(budget, vocab)
    values, indices = torch.topk(q, width, dim=-1, sorted=True)
    adjusted = values.double() * rank_log_bias[:, :width].double().exp()
    tail = (1. - values.double().sum(-1)).clamp_min(0.)
    normalizer = (tail + adjusted.sum(-1)).clamp_min(
        torch.finfo(torch.float64).tiny,
    )
    calibrated = adjusted / normalizer[:, None]
    calibrated, order = calibrated.sort(dim=-1, descending=True)
    indices = indices.gather(1, order)
    logs = calibrated.log().tolist()
    ids = indices.tolist()
    tokens, parents, depths = [], [-1], [0]
    heap = [(-logs[0][0], 0, 0, 0, 0.0)]
    while heap and len(tokens) < budget:
        negative, parent, depth, rank, parent_log = heapq.heappop(heap)
        node = len(parents)
        tokens.append(ids[depth][rank])
        parents.append(parent)
        depths.append(depth + 1)
        if rank + 1 < width:
            heapq.heappush(heap, (
                -(parent_log + logs[depth][rank + 1]),
                parent, depth, rank + 1, parent_log,
            ))
        if depth + 1 < length:
            heapq.heappush(heap, (
                negative - logs[depth + 1][0], node, depth + 1,
                0, -negative,
            ))
    return Tree(tokens, parents, depths, [])


def update_online_rank_bias(q: torch.Tensor, tree: Tree,
                            target_probabilities: torch.Tensor,
                            rank_log_bias: torch.Tensor, ewma: float,
                            clip: float) -> torch.Tensor:
    """Update depth/rank ``log(p/q)`` from an already verified tree."""
    if (q.ndim != 2 or target_probabilities.ndim != 2
            or target_probabilities.shape[0] != len(tree.parents)
            or rank_log_bias.ndim != 2
            or rank_log_bias.shape[0] != q.shape[0]
            or not 0 < ewma <= 1 or not math.isfinite(clip) or clip <= 0):
        raise ValueError("Invalid online rank-bias update")
    width = min(rank_log_bias.shape[1], q.shape[1])
    ranked_tokens = q.topk(width, dim=-1, sorted=True).indices
    depth = torch.tensor(
        [value - 1 for value in tree.depths[1:]],
        dtype=torch.long, device=q.device,
    )
    parent = torch.tensor(
        tree.parents[1:], dtype=torch.long, device=q.device,
    )
    token = torch.tensor(tree.tokens, dtype=torch.long, device=q.device)
    if token.numel() == 0:
        return rank_log_bias
    matches = ranked_tokens.index_select(0, depth).eq(token[:, None])
    observed = matches.any(-1)
    if not bool(observed.any()):
        return rank_log_bias
    depth = depth[observed]
    parent = parent[observed]
    token = token[observed]
    rank = matches[observed].to(torch.int64).argmax(-1)
    tiny = torch.finfo(torch.float64).tiny
    ratio = (
        target_probabilities[parent, token].double().clamp_min(tiny).log()
        - q[depth, token].double().clamp_min(tiny).log()
    ).clamp(-clip, clip)
    flat = depth * rank_log_bias.shape[1] + rank
    size = rank_log_bias.numel()
    sums = ratio.new_zeros(size).scatter_add_(0, flat, ratio)
    counts = ratio.new_zeros(size).scatter_add_(
        0, flat, torch.ones_like(ratio),
    )
    mask = counts > 0
    estimate = rank_log_bias.double().flatten().clone()
    estimate[mask] = sums[mask] / counts[mask]
    previous = rank_log_bias.double().flatten()
    updated = torch.where(
        mask, previous.lerp(estimate, float(ewma)), previous,
    )
    return updated.view_as(rank_log_bias).to(rank_log_bias.dtype)


def _finite_path_proposal(q: torch.Tensor,
                          weighted_paths: list[tuple[tuple[int, ...], float]]):
    if not weighted_paths:
        raise FloatingPointError("Finite-tree proposal has no positive path")
    maximum = max(score for _, score in weighted_paths)
    path_mass: dict[tuple[int, ...], float] = {}
    for path, score in weighted_paths:
        if math.isfinite(score):
            path_mass[path] = path_mass.get(path, 0.0) + math.exp(score - maximum)
    if not path_mass:
        raise FloatingPointError("Finite-tree proposal has no positive path")
    support_paths = list(path_mass)
    total = sum(path_mass.values())
    leaf_weights = [path_mass[path] / total for path in support_paths]

    child_mass: dict[tuple[int, ...], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    prefix_mass: dict[tuple[int, ...], float] = defaultdict(float)
    for path, weight in zip(support_paths, leaf_weights):
        prefix: tuple[int, ...] = ()
        for token in path:
            prefix_mass[prefix] += weight
            child_mass[prefix][token] += weight
            prefix += (token,)
    children = {
        prefix: (
            sorted(masses),
            [masses[token] / prefix_mass[prefix] for token in sorted(masses)],
        )
        for prefix, masses in child_mass.items()
    }
    token_probabilities = []
    for path in support_paths:
        prefix: tuple[int, ...] = ()
        row = []
        for token in path:
            tokens, probabilities = children[prefix]
            row.append(probabilities[tokens.index(token)])
            prefix += (token,)
        token_probabilities.append(row)
    return AdaptivePathProposal(
        paths=torch.tensor(support_paths, dtype=torch.long, device=q.device),
        leaf_probabilities=q.new_tensor(leaf_weights),
        token_probabilities=q.new_tensor(token_probabilities),
        children=children,
        vocab_size=q.shape[1],
    )


def adaptive_path_proposal(q: torch.Tensor, leaves: int):
    """Build a normalized proposal on the highest-mass full draft paths.

    DFlash exposes one marginal distribution per future position.  A best-first
    Cartesian-product search finds the most likely complete paths without
    materializing the vocabulary-to-the-power-of-length path space.  The
    returned sparse prefix conditionals make this a genuine autoregressive
    proposal, so it can be used by lossless block verification.
    """
    if q.ndim != 2 or q.shape[0] < 1 or leaves < 1:
        raise ValueError("Expected a nonempty probability matrix and positive leaf budget")
    if not bool(torch.isfinite(q).all() & (q >= 0).all() & (q.sum(-1) > 0).all()):
        raise FloatingPointError("Invalid adaptive-tree draft probabilities")
    length, vocab = q.shape
    width = min(leaves, vocab)
    values, indices = torch.topk(q, width, dim=-1, sorted=True)
    log_values = values.log().tolist()
    token_ids = indices.tolist()

    initial = (0,) * length
    initial_score = sum(row[0] for row in log_values)
    heap = [(-initial_score, initial)]
    seen = {initial}
    ranked: list[tuple[float, tuple[int, ...]]] = []
    while heap and len(ranked) < leaves:
        negative_score, ranks = heapq.heappop(heap)
        score = -negative_score
        if not math.isfinite(score):
            break
        ranked.append((score, ranks))
        for depth in range(length):
            if ranks[depth] + 1 >= width:
                continue
            neighbor = list(ranks)
            neighbor[depth] += 1
            neighbor = tuple(neighbor)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            neighbor_score = score - log_values[depth][ranks[depth]] + log_values[depth][neighbor[depth]]
            heapq.heappush(heap, (-neighbor_score, neighbor))
    if not ranked:
        raise FloatingPointError("Adaptive-tree proposal has no positive path")

    weighted_paths = [
        (tuple(token_ids[d][ranks[d]] for d in range(length)), score)
        for score, ranks in ranked
    ]
    return _finite_path_proposal(q, weighted_paths)


def block_aligned_spine_tree(q: torch.Tensor, budget: int,
                             max_spines: int) -> Tree:
    """Pack highest-mass complete Draft paths into one ancestral tree.

    Unlike DDTree's prefix-mass allocation, every admitted branch is a full
    contiguous block.  Shared prefixes are represented once and candidates
    are admitted only while their union trie fits the Target-row budget.
    Standard ancestral Target verification remains exact for this deterministic
    scaffold; no proposal correction law or additional model call is needed.
    """
    if budget < q.shape[0] or max_spines < 1:
        raise ValueError("Block-aligned trees require one full spine")
    candidates = adaptive_path_proposal(q, max_spines).paths.tolist()
    admitted: list[list[int]] = []
    tree = sampled_tree(torch.tensor(
        [candidates[0]], dtype=torch.long, device=q.device,
    ))
    for path in candidates:
        trial = sampled_tree(torch.tensor(
            admitted + [path], dtype=torch.long, device=q.device,
        ))
        if len(trial.tokens) <= budget:
            admitted.append(path)
            tree = trial
    if not admitted or max(tree.depths) != q.shape[0]:
        raise RuntimeError("Block-aligned tree lost its full-depth spine")
    return tree


def adaptive_prefix_proposal(q: torch.Tensor, budget: int):
    """Complete DDTree-style high-mass prefixes into a finite path proposal.

    Each non-root node in the prefix tree contributes its prefix mass to the
    path obtained by greedily completing the remaining positions.  Duplicate
    completions are merged.  This favors early-token diversity that a ranking
    of complete product paths can miss.
    """
    if q.ndim != 2 or q.shape[0] < 1 or budget < 1:
        raise ValueError("Expected a nonempty probability matrix and positive prefix budget")
    if not bool(torch.isfinite(q).all() & (q >= 0).all() & (q.sum(-1) > 0).all()):
        raise FloatingPointError("Invalid prefix-tree draft probabilities")
    tree = probability_tree(q, budget)
    greedy = q.argmax(-1).tolist()
    weighted_paths = []
    node_prefixes: list[tuple[int, ...]] = [()]
    node_scores = [0.0]
    depths = torch.tensor(tree.depths[1:], dtype=torch.long, device=q.device) - 1
    tokens = torch.tensor(tree.tokens, dtype=torch.long, device=q.device)
    selected_log_q = q[depths, tokens].log().tolist()
    for node in range(1, len(tree.parents)):
        parent = tree.parents[node]
        token = tree.tokens[node - 1]
        depth = tree.depths[node]
        prefix = node_prefixes[parent] + (token,)
        score = node_scores[parent] + selected_log_q[node - 1]
        node_prefixes.append(prefix)
        node_scores.append(score)
        weighted_paths.append((prefix + tuple(greedy[depth:]), score))
    return _finite_path_proposal(q, weighted_paths)


def budgeted_prefix_proposal(q: torch.Tensor, budget: int):
    """Build a prefix-diverse proposal under a hard verified-node budget.

    The older prefix constructor limits source prefixes before completing them
    to full paths, so its final verification trie can exceed ``budget``.  This
    version admits completed paths in deterministic DDTree best-first source
    order only while the union trie remains within the declared cap.  Its leaf
    masses are then normalized on exactly that finite support.
    """
    if q.ndim != 2 or q.shape[0] < 1 or budget < q.shape[0]:
        raise ValueError("Hard-budget prefix proposal requires budget >= length")
    if not bool(torch.isfinite(q).all() & (q >= 0).all()
                & (q.sum(-1) > 0).all()):
        raise FloatingPointError("Invalid hard-budget draft probabilities")

    source = probability_tree(q, budget)
    greedy = q.argmax(-1).tolist()
    node_prefixes: list[tuple[int, ...]] = [()]
    node_scores = [0.0]
    depths = torch.tensor(source.depths[1:], dtype=torch.long, device=q.device) - 1
    tokens = torch.tensor(source.tokens, dtype=torch.long, device=q.device)
    selected_log_q = q[depths, tokens].log().tolist()

    candidates: list[tuple[tuple[int, ...], float]] = []
    for node in range(1, len(source.parents)):
        parent = source.parents[node]
        token = source.tokens[node - 1]
        depth = source.depths[node]
        prefix = node_prefixes[parent] + (token,)
        score = node_scores[parent] + selected_log_q[node - 1]
        node_prefixes.append(prefix)
        node_scores.append(score)
        candidates.append((prefix + tuple(greedy[depth:]), score))

    admitted: list[tuple[tuple[int, ...], float]] = []
    trie_edges: set[tuple[tuple[int, ...], int]] = set()
    for path, score in candidates:
        prefix: tuple[int, ...] = ()
        path_edges: list[tuple[tuple[int, ...], int]] = []
        for token in path:
            edge = (prefix, token)
            path_edges.append(edge)
            prefix += (token,)
        new_edges = [edge for edge in path_edges if edge not in trie_edges]
        if len(trie_edges) + len(new_edges) <= budget:
            admitted.append((path, score))
            trie_edges.update(new_edges)

    proposal = _finite_path_proposal(q, admitted)
    if len(sampled_tree(proposal.paths).tokens) > budget:
        raise AssertionError("Hard-budget proposal exceeded verification cap")
    return proposal


def embedded_prefix_proposal(q: torch.Tensor, tree: Tree, length: int):
    """Put a finite block proposal inside an already allocated DDTree.

    Only prefixes that already reach ``length`` in ``tree`` become proposal
    leaves.  Consequently block verification and correction recycling do not
    consume a single extra Target row: the verified topology remains exactly
    the original B-node DDTree.
    """
    if (q.ndim != 2 or not 1 <= length <= q.shape[0]
            or len(tree.parents) != len(tree.tokens) + 1
            or tree.parents[0] != -1):
        raise ValueError("Invalid embedded-prefix proposal inputs")
    # High-entropy rounds can spend all B nodes before the requested depth.
    # Use the deepest actually verified prefix rather than adding nodes or
    # silently changing the DDTree allocation.
    actual_length = min(length, max(tree.depths))
    prefixes: list[tuple[int, ...]] = [()]
    node_paths: list[list[int]] = [[]]
    log_mass = [0.]
    weighted_paths = []
    path_nodes_by_path: dict[tuple[int, ...], list[int]] = {}
    for node, (parent, token) in enumerate(
            zip(tree.parents[1:], tree.tokens), 1):
        if parent < 0 or parent >= node:
            raise ValueError("Embedded-prefix tree is not topological")
        prefix = prefixes[parent] + (token,)
        nodes = node_paths[parent] + [node]
        depth = len(prefix)
        score = log_mass[parent] + math.log(float(q[depth - 1, token]))
        prefixes.append(prefix)
        node_paths.append(nodes)
        log_mass.append(score)
        if depth == actual_length:
            weighted_paths.append((prefix, score))
            path_nodes_by_path[prefix] = nodes
    if not weighted_paths:
        raise ValueError("DDTree has no prefix at the requested block length")
    proposal = _finite_path_proposal(q[:actual_length], weighted_paths)
    tree.path_nodes = [
        path_nodes_by_path[tuple(path)] for path in proposal.paths.tolist()
    ]
    if any(len(nodes) != actual_length for nodes in tree.path_nodes):
        raise AssertionError("Embedded proposal lost a verified prefix")
    return proposal


def compact_cache(cache, prefix_length: int, rows: list[int], device):
    keep = torch.tensor(rows, device=device) + prefix_length
    if hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    elif hasattr(cache, "key_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache))
    else:
        raise TypeError("Expected a DynamicCache-compatible key/value cache")
    for keys, values in pairs:
        if keys is None:
            continue
        selected_keys = keys.index_select(-2, keep)
        selected_values = values.index_select(-2, keep)
        keys[..., prefix_length:prefix_length + len(rows), :].copy_(selected_keys)
        values[..., prefix_length:prefix_length + len(rows), :].copy_(selected_values)
    cache.crop(prefix_length + len(rows))
    if cache.get_seq_length() != prefix_length + len(rows):
        raise RuntimeError("Cache length invariant failed")
