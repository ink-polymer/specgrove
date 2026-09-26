"""Sampling verifiers used by the temperature>0 comparison benchmark."""

from __future__ import annotations

import torch


def sampling_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("probabilistic verification requires temperature > 0")
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_probs(probs: torch.Tensor) -> torch.Tensor:
    shape = probs.shape[:-1]
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(shape)


def token_rejection_sample(
    drafts: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Standard token-level speculative rejection sampling."""
    gamma = int(drafts.numel())
    p = target_probs[:gamma].gather(-1, drafts[:, None])[:, 0]
    q = draft_probs.gather(-1, drafts[:, None])[:, 0]
    accepted = int(
        ((torch.rand_like(q) * q) < p).to(torch.int32).cumprod(0).sum().item()
    )
    if accepted == gamma:
        return accepted, sample_probs(target_probs[gamma : gamma + 1])[0]
    residual = (target_probs[accepted] - draft_probs[accepted]).clamp_min(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs[accepted],
    )
    return accepted, sample_probs(residual[None])[0]


def block_rejection_sample(
    drafts: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Joint Block Verification from Sun et al. (2024), Appendix A.

    Returns the longest accepted draft-prefix length and its residual/bonus token.
    """
    gamma = int(drafts.numel())
    vocab = int(target_probs.shape[-1])
    if target_probs.shape != (gamma + 1, vocab):
        raise ValueError("target_probs must have shape [gamma + 1, vocab]")
    if draft_probs.shape != (gamma, vocab):
        raise ValueError("draft_probs must have shape [gamma, vocab]")

    accept_probability = torch.ones((), dtype=torch.float32, device=drafts.device)
    best_length = 0
    bonus: torch.Tensor | None = None
    zero_q = torch.zeros((vocab,), dtype=target_probs.dtype, device=drafts.device)
    for index in range(gamma + 1):
        q_row = draft_probs[index] if index < gamma else zero_q
        residual = (target_probs[index] * accept_probability - q_row).clamp_min(0)
        reject = (1.0 - accept_probability).clamp_min(0).reshape(1)
        weights = torch.cat((residual, reject))
        total = weights.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0:
            chosen = int(sample_probs(target_probs[index : index + 1])[0].item())
        else:
            chosen = int(torch.multinomial(weights / total, 1).item())
        if chosen < vocab:
            best_length = index
            bonus = torch.tensor(chosen, dtype=torch.long, device=drafts.device)

        if index < gamma:
            token = drafts[index]
            p_token = target_probs[index, token]
            q_token = draft_probs[index, token]
            ratio = p_token / q_token.clamp_min(torch.finfo(q_token.dtype).tiny)
            accept_probability = torch.minimum(
                torch.ones_like(accept_probability), accept_probability * ratio
            )

    if bonus is None:
        bonus = sample_probs(target_probs[best_length : best_length + 1])[0]
    return best_length, bonus


def gbv_select_path_and_probs(
    paths: torch.Tensor,
    target_probs_by_path: torch.Tensor,
    draft_probs: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Select a path and compute its skewed proposal distribution for GBV.

    This is Algorithm 1 and Equations (27)--(29) of Thomas & Pal (2026).
    ``paths`` are K iid samples from the length-L product distribution induced by
    DFlash's L parallel slots.  Token id is used only as a deterministic tie-break
    to make every local p/q ordering injective.
    """
    if paths.ndim != 2:
        raise ValueError("paths must have shape [K, L]")
    path_count, length = paths.shape
    vocab = draft_probs.shape[-1]
    if draft_probs.shape != (length, vocab):
        raise ValueError("draft_probs must have shape [L, vocab]")
    if target_probs_by_path.shape != (path_count, length + 1, vocab):
        raise ValueError("target_probs_by_path must have shape [K, L + 1, vocab]")

    tiny = torch.finfo(draft_probs.dtype).tiny
    ratios = []
    for k in range(path_count):
        token_p = target_probs_by_path[k, :length].gather(
            -1, paths[k, :, None]
        )[:, 0]
        token_q = draft_probs.gather(-1, paths[k, :, None])[:, 0]
        ratios.append((token_p / token_q.clamp_min(tiny)).tolist())
    # Python tuple comparison is lexicographic.  Token ids make duplicate-ratio
    # ties deterministic without changing the probability ordering.
    selected = max(
        range(path_count),
        key=lambda k: tuple((ratios[k][i], int(paths[k, i])) for i in range(length)),
    )

    selected_path = paths[selected]
    selected_target = target_probs_by_path[selected]
    q_prefix = torch.ones((), dtype=draft_probs.dtype, device=paths.device)
    q_gamma_prefix = torch.ones_like(q_prefix)
    lower_path_mass = torch.zeros_like(q_prefix)
    q_gamma_rows = []
    for depth in range(length):
        q_row = draft_probs[depth]
        p_row = selected_target[depth]
        ratio_row = p_row / q_row.clamp_min(tiny)
        # Strict ordering by (p/q, token_id), matching the path-selection tie-break.
        # Stable GPU sort preserves token-id order for equal ratios and avoids a
        # prohibitively expensive Python sort over a 150K-token vocabulary.
        order_tensor = torch.argsort(ratio_row, stable=True)
        ordered_q = q_row.index_select(0, order_tensor)
        lower_ordered = torch.cat((torch.zeros_like(ordered_q[:1]), ordered_q.cumsum(0)[:-1]))
        lower = torch.empty_like(q_row)
        lower.scatter_(0, order_tensor, lower_ordered)

        a = lower_path_mass + q_prefix * lower
        joint = (a + q_prefix * q_row).pow(path_count) - a.pow(path_count)
        conditional = joint / q_gamma_prefix.clamp_min(tiny)
        conditional = conditional.clamp_min(0)
        conditional = conditional / conditional.sum().clamp_min(tiny)
        q_gamma_rows.append(conditional)

        token = selected_path[depth]
        lower_path_mass = a[token]
        q_prefix = q_prefix * q_row[token]
        q_gamma_prefix = joint[token]

    return selected, torch.stack(q_gamma_rows)
