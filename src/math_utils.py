from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor | float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE-λ).

    Args:
        rewards    : (T,) reward at each step.
        values     : (T,) V(s_t) from critic for each step.
        dones      : (T,) 1.0 if episode ended at step t, else 0.0.
        next_value : V(s_T) bootstrapped from the critic after the rollout.
        gamma      : discount factor.
        lam        : GAE lambda (λ=1 → Monte-Carlo, λ=0 → 1-step TD).

    Returns:
        advantages : (T,) normalized GAE advantages.
        returns    : (T,) target values = advantages (unnormalized) + values.
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    gae = 0.0

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns
