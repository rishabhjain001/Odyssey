# Feature: odyssey, Property 21: λ-returns match the bootstrapped recursive definition
"""Property-based tests verifying that lambda_returns satisfies its mathematical
specification: shape contract, lam=0 one-step TD, lam=1/γ=1 cumulative sum, and
the general backward recursion.

Property 21: λ-returns match the bootstrapped recursive definition

    - Shape: returns (H, B) given rewards (H,B), discounts (H,B), values (H+1,B).
    - lam=0: G[t] == rewards[t] + discounts[t] * values[t+1]  (one-step TD)
    - lam=1, discounts=1: G[t] == sum(rewards[t:]) + values[H]  (cumulative sum + bootstrap)
    - General recursion: G[H-1] = r[H-1] + γ[H-1]*V[H]
                         G[t]   = r[t] + γ[t]*((1-λ)*V[t+1] + λ*G[t+1])

Validates: Requirements 10.5
"""

from __future__ import annotations

import torch
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from agent.behavior import lambda_returns

# Tolerance for floating-point comparisons
_ATOL = 1e-5


@seed(20240621)
@settings(max_examples=20, deadline=None)
@given(
    H=st.integers(min_value=2, max_value=8),
    B=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_lambda_returns_shape(H: int, B: int, torch_seed: int) -> None:
    """lambda_returns returns a (H, B) tensor given correctly shaped inputs.

    Validates: Requirements 10.5
    """
    torch.manual_seed(torch_seed)

    rewards = torch.randn(H, B)
    discounts = torch.sigmoid(torch.randn(H, B))  # in (0, 1)
    values = torch.randn(H + 1, B)
    lam = float(torch.rand(1).item())

    G = lambda_returns(rewards, discounts, values, lam)

    assert G.shape == (H, B), (
        f"Expected shape ({H}, {B}), got {G.shape}"
    )


@seed(20240621)
@settings(max_examples=20, deadline=None)
@given(
    H=st.integers(min_value=2, max_value=8),
    B=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_lambda_returns_lam0_is_one_step_td(H: int, B: int, torch_seed: int) -> None:
    """With lam=0, G[t] == rewards[t] + discounts[t] * values[t+1] for all t.

    lam=0 reduces lambda_returns to the standard one-step TD target: the return
    at each step is just the immediate reward plus the discounted next value
    (no multi-step bootstrapping).

    Validates: Requirements 10.5
    """
    torch.manual_seed(torch_seed)

    rewards = torch.randn(H, B)
    discounts = torch.sigmoid(torch.randn(H, B))  # in (0, 1)
    values = torch.randn(H + 1, B)

    G = lambda_returns(rewards, discounts, values, lam=0.0)

    # Compute expected one-step TD targets manually:
    # G[t] = rewards[t] + discounts[t] * values[t+1]
    # values has shape (H+1, B); values[1:H+1] gives values[1], ..., values[H]
    expected = rewards + discounts * values[1:H + 1]

    assert torch.allclose(G, expected, atol=_ATOL), (
        f"lam=0 returns differ from one-step TD.\n"
        f"Max abs diff: {(G - expected).abs().max().item()}"
    )


@seed(20240621)
@settings(max_examples=20, deadline=None)
@given(
    H=st.integers(min_value=2, max_value=8),
    B=st.integers(min_value=1, max_value=4),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_lambda_returns_lam1_discounts1_is_cumulative_sum(
    H: int, B: int, torch_seed: int
) -> None:
    """With lam=1 and all discounts=1, G[t] == sum(rewards[t:]) + values[H].

    When λ=1 and γ=1 (no discounting), the λ-return degenerates to the
    undiscounted sum of all future rewards from step t to H-1, plus the terminal
    bootstrap value[H]. This is the Monte Carlo return with no discounting.

    Validates: Requirements 10.5
    """
    torch.manual_seed(torch_seed)

    rewards = torch.randn(H, B)
    discounts = torch.ones(H, B)   # all discounts = 1
    values = torch.randn(H + 1, B)

    G = lambda_returns(rewards, discounts, values, lam=1.0)

    # Build expected: G[t] = rewards[t] + rewards[t+1] + ... + rewards[H-1] + values[H]
    # i.e. cumulative sum from the right
    expected = torch.zeros(H, B)
    running = values[H].clone()
    for t in range(H - 1, -1, -1):
        running = rewards[t] + running
        expected[t] = running

    assert torch.allclose(G, expected, atol=_ATOL), (
        f"lam=1, discounts=1: returns differ from cumulative sum + bootstrap.\n"
        f"Max abs diff: {(G - expected).abs().max().item()}"
    )


@seed(20240621)
@settings(max_examples=20, deadline=None)
@given(
    H=st.integers(min_value=2, max_value=8),
    B=st.integers(min_value=1, max_value=4),
    lam=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    torch_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_lambda_returns_general_recursion(
    H: int, B: int, lam: float, torch_seed: int
) -> None:
    """lambda_returns satisfies the general backward recursion for all (H, B, lam).

    The recursion is:
        G[H-1] = r[H-1] + γ[H-1] * V[H]
        G[t]   = r[t] + γ[t] * ((1-λ)*V[t+1] + λ*G[t+1])

    We verify that the output of lambda_returns matches a reference implementation
    of this recursion computed independently.

    Validates: Requirements 10.5
    """
    torch.manual_seed(torch_seed)

    rewards = torch.randn(H, B)
    discounts = torch.sigmoid(torch.randn(H, B))  # in (0, 1) for realism
    values = torch.randn(H + 1, B)

    G = lambda_returns(rewards, discounts, values, lam=lam)

    # Reference: compute the recursion directly
    ref = torch.zeros(H, B)
    # Terminal step
    ref[H - 1] = rewards[H - 1] + discounts[H - 1] * values[H]
    # Backward pass
    for t in range(H - 2, -1, -1):
        ref[t] = (
            rewards[t]
            + discounts[t] * ((1.0 - lam) * values[t + 1] + lam * ref[t + 1])
        )

    assert torch.allclose(G, ref, atol=_ATOL), (
        f"General recursion mismatch for H={H}, B={B}, lam={lam:.4f}.\n"
        f"Max abs diff: {(G - ref).abs().max().item()}"
    )
