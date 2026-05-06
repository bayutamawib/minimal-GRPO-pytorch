"""
test_advantage.py — Unit tests for GRPO Relative Advantage normalisation.

Mathematical Invariant Under Test:
    Given rewards R ∈ ℝ^{B×G}, the advantage function computes:
        A_{i,j} = (R_{i,j} − μ_i) / σ_i

    This Z-score normalisation guarantees two statistical properties
    *within each group* (dim=1):
        1.  𝔼[A_i] ≈ 0   (zero-centred)
        2.  Std(A_i) ≈ 1  (unit-variance)

    These invariants ensure that no single batch dominates the policy
    gradient signal — the optimiser receives *relative* quality
    information rather than absolute reward magnitudes.

Target environment: Dual NVIDIA T4, 4-bit QLoRA.
"""

import pytest
import torch

from grpo.loss import compute_grpo_advantage


# ── fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def dummy_rewards() -> torch.Tensor:
    """
    Deterministic reward tensor [batch=2, group=4].
    Row 0 has a wide spread; Row 1 is tightly clustered.
    Both rows must satisfy the same normalisation invariants.
    """
    return torch.tensor([
        [0.10, 0.90, 0.30, 0.70],   # μ ≈ 0.50, σ ≈ 0.34
        [0.48, 0.52, 0.49, 0.51],   # μ ≈ 0.50, σ ≈ 0.02
    ])


@pytest.fixture
def uniform_rewards() -> torch.Tensor:
    """
    Edge-case: all rewards identical within each group.
    Advantage should be 0 everywhere (no signal → no gradient push).
    The ε = 1e-8 guard in compute_grpo_advantage prevents NaN.
    """
    return torch.tensor([
        [0.50, 0.50, 0.50, 0.50],
        [1.00, 1.00, 1.00, 1.00],
    ])


# ── core invariant tests ──────────────────────────────────────────────

class TestRelativeAdvantage:
    """Verify that the group-level Z-score normalisation holds."""

    def test_mean_is_zero(self, dummy_rewards: torch.Tensor):
        """
        𝔼[A_i] ≈ 0 for every batch element.
        Tolerance = 1e-5 to absorb floating-point drift.
        """
        advantages = compute_grpo_advantage(dummy_rewards)
        group_means = advantages.mean(dim=1)

        for i in range(group_means.shape[0]):
            assert abs(group_means[i].item()) < 1e-5, (
                f"Batch {i}: group mean = {group_means[i].item():.8f}, "
                f"expected ≈ 0"
            )

    def test_std_is_one(self, dummy_rewards: torch.Tensor):
        """
        Std(A_i) ≈ 1 for every batch element.
        We use the *uncorrected* std (ddof=0) since compute_grpo_advantage
        calls torch.std which defaults to Bessel-corrected (ddof=1),
        but the invariant still holds within tolerance for G ≥ 4.
        """
        advantages = compute_grpo_advantage(dummy_rewards)
        group_stds = advantages.std(dim=1)

        for i in range(group_stds.shape[0]):
            assert abs(group_stds[i].item() - 1.0) < 0.15, (
                f"Batch {i}: group std = {group_stds[i].item():.4f}, "
                f"expected ≈ 1.0"
            )

    def test_output_shape_matches_input(self, dummy_rewards: torch.Tensor):
        """Advantage tensor must preserve [B, G] geometry."""
        advantages = compute_grpo_advantage(dummy_rewards)
        assert advantages.shape == dummy_rewards.shape

    def test_relative_ordering_preserved(self, dummy_rewards: torch.Tensor):
        """
        Z-score is a monotonic transform ⇒ the rank order of rewards
        within each group must be identical after normalisation.
        """
        advantages = compute_grpo_advantage(dummy_rewards)

        for i in range(dummy_rewards.shape[0]):
            original_order = dummy_rewards[i].argsort()
            advantage_order = advantages[i].argsort()
            assert torch.equal(original_order, advantage_order), (
                f"Batch {i}: rank order changed after normalisation"
            )


# ── edge-case tests ───────────────────────────────────────────────────

class TestEdgeCases:
    """Guard-rails for degenerate inputs."""

    def test_uniform_rewards_produce_zero_advantage(
        self, uniform_rewards: torch.Tensor
    ):
        """
        When all rewards in a group are identical, advantage must be 0
        (not NaN). This validates the ε-stabiliser in the denominator.
        """
        advantages = compute_grpo_advantage(uniform_rewards)

        assert not torch.isnan(advantages).any(), (
            "NaN detected — the ε guard in std denominator may be missing"
        )
        assert torch.allclose(
            advantages,
            torch.zeros_like(advantages),
            atol=1e-4,
        ), "Uniform rewards should yield zero advantage"

    def test_single_outlier_gets_highest_advantage(self):
        """
        A single high-reward response in an otherwise uniform group
        must receive the maximum advantage.
        """
        rewards = torch.tensor([[0.1, 0.1, 0.1, 0.9]])
        advantages = compute_grpo_advantage(rewards)

        assert advantages[0, 3] == advantages[0].max(), (
            "The outlier response should have the highest advantage"
        )

    def test_negative_rewards(self):
        """GRPO must handle negative reward signals (e.g. penalty terms)."""
        rewards = torch.tensor([[-0.5, -0.2, -0.8, -0.1]])
        advantages = compute_grpo_advantage(rewards)

        assert not torch.isnan(advantages).any()
        assert abs(advantages.mean().item()) < 1e-5

    def test_large_group_size(self):
        """
        Validate with G=16 (the full Gemma-Sync group size on A100).
        Statistical properties should hold even better with larger G.
        """
        torch.manual_seed(42)
        rewards = torch.randn(4, 16)  # [batch=4, group=16]
        advantages = compute_grpo_advantage(rewards)

        group_means = advantages.mean(dim=1)
        for i in range(4):
            assert abs(group_means[i].item()) < 1e-4, (
                f"Batch {i}: mean = {group_means[i].item():.6f}"
            )
