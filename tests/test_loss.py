"""
test_loss.py — Unit tests for GRPO Loss and KL Divergence penalty.

Mathematical Background:
    The GRPO objective consists of two terms:

    1.  Surrogate Policy Gradient:
            L_pg = −𝔼[ r(θ) · A_i ]
        where r(θ) = exp(log π_θ − log π_θ_old) is the importance-
        sampling ratio (≈ 1.0 for on-policy GRPO).

    2.  KL Divergence Penalty (reverse KL approximation):
            KL ≈ exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1
        This is the *unbiased single-sample* estimator of
        D_KL(π_θ ‖ π_ref) derived from the identity:
            KL = 𝔼_{π_θ}[ log(π_θ/π_ref) ]

    Combined:
            L_total = L_pg + β · KL

    Key invariant: when π_θ = π_ref, the KL term is *exactly* zero,
    providing no penalty for an un-drifted policy.

Target environment: Dual NVIDIA T4, 4-bit QLoRA.
"""

import pytest
import torch
import torch.nn as nn

from grpo.loss import compute_grpo_loss, compute_grpo_advantage


# ── fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def identical_log_probs() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates π_θ = π_ref (no drift from the reference model).
    Both tensors share the same values ⇒ KL must be exactly 0.
    """
    log_probs = torch.log(torch.tensor([0.3, 0.5, 0.2, 0.8]))
    ref_log_probs = log_probs.clone()
    return log_probs, ref_log_probs


@pytest.fixture
def divergent_log_probs() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates a policy that has drifted significantly from the
    reference. KL should be strictly positive.
    """
    log_probs = torch.log(torch.tensor([0.9, 0.1, 0.8, 0.2]))
    ref_log_probs = torch.log(torch.tensor([0.1, 0.9, 0.2, 0.8]))
    return log_probs, ref_log_probs


@pytest.fixture
def unit_advantages() -> torch.Tensor:
    """Simple advantage vector with known values."""
    return torch.tensor([1.0, -1.0, 0.5, -0.5])


# ── KL divergence tests ──────────────────────────────────────────────

class TestKLDivergence:
    """Verify the KL penalty estimator behaves correctly."""

    def test_kl_is_zero_when_policies_identical(
        self, identical_log_probs, unit_advantages
    ):
        """
        Core invariant: D_KL(π_θ ‖ π_ref) = 0 when π_θ ≡ π_ref.
        This is the foundational sanity check — if this fails,
        the entire alignment loop is mathematically broken.
        """
        log_probs, ref_log_probs = identical_log_probs
        _, kl_loss = compute_grpo_loss(
            log_probs, ref_log_probs, unit_advantages
        )
        assert abs(kl_loss.item()) < 1e-6, (
            f"KL should be 0 when policies are identical, got {kl_loss.item()}"
        )

    def test_kl_is_positive_when_policies_diverge(
        self, divergent_log_probs, unit_advantages
    ):
        """
        D_KL ≥ 0 always (Gibbs' inequality).
        When π_θ ≠ π_ref, the penalty must be strictly positive.
        """
        log_probs, ref_log_probs = divergent_log_probs
        _, kl_loss = compute_grpo_loss(
            log_probs, ref_log_probs, unit_advantages
        )
        assert kl_loss.item() > 0, (
            f"KL should be > 0 for divergent policies, got {kl_loss.item()}"
        )

    def test_kl_increases_with_greater_divergence(self, unit_advantages):
        """
        Monotonicity: larger distributional gaps ⇒ larger KL.
        """
        base = torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        mild_drift = torch.log(torch.tensor([0.4, 0.6, 0.4, 0.6]))
        heavy_drift = torch.log(torch.tensor([0.1, 0.9, 0.1, 0.9]))

        _, kl_mild = compute_grpo_loss(base, mild_drift, unit_advantages)
        _, kl_heavy = compute_grpo_loss(base, heavy_drift, unit_advantages)

        assert kl_heavy.item() > kl_mild.item(), (
            f"Heavier drift should yield larger KL: "
            f"mild={kl_mild.item():.4f}, heavy={kl_heavy.item():.4f}"
        )


# ── loss computation tests ────────────────────────────────────────────

class TestGRPOLoss:
    """Verify the combined GRPO loss (surrogate + KL)."""

    def test_loss_returns_two_tensors(
        self, identical_log_probs, unit_advantages
    ):
        """compute_grpo_loss must return (total_loss, kl_loss)."""
        log_probs, ref_log_probs = identical_log_probs
        result = compute_grpo_loss(log_probs, ref_log_probs, unit_advantages)
        assert isinstance(result, tuple) and len(result) == 2

    def test_loss_is_scalar(self, identical_log_probs, unit_advantages):
        """Both output tensors must be 0-dimensional scalars."""
        log_probs, ref_log_probs = identical_log_probs
        total_loss, kl_loss = compute_grpo_loss(
            log_probs, ref_log_probs, unit_advantages
        )
        assert total_loss.dim() == 0, "total_loss must be a scalar"
        assert kl_loss.dim() == 0, "kl_loss must be a scalar"

    def test_kl_coeff_scales_penalty(self, divergent_log_probs, unit_advantages):
        """
        Increasing β should increase total_loss when KL > 0,
        because L_total = L_pg + β · KL.
        """
        log_probs, ref_log_probs = divergent_log_probs

        loss_low, _ = compute_grpo_loss(
            log_probs, ref_log_probs, unit_advantages, kl_coeff=0.001
        )
        loss_high, _ = compute_grpo_loss(
            log_probs, ref_log_probs, unit_advantages, kl_coeff=1.0
        )
        assert loss_high.item() > loss_low.item(), (
            "Higher kl_coeff should produce a larger total loss"
        )


# ── gradient flow tests ──────────────────────────────────────────────

class TestBackpropagation:
    """
    Ensure gradients flow correctly through the GRPO loss.
    This is the 'No-Magic' verification: we confirm that our manual
    PyTorch implementation produces valid gradients without relying
    on any high-level RL library.
    """

    def test_loss_backpropagates_through_linear(self):
        """
        End-to-end gradient check with a minimal trainable layer.

        Architecture:
            Linear(4 → 4) → log_softmax → compute_grpo_loss → backward

        After .backward(), every parameter in the linear layer must
        have a non-None, non-zero gradient tensor.
        """
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        x = torch.randn(4, 4)

        # Forward pass: produce log-probabilities
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1).sum(dim=-1)

        # Frozen reference (detached from the computation graph)
        ref_log_probs = log_probs.detach()

        # Synthetic advantages (pre-computed, no gradient needed)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

        # Compute loss and backpropagate
        loss, kl = compute_grpo_loss(log_probs, ref_log_probs, advantages)
        loss.backward()

        # Verify gradient existence and non-triviality
        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient — backward graph is broken"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"Parameter '{name}' has all-zero gradients — "
                f"no learning signal is reaching this layer"
            )

    def test_advantage_does_not_require_grad(self):
        """
        Advantages are computed from rewards (external signal) and must
        NOT carry gradients. This prevents reward-hacking through the
        advantage term.
        """
        rewards = torch.tensor([[0.1, 0.9, 0.3, 0.7]])
        advantages = compute_grpo_advantage(rewards)
        assert not advantages.requires_grad, (
            "Advantages must be detached from the computation graph"
        )

    def test_gradient_magnitude_is_reasonable(self):
        """
        Smoke test: gradients should not explode or vanish for
        well-conditioned inputs. This catches numerical instability
        before it surfaces as NaN loss during training.
        """
        torch.manual_seed(0)
        model = nn.Linear(8, 8)
        x = torch.randn(4, 8)

        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1).sum(dim=-1)
        ref_log_probs = log_probs.detach()
        advantages = torch.tensor([0.5, -0.5, 0.3, -0.3])

        loss, _ = compute_grpo_loss(log_probs, ref_log_probs, advantages)
        loss.backward()

        for name, param in model.named_parameters():
            grad_norm = param.grad.norm().item()
            assert grad_norm < 100.0, (
                f"Gradient explosion in '{name}': norm = {grad_norm:.2f}"
            )
            assert grad_norm > 1e-10, (
                f"Gradient vanishing in '{name}': norm = {grad_norm:.2e}"
            )
