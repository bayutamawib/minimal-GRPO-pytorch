import torch
import torch.nn.functional as F

def compute_grpo_advantage(rewards: torch.Tensor):
    """
    Menghitung relative advantage di dalam kelompok (Group).
    Input rewards: Tensor [batch_size, group_size]
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    # Advantage: seberapa jauh sebuah respon lebih baik dibanding rata-rata kelompoknya
    return (rewards - mean) / std

def compute_grpo_loss(
    log_probs: torch.Tensor, 
    ref_log_probs: torch.Tensor, 
    advantages: torch.Tensor, 
    kl_coeff: float = 0.01
):
    """
    Implementasi manual GRPO Loss.
    
    Formula: L = -E[ Advantage * exp(log_prob - old_log_prob) ] + kl_coeff * KL
    Namun karena GRPO adalah on-policy, kita fokus pada ratio gradien dan KL penalty.
    """
    # 1. Policy Gradient Ratio (Importance Sampling ratio)
    # Karena kita baru saja melakukan sampling, ratio awalnya adalah 1.0 (exp(0))
    # Namun kita tetap menghitung log_prob untuk backpropagation.
    ratio = torch.exp(log_probs - log_probs.detach())
    
    # 2. Surrogate Loss
    surrogate_loss = -(ratio * advantages).mean()
    
    # 3. KL Divergence Penalty (menjaga agar model tidak melenceng dari reference model)
    # KL = exp(ref_log_prob - log_prob) - (ref_log_prob - log_prob) - 1
    kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    kl_loss = kl_div.mean()
    
    total_loss = surrogate_loss + kl_coeff * kl_loss
    return total_loss, kl_loss