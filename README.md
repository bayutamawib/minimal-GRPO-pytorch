<p align="center">
  <strong>Gemma-Sync</strong><br>
  <em>Calibrated Abstention & Clinical Deferral via Minimal GRPO</em>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19913606"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19913606.svg" alt="DOI"></a>
  <img src="https://img.shields.io/badge/PyTorch-only-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/precision-4bit%20QLoRA-blueviolet" alt="QLoRA">
  <img src="https://img.shields.io/badge/hardware-2×T4-green" alt="Hardware">
</p>

---

## Overview

**Gemma-Sync** is a minimal, research-grade implementation of **Group Relative Policy Optimization (GRPO)** and **Reinforcement Learning from Verifiable Rewards (RLVR)**, built entirely in raw PyTorch. It is designed to instill *Calibrated Abstention* in clinical Small Language Models (Gemma 4 2B) using the **MedQA-USMLE** benchmark and the **SOFA (Sequential Organ Failure Assessment)** clinical protocol.

> **Paper:** Wibisono, N. B. (2026). *Calibrated Abstention and Clinical Deferral in Small Language Models via RLVR and GRPO.* Zenodo.  
> [https://doi.org/10.5281/zenodo.19913606](https://doi.org/10.5281/zenodo.19913606)

---

## The "No-Magic" Philosophy

Most RLHF/GRPO implementations rely on high-level libraries (TRL, OpenRLHF, DeepSpeed-Chat) that abstract away the gradient computation. This creates two problems for clinical AI:

1. **Sycophancy by abstraction.** When the loss function is a black box, the researcher cannot verify that the model is learning *what* to abstain from versus *how* to please the reward signal.
2. **Gradient impurity.** Library-injected wrappers (e.g., automatic KV-cache management, hidden gradient accumulation) can silently corrupt the policy gradient, producing models that appear aligned but fail under distributional shift.

Gemma-Sync solves this by implementing every component from scratch:

| Component | File | What it does |
|---|---|---|
| **GRPO Loss** | `grpo/loss.py` | Manual surrogate loss + KL penalty (40 lines) |
| **Advantage** | `grpo/loss.py` | Group-level Z-score normalisation |
| **Policy Rollout** | `grpo/policy.py` | Group generation with `G` parallel samples |
| **Reward Manager** | `grpo/reward_manager.py` | 4-tier deterministic reward (bridges `sofa-eval`) |
| **Trainer** | `grpo/trainer.py` | The full GRPO loop: rollout → evaluate → optimise |
| **Data Pipeline** | `data/medqa_loader.py` | MedQA-USMLE → instruction-tuning format |

**Total core engine: ~210 lines of Python.** No hidden state. No magic.

---

## The Cactus Signal — Clinical Routing Protocol

A core contribution of Gemma-Sync is the **Cactus Signal** routing protocol, which teaches the model to *know what it doesn't know*:

```
┌─────────────────────────────────────────────┐
│              Model Inference                │
│                                             │
│  Prompt → SOFA-First Reasoning → Answer     │
│                    │                        │
│         ┌─────────┴──────────┐              │
│         │                    │              │
│    Confident?            Uncertain?         │
│         │                    │              │
│    <|local_ok|>         <|escalate|>        │
│         │                    │              │
│  "Proceed with            "Defer to         │
│   local treatment"        senior physician" │
│                                             │
└─────────────────────────────────────────────┘
```

### Signal Semantics

| Signal | Meaning | Clinical Action |
|---|---|---|
| `<|local_ok|>` | Model confidence exceeds calibration threshold | Safe to act on the recommendation locally |
| `<|escalate|>` | Insufficient data or low confidence | Escalate to a senior clinician; do **not** act on model output |

The Cactus Signal is enforced through the **Process Quality** reward component (`reward_process_quality` in `sofa-eval`), which penalises responses that:
- Provide a definitive answer without sufficient SOFA data → should have escalated
- Escalate when all SOFA parameters are present and unambiguous → over-conservative

---

## Alignment Tax vs. Safety Gain

Our experiments reveal a clear trade-off:

| Metric | Baseline (SFT-only) | Gemma-Sync (GRPO) | Delta |
|---|---|---|---|
| **MedQA Accuracy** | 48.2% | 44.7% | −3.5% |
| **Abstention Rate** | 11.3% | 25.9% | **+129%** |
| **False Confidence** | 34.1% | 12.8% | −62.5% |
| **SOFA Compliance** | 61.0% | 89.4% | +46.6% |

> **The Alignment Tax:** A small drop in raw accuracy (−3.5%) buys a massive reduction in dangerous false confidence (−62.5%) and a +129% increase in appropriate abstention. In clinical AI, this is the correct trade-off — *it is safer to say "I don't know" than to guess.*

---

## GRPO: The Mathematics

### 1. Group Relative Advantage

Given a batch of `B` prompts, we generate `G` responses per prompt and compute rewards `R ∈ ℝ^{B×G}`. The advantage is a per-group Z-score:

$$A_{i,j} = \frac{R_{i,j} - \mu_i}{\sigma_i + \epsilon}$$

where $\mu_i$ and $\sigma_i$ are the mean and standard deviation of rewards *within group $i$*. This eliminates absolute reward scale and focuses the optimiser on *relative quality*.

### 2. Surrogate Loss

$$\mathcal{L}_{pg} = -\mathbb{E}\left[ r(\theta) \cdot A_{i,j} \right]$$

where $r(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$ is the importance-sampling ratio. For on-policy GRPO, $r(\theta) \approx 1$.

### 3. KL Divergence Penalty

$$D_{\text{KL}} \approx \exp(\log \pi_{\text{ref}} - \log \pi_\theta) - (\log \pi_{\text{ref}} - \log \pi_\theta) - 1$$

This single-sample estimator of reverse KL keeps the policy tethered to the reference model, preventing catastrophic drift during alignment.

### 4. Total Objective

$$\mathcal{L} = \mathcal{L}_{pg} + \beta \cdot D_{\text{KL}}$$

where $\beta = 0.01$ (default) controls the strength of the KL constraint.

---

## 4-Tier Reward Architecture

```
Total Reward = 0.50 × Correctness    (RLVR — verifiable answer match)
             + 0.20 × SOFA Oracle    (clinical protocol compliance)
             + 0.10 × Format         (structured output: \boxed{})
             + 0.20 × Process        (Cactus Signal adherence)
```

All rewards are **deterministic and verifiable** — no learned reward model, no human preference data. This is the RLVR paradigm: the reward comes from *ground-truth verification*, not subjective judgment.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/narendrabayutamaw/gemma-sync.git
cd gemma-sync

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Verify the mathematical invariants
pytest tests/ -v
```

### Training

```bash
# Launch on dual T4 GPUs
python scripts/train.py
```

### Target Environment

| Spec | Value |
|---|---|
| **GPU** | 2× NVIDIA T4 (16 GB VRAM each) |
| **Precision** | 4-bit QLoRA (NF4 + double quantisation) |
| **Base Model** | `google/gemma-4-e2b-it` |
| **Group Size** | G=4 (T4-optimised) |
| **Learning Rate** | 5×10⁻⁶ |
| **Max Steps** | 200 |

---

## Repository Structure

```
minimal-grpo-pytorch/
├── grpo/
│   ├── loss.py              # GRPO loss + KL divergence (40 lines)
│   ├── policy.py            # Group generation rollout
│   ├── reward_manager.py    # 4-tier reward bridge to sofa-eval
│   └── trainer.py           # Full GRPO training loop
├── data/
│   └── medqa_loader.py      # MedQA-USMLE data pipeline
├── scripts/
│   └── train.py             # Training entry point
├── tests/
│   ├── test_advantage.py    # Z-score normalisation invariants
│   └── test_loss.py         # Loss + KL + backpropagation tests
├── pyproject.toml           # Dependencies & build configuration
└── README.md                # ← You are here
```

---

## Citation

```bibtex
@article{wibisono2026calibrated,
  title   = {Calibrated Abstention and Clinical Deferral in Small Language
             Models via RLVR and GRPO},
  author  = {Wibisono, Narendra Bayutama},
  year    = {2026},
  doi     = {10.5281/zenodo.19913606},
  url     = {https://doi.org/10.5281/zenodo.19913606},
  journal = {Zenodo}
}
```

---

## License

MIT — See [LICENSE](LICENSE) for details.

<sub>Built with the "No-Magic" philosophy: every gradient is earned, not borrowed.</sub>
