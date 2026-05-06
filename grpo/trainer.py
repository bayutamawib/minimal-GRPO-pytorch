import torch
from torch.optim import AdamW
from .policy import PolicyManager
from .reward_manager import RewardManager
from .loss import compute_grpo_advantage, compute_grpo_loss

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, config):
        self.model = model
        self.ref_model = ref_model # Model referensi (statis) untuk KL Penalty
        self.policy = PolicyManager(model, tokenizer, config.group_size)
        self.reward_fn = RewardManager(config.weights)
        self.optimizer = AdamW(model.parameters(), lr=config.lr)[cite: 3]

    def train_step(self, batch):
        # 1. ROLLOUT: Generate G jawaban untuk setiap prompt[cite: 2, 3]
        # 
        prompt_ids = batch["input_ids"]
        gen_tokens, scores = self.policy.generate_group(prompt_ids)

        # 2. EVALUATION: Kasih skor pake SOFA Oracle[cite: 2, 3]
        completions = self.tokenizer.batch_decode(gen_tokens)
        prompts = self.tokenizer.batch_decode(prompt_ids)
        rewards = self.reward_fn.compute_rewards(prompts, completions, batch["answers"])

        # 3. MATH: Hitung Advantage (skor relatif antar jawaban)[cite: 2, 3]
        advantages = compute_grpo_advantage(rewards)

        # 4. KL DIVERGENCE: Bandingkan log_probs model vs model referensi
        # Kita hitung seberapa jauh model melenceng dari 'cara bicara dokter' yang asli
        log_probs = self.get_log_probs(self.model, gen_tokens)
        with torch.no_grad():
            ref_log_probs = self.get_log_probs(self.ref_model, gen_tokens)

        # 5. OPTIMIZE: Hitung Loss dan update bobot
        loss, kl = compute_grpo_loss(log_probs, ref_log_probs, advantages)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()[cite: 3]

        return loss.item(), kl.item(), rewards.mean().item()