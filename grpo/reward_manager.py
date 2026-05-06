import torch
from typing import List, Dict
from sofa_eval import score_sofa_oracle, reward_correctness, reward_format, reward_process_quality

class RewardManager:
    def __init__(self, weights: Dict[str, float] = None):
        # Menggunakan bobot yang sama dengan riset Gemma-Sync kamu
        self.weights = weights or {
            "correctness": 0.50,
            "sofa": 0.20,
            "format": 0.10,
            "process": 0.20
        }

    def compute_rewards(self, prompts: List[str], completions: List[str], ground_truths: List[str]) -> torch.Tensor:
        """
        Menghitung total reward untuk satu batch (Group).
        Output: Tensor [Batch, Group]
        """
        batch_rewards = []
        
        # Sesuai risetmu, kita hitung 4-tier reward secara deterministik
        # 1. Correctness (RLVR)
        r_correct = reward_correctness(prompts, completions, ground_truths)
        
        # 2. SOFA Oracle + Abstention Bonus (+0.20)
        r_sofa = reward_sofa_oracle(prompts, completions)
        
        # 3. Format Compliance
        r_format = reward_format(prompts, completions)
        
        # 4. Process Quality (Cactus Signal compliance)
        r_process = reward_process_quality(prompts, completions)

        for i in range(len(completions)):
            # Weighted sum untuk mendapatkan satu angka final
            total = (
                self.weights["correctness"] * r_correct[i] +
                self.weights["sofa"] * r_sofa[i] +
                self.weights["format"] * r_format[i] +
                self.weights["process"] * r_process[i]
            )
            batch_rewards.append(total)

        # Ubah list menjadi PyTorch Tensor untuk perhitungan Advantage di loss.py
        return torch.tensor(batch_rewards).view(len(prompts), -1)