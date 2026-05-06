import torch

class PolicyManager:
    def __init__(self, model, tokenizer, group_size: int = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.group_size = group_size

    def generate_group(self, prompt_ids: torch.Tensor, max_len: int = 512):
        """
        Menghasilkan G respon untuk 1 prompt.
        Input: prompt_ids [batch_size, seq_len]
        Output: 
            completions_ids [batch_size * group_size, gen_len]
            log_probs [batch_size * group_size, gen_len]
        """
        # 1. Duplikasi prompt sebanyak Group Size (G)
        # Jika batch=1 dan G=16, maka kita akan memproses 16 baris sekaligus
        expanded_prompt = prompt_ids.repeat_interleave(self.group_size, dim=0)[cite: 6]

        # 2. Rollout / Generation
        # Kita gunakan sampling agar 16 jawaban tersebut bervariasi
        output = self.model.generate(
            expanded_prompt,
            max_new_tokens=max_len,
            do_sample=True,
            temperature=0.7,[cite: 6]
            return_dict_in_generate=True,
            output_scores=True
        )

        # 3. Ekstrak Token & Hitung Log Probs
        # Log Probs ini yang akan kita pakai di loss.py untuk KL Divergence
        gen_tokens = output.sequences[:, prompt_ids.shape[-1]:]
        
        # Sederhananya, kita ambil probabilitas dari setiap token yang dipilih model
        # Ini adalah 'jejak pemikiran' model yang akan kita evaluasi
        return gen_tokens, output.scores