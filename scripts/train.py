import torch
from types import SimpleNamespace
from src.grpo.trainer import GRPOTrainer
from data.medqa_loader import load_and_prepare_medqa
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter # Import TensorBoard

def main():
    config = SimpleNamespace(
        model_id="google/gemma-4-e2b-it",
        lr=5e-6,
        group_size=4,
        max_steps=200,
        weights={
            "correctness": 0.50,
            "sofa": 0.20,
            "format": 0.10,
            "process": 0.20
        }
    )

    # 1. Inisialisasi TensorBoard
    # Data grafik akan disimpan di folder "runs/gemma-sync-run"
    writer = SummaryWriter(log_dir="./runs/gemma-sync-run")

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, 
        device_map="cuda:0", 
        torch_dtype=torch.float16
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_id, 
        device_map="cuda:1", 
        torch_dtype=torch.float16
    ).eval() 

    dataset = load_and_prepare_medqa("./medqa_dataset", max_samples=500)
    trainer = GRPOTrainer(model, ref_model, tokenizer, config)

    print("Starting Alignment...")
    for step in range(config.max_steps):
        batch = dataset["train"][step % len(dataset["train"])]
        
        loss, kl, avg_reward = trainer.train_step(batch)

        # 2. Catat metrik ke TensorBoard
        writer.add_scalar("Training/Loss", loss, step)
        writer.add_scalar("Training/KL_Divergence", kl, step)
        writer.add_scalar("Training/Avg_Reward", avg_reward, step)

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss:.4f} | KL: {kl:.4f} | Reward: {avg_reward:.2f}")

    model.save_pretrained("./outputs/gemma-sync-final")
    print("Training Complete. Model Gemma-Sync siap di-deploy.")
    
    # 3. Tutup pencatatan
    writer.close()

if __name__ == "__main__":
    main()