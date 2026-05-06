import json
from pathlib import Path
from datasets import Dataset, DatasetDict

def load_and_prepare_medqa(data_dir: str, max_samples: int = None):
    """
    Memuat dataset MedQA-USMLE dan memformatnya untuk instruksi klinis.
    """
    data_path = Path(data_dir) / "train.jsonl"
    
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if max_samples and len(records) >= max_samples:
                break

    def format_prompt(ex):
        # 1. Ambil pertanyaan dan opsi jawaban
        q = ex.get("question", "")
        opts = ex.get("options", {})
        opts_str = "\n".join([f"({k}) {v}" for k, v in sorted(opts.items())])
        
        # 2. SUNTIKKAN SYSTEM PROMPT (Kunci Alignment Science kamu)
        system_prompt = (
            "You are a clinical assistant. Follow the SOFA-First protocol. "
            "Extract SOFA parameters. If data is missing, use N/P. "
            "End with <|local_ok|> if confident, or <|escalate|> if uncertain. "
            "Answer in \\boxed{} format."
        )
        
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{q}\n\nOptions:\n{opts_str}"}
            ],
            "answer": ex.get("answer_idx") # Indeks jawaban benar (misal: 'A')
        }

    # Ubah list menjadi objek Dataset HuggingFace agar kompatibel dengan Trainer[cite: 3]
    formatted_data = [format_prompt(r) for r in records]
    dataset = Dataset.from_list(formatted_data)
    
    # Bagi menjadi Train dan Validation (90/10)[cite: 3]
    return dataset.train_test_split(test_size=0.1)