minimal-grpo-pytorch/
├── src/
│   └── grpo/
│       ├── __init__.py
│       ├── trainer.py       # Inti loop pelatihan (manual PyTorch)
│       ├── loss.py          # Implementasi fungsi loss GRPO & KL Divergence
│       ├── policy.py        # Logic untuk group generation (G=16)
│       └── reward_manager.py # Bridge ke library sofa-eval kamu
├── configs/
│   └── default_config.yaml  # Hyperparams (LR, Group Size, KL Coeff)
├── data/
│   └── medqa_loader.py      # Pipeline data MedQA-USMLE
├── scripts/
│   └── train.py             # Entry point untuk running training
├── tests/
│   ├── test_loss.py         # Unit test untuk memastikan matematika loss benar
│   └── test_advantage.py    # Memastikan normalisasi reward kelompok akurat
├── README.md                # Dokumentasi "Research-Grade"
└── pyproject.toml           # Dependensi & Build system
