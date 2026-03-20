
 # AOOF: Adaptive Operational Optimization Framework for UPI

Repository for the paper **"AI-Enhanced Stochastic Optimization for Efficient UPI Banking Operations"**.

## Structure
See folder tree in the query.
UPI-AOOF/
├── data/
│   ├── raw/                      # Raw NPCI UPI CSV datasets
│   ├── processed/                # Preprocessed datasets
│   │   ├── train_2021-2023.csv
│   │   ├── val_2024.csv
│   │   └── test_2025.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb    # Data cleaning and preprocessing
│   ├── 02_hybrid_arima_lstm.ipynb # Hybrid forecasting model
│   └── 03_stochastic_optimization.ipynb # Optimization experiments
│
├── scripts/
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── hybrid_forecast.py        # ARIMA–LSTM hybrid model
│   ├── stochastic_opt.py         # Two-stage stochastic optimization
│   └── rl_simulation.py          # Reinforcement learning (PPO) simulation
│
├── results/
│   └── metrics_summary.txt       # Evaluation metrics and results
│
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
└── LICENSE                       # (Recommended to include)
## How to Run (Reproduce Results)

```bash
pip install -r requirements.txt

# 1. Preprocess (create train/val/test splits)
python scripts/preprocess.py

# 2. Hybrid Forecasting
python scripts/hybrid_forecast.py

# 3. Stochastic Optimization
python scripts/stochastic_opt.py

# 4. RL Simulation (PPO)
python scripts/rl_simulation.py
