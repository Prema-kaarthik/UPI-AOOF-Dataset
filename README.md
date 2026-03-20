# UPI-AOOF-Dataset
AI-based dataset for UPI optimization using forecasting, stochastic optimization, and reinforcement learning to improve SLA, cost, and delay performance in digital payment systems.


#AI #MachineLearning #DeepLearning #ReinforcementLearning #UPI #FinTech 
#Optimization #StochasticOptimization #ARIMA #LSTM #PPO #DataScience


 # AOOF: Adaptive Operational Optimization Framework for UPI

Repository for the paper **"AI-Enhanced Stochastic Optimization for Efficient UPI Banking Operations"**.

## Structure
See folder tree in the query.

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
