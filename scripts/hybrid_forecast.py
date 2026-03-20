import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def hybrid_forecast():
    train = pd.read_csv("data/processed/train_2021-2023.csv", parse_dates=['Date'])
    test  = pd.read_csv("data/processed/test_2025.csv", parse_dates=['Date'])
    
    series = train['Volume'].values
    
    # ARIMA component
    arima = ARIMA(series, order=(5,1,2)).fit()
    arima_pred = arima.forecast(steps=len(test))
    
    # LSTM component
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1,1))
    
    X, y = [], []
    for i in range(30, len(scaled)):
        X.append(scaled[i-30:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    
    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()
    
    # Simple training loop (in practice use DataLoader + early stopping)
    for epoch in range(50):
        optimizer.zero_grad()
        pred = model(torch.tensor(X, dtype=torch.float32))
        loss = criterion(pred, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    
    # LSTM forecast (simplified rolling)
    lstm_pred = []
    last_seq = scaled[-30:].reshape(1,30,1)
    for _ in range(len(test)):
        with torch.no_grad():
            next_val = model(torch.tensor(last_seq, dtype=torch.float32)).item()
        lstm_pred.append(next_val)
        last_seq = np.append(last_seq[:,1:,:], [[[next_val]]], axis=1)
    
    lstm_pred = scaler.inverse_transform(np.array(lstm_pred).reshape(-1,1)).flatten()
    
    # Hybrid (optimal weight from validation - here 0.6 for illustration)
    hybrid_pred = 0.6 * arima_pred + 0.4 * lstm_pred
    
    rmse = np.sqrt(mean_squared_error(test['Volume'], hybrid_pred))
    print(f"Hybrid RMSE: {rmse:,.0f} transactions/day ({rmse/test['Volume'].mean()*100:.1f}% of mean)")
    
    # Save results
    pd.DataFrame({'Date': test['Date'], 'Actual': test['Volume'], 
                  'ARIMA': arima_pred, 'LSTM': lstm_pred, 'Hybrid': hybrid_pred}).to_csv("results/forecast_results.csv", index=False)
    
    return hybrid_pred

if __name__ == "__main__":
    hybrid_forecast()
