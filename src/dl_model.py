"""
Deep Learning module for modeling temporal spread dynamics using LSTM in PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class SpreadLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.2):
        super(SpreadLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :]) 
        return out

def prepare_dl_sequences(spread, seq_length=20):
    """
    Prepare sequential data for LSTM.
    Predicts the spread value 5 days ahead.
    """
    X, y = [], []
    for i in range(len(spread) - seq_length - 5):
        X.append(spread[i:(i + seq_length)])
        y.append(spread[i + seq_length + 5]) # Target is 5 days ahead
        
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def train_lstm(X_train, y_train, epochs=30, batch_size=32, lr=0.001):
    """
    Train the LSTM model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LSTM on device: {device}")
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SpreadLSTM(input_size=1, hidden_size=32, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}')
            
    return model, device

def predict_lstm(model, X_test, device):
    """
    Predict using trained LSTM.
    """
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    return preds
