## Import Section ##

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf


## Data Download & Normalization ##

df = yf.download("TSLA", period="5y")  # 'period=5y' means go back 5 years to get the data from.
data = df["Close"].values  # ".values" converts it to a numpy array. Pandas Dataframe have column names, index, numbers etc. We just need the raw numbers.
# PyTorch expects numpy arrays as input. They work together cleanly.

scaler = MinMaxScaler()  # creates a scaler object
data = scaler.fit_transform(data.reshape(-1, 1))  # "reshape()" turns flat array [100, 200, 300...] -> into column [[100], [200], [300]...].
# "fit_transform" learns the min and max that squishes everything between 0 and 1.

## Sequence Creation ##


def create_sequences(data, seq_length=60):  # We define 2 fxns data(our normalized TSLA price) & seq_length=60 default length is 60.
    X, y = [], []  # Creates two empty lists. Questions(X) and Answers(y).
    for i in range(len(data) - seq_length):  # Loops through the data, stopping 60 steps before the end.
        X.append(data[i:i+seq_length])  # Grabs 60 days(1 to 60) as a question. You are slicing and slicing includes i and ends before the last one value sliced.
        y.append(data[i+seq_length])  # Grabs 61 as  the answer.
    return np.array(X), np.array(y)  # Returns both as numpy arrays. X will have inner brackets [[],[]]. y will not have.
# Reason why X has inner brackets is each bracket/element consists of a list of values(60 values)


X, y = create_sequences(data)


## train/test split ##

split = int(len(X) * 0.8)  # we are basically doing 1140 * 0.8 = 912 -> meaning 80% of our set for training and 20% for testing.

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

## LSTM Model ##


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)  # nn.Linear(featuresin, featuresout) so 64 features in and then 1 feature out.

    def forward(self, x):
        out, _ = self.lstm(x)  # feed sequence in LSTM -> LSTM returns output & hidden state -> we only need output so we discard hidden state with '_'
        out = self.fc(out[:, -1, :])  # LSTM processes each step and gives output for each one. ':' means all sequence in batch.'-1' last timestep/element only. ':' all 64 hidden features.
        return out

## Training ##


model = LSTMModel()  # Creates an instance model of our LSTM model.
loss_fn = nn.MSELoss()  # We use MSELoss instead of CrossEntropyLoss because here we are using regression aka we are predicting a stock price. Not Classification aka a category.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer - adjusts weights to reduce loss. model.parameters tells adam which weights to adjust.
# lr is learning rate how big our each step is. our step is small 0.001 vs for e.g 0.1 is a big step.


## Training Loop ##

epochs = 100  # We will pass through the training data a 100 times.
for epoch in range(epochs):
    model.train()  # tells PyTorch we are in training mode.
    outputs = model(X_train)  # Forward pass - feed training data through the model & get predictions.
    loss = loss_fn(outputs, y_train)  # Compare predictions vs real prices. Calculate how wrong we are.
    optimizer.zero_grad()  # Clear old gradients from previous epoch. For the next step.
    loss.backward()  # Backpropagation - Calculate new gradients.
    optimizer.step()  # Update weights using gradients.

    if (epoch+1) % 10 == 0:  # epoch starts at 0 so plus 1. It keeps adding one until the no. will give a remainder of zero if divided by 10.
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")  # Print loss every 10 epochs. 'item()' converts PyTorch tensor to a plain Python number(we need raw no.). Shows 6 decimal places.


## Precitions & Plot ##

model.eval()  # turns off training mode. we want testing mode on so we have to turn off training mode obv.
with torch.no_grad():  # tells pytorch to turn off gradient tracking. we need that during training because it backpropagates.
    predicted = model(X_test)

predicted = scaler.inverse_transform(predicted.numpy())  # inverse transform scales it back up to real prices. for e.g will turn 0.43 to $242.50.
y_actual = scaler.inverse_transform(y_test.numpy())  # same here

plt.figure(figsize=(12, 6))
plt.plot(y_actual, label="Real Price")  # line 1
plt.plot(predicted, label="Predicted Price")  # line 2
plt.title("TSLA Stock Price - Real vs Predicted")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
