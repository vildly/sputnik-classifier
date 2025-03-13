import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pylo import get_logger

logger = get_logger(__name__)


# ------------------- Summary ------------------- #
# 🔵 Summary
# ✔  Steps Followed:
#
# 1. Generated time series data (sin wave) 📊
# 2. Preprocessed it by scaling and creating sequences 🔄
# 3. Built a simple LSTM model with just two layers 🏗
# 4. Trained it using Mean Squared Error loss 📉
# 5. Made future predictions 🤖
# 6. Visualized actual vs predicted values 🎨


# ------------------- Data Generation ------------------- #
logger.info("📊 Generating sine wave data...")

# Generate 200 data points (sinusoidal function)
data = np.sin(np.linspace(0, 20, 200))

# Reshape to a column vector for scaling
data = data.reshape(-1, 1)

# Visualize the dataset
plt.plot(data, label="Time Series Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.show()

logger.info("✅ Data generation complete. Shape: %s", data.shape)

# ------------------- Data Preprocessing ------------------- #
logger.info("🔄 Normalizing data using MinMaxScaler...")

# Normalize data to the range (0,1)
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)


# Convert to PyTorch tensor
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])  # Input: past 10 values
        y.append(data[i + time_steps])  # Output: next value
    return np.array(X), np.array(y)


# Apply the function to convert data to sequences
TIME_STEPS = 10  # Number of past time steps for prediction

# Generate sequences
X, y = create_sequences(data, TIME_STEPS)

logger.info("✅ Data preprocessed: X shape = %s, y shape = %s", X.shape, y.shape)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Reshape X to 3D format (samples, time_steps, features=1)
X = X.view(X.shape[0], TIME_STEPS, 1)

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ensure X_train, y_train, X_test, y_test are tensors before logging their shapes
X_train = (
    torch.tensor(X_train, dtype=torch.float32) if isinstance(X_train, list) else X_train
)
y_train = (
    torch.tensor(y_train, dtype=torch.float32) if isinstance(y_train, list) else y_train
)
X_test = (
    torch.tensor(X_test, dtype=torch.float32) if isinstance(X_test, list) else X_test
)
y_test = (
    torch.tensor(y_test, dtype=torch.float32) if isinstance(y_test, list) else y_test
)

# Log shapes properly
logger.info(
    "📊 Training set shape: X_train = %s, y_train = %s", X_train.shape, y_train.shape
)
logger.info(
    "📊 Testing set shape:  X_test = %s, y_test = %s", X_test.shape, y_test.shape
)


# ------------------- LSTM Model ------------------- #
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(SimpleLSTM, self).__init__()

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM output
        lstm_out = lstm_out[:, -1, :]  # Take last time step output
        return self.fc(lstm_out)  # Fully connected layer for final prediction


# ------------------- Training the Model ------------------- #
# Initialize the model
model = SimpleLSTM()

# Define loss function (Mean Squared Error for regression)
loss_function = nn.MSELoss()

# Use Adam optimizer (good default choice)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 30  # Number of times the model sees the data

logger.info("🚀 Starting Training for %d epochs...", EPOCHS)

for epoch in range(EPOCHS):
    model.train()  # Set model to training mode

    # Forward pass
    predictions = model(X_train)
    loss = loss_function(predictions, y_train)

    # Backward pass (gradient descent)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        logger.info(f"📉 Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}")

logger.info("✅ Training Complete.")

# ------------------- Evaluation ------------------- #
# Set model to evaluation mode
logger.info("🤖 Generating predictions on test data...")

model.eval()

# Make predictions on test set
with torch.no_grad():
    y_pred = model(X_test)

# Convert predictions back to original scale
y_pred_inv = scaler.inverse_transform(y_pred.numpy())

# Convert y_test to tensor explicitly if needed
if isinstance(y_test, list):
    y_test = torch.tensor(y_test, dtype=torch.float32)

# Convert tensor to numpy
y_test_inv = scaler.inverse_transform(y_test.detach().cpu().numpy().reshape(-1, 1))


plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual Values", marker="o")
plt.plot(y_pred_inv, label="Predicted Values", linestyle="dashed", marker="x")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.show()

logger.info("📊 Finished! Predictions visualized successfully.")
