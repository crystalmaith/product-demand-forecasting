import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Check if file exists
file_path = 'C:/Users/Maithili/Desktop/GITHUB/product demand forecasting/Fitness_trackers_updated.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Check the path and filename.")

# Load dataset
df = pd.read_csv(file_path)

# Remove commas and convert to float where necessary
numeric_columns = ['Selling Price', 'Original Price', 'Average Battery Life (in days)', 'Reviews']
for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(",", "", regex=True).astype(float)

# Drop categorical columns
df = df[['Selling Price', 'Original Price', 'Rating (Out of 5)', 'Average Battery Life (in days)', 'Reviews']]

# Handle missing values
df.fillna(df.median(), inplace=True)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Prepare sequences for LSTM
sequence_length = 30  # Using past 30 days to predict next value
X, y = [], []
for i in range(len(df_scaled) - sequence_length):
    X.append(df_scaled[i:i+sequence_length])
    y.append(df_scaled[i+sequence_length, -1])  # Predicting the 'Reviews' column

X, y = np.array(X), np.array(y)

# Split data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
def inverse_transform_column(original_data, scaled_data, scaler, col_index):
    temp = np.mean(original_data, axis=0)  # Use column means instead of zero padding
    temp = np.tile(temp, (len(scaled_data), 1))  # Repeat row-wise
    temp[:, col_index] = scaled_data.flatten()
    return scaler.inverse_transform(temp)[:, col_index]

y_pred_inv = inverse_transform_column(df_scaled, y_pred, scaler, -1)
y_test_inv = inverse_transform_column(df_scaled, y_test.reshape(-1, 1), scaler, -1)

# Check for NaN values in predictions
print(f"y_pred_inv contains NaN: {np.isnan(y_pred_inv).sum()} values")
print(f"y_test_inv contains NaN: {np.isnan(y_test_inv).sum()} values")

# Evaluate model
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual', color='blue')
plt.plot(y_pred_inv, label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted Reviews')
plt.xlabel('Time')
plt.ylabel('Reviews')
plt.show()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Additional graphs
plt.figure(figsize=(12, 6))
plt.hist(y_test_inv - y_pred_inv, bins=30, edgecolor='black')
plt.title('Error Distribution (Actual - Predicted)')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.5)
plt.xlabel('Actual Reviews')
plt.ylabel('Predicted Reviews')
plt.title('Actual vs Predicted Scatter Plot')
plt.show()
