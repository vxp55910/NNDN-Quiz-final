import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('timeseries.csv')

# Assuming 'High' is the column containing the feature data to scale
# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

# Scale the data to a range (e.g., [0, 1])
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data[['High']])
test_data_scaled = scaler.transform(test_data[['High']])

# Define a function to create input sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 20  # Adjust window size as needed
X_train, y_train = create_sequences(train_data_scaled, window_size)
X_test, y_test = create_sequences(test_data_scaled, window_size)

# Define the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'MAE: {mae}, RMSE: {rmse}')
