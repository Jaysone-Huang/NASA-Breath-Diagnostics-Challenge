import tensorflow as tf
import pandas as pd
import numpy as np

# Load data from the text file
with open('dataset/train/NTL E-Nose - Patient 1.txt', 'r') as file:
    lines = file.readlines()

# Extract label from line 2
label_line = lines[1].strip()
label = 1 if 'POSITIVE' in label_line else 0

# Load the rest of the data, skipping the first three lines
data = pd.read_csv('dataset/train/NTL E-Nose - Patient 1.txt', delimiter='\t', skiprows=3)

# Extract features and create labels array
X = data.iloc[:, 1:].values  # Features (D1 to D64)
y = np.array([label] * X.shape[0])  # Label array

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the logistic regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
