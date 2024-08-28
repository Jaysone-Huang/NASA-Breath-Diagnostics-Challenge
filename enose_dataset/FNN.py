import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data_from_directory(directory_path):
    all_features = []
    all_labels = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            
            # Read the file and extract the label
            with open(filepath, 'r') as file:
                lines = file.readlines()
            label_line = lines[1].strip()
            label = 1 if 'POSITIVE' in label_line else 0

            # Load the data, skipping the first three lines
            data = pd.read_csv(filepath, delimiter='\t', skiprows=3)
            
            # Extract features
            X = data.iloc[:, 1:].values
            y = np.array([label] * X.shape[0])
            
            # Append to lists
            all_features.append(X)
            all_labels.append(y)
    
    # Combine all data into a single array
    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_labels)
    
    return X_combined, y_combined

# Load training and testing data
train_directory = 'dataset/train/'
test_directory = 'dataset/test/'

X_train, y_train = load_data_from_directory(train_directory)
X_test, y_test = load_data_from_directory(test_directory)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality Reduction (PCA)
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# # Define a more complex neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model with regularization and adjusted learning rate
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train the model with early stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
