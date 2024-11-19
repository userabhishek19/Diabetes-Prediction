import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle

# Load the dataset
dataset = pd.read_csv('diabetes.csv')

# Preprocess the data
X = dataset.iloc[:, [1, 4, 5, 7, 3, 2, 6, 8]].values  # All 8 features used for prediction
y = dataset.iloc[:, 8].values  # Outcome column (target)

# Scaling the features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Train the model
model = SVC(kernel='linear')
model.fit(X_scaled, y)

# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Data preprocessing complete.")
print("Model training complete.")
print("Model and scaler saved.")
