import pandas as pd
import numpy as np
import re
import cloudpickle  # safer than joblib for model saving
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('5G.csv')

# Function to remove units from a column
def remove_units(value):
    return re.sub(r'\s*(dBm|ms|Mbps|Kbps|GHz|Hz|kbps|Gbps|%)\s*', '', str(value))

# Clean the columns that need unit removal
columns_to_clean = ['Required_Bandwidth', 'Allocated_Bandwidth', 'Resource_Allocation', 'Signal_Strength', 'Latency']
for col in columns_to_clean:
    df[col] = df[col].apply(remove_units)

# Convert columns to numeric
df[columns_to_clean] = df[columns_to_clean].apply(pd.to_numeric, errors='coerce')

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features and target
X = df[['Required_Bandwidth', 'Allocated_Bandwidth', 'Signal_Strength', 'Latency']]
y = df['Resource_Allocation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save model using cloudpickle
with open('resource_allocation_model.pkl', 'wb') as f:
    cloudpickle.dump(model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    cloudpickle.dump(scaler, f)

# Save metrics (optional)
with open('model_metrics.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R-squared Score: {r2}\n")

# Plot results
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Resource Allocation')
plt.ylabel('Predicted Resource Allocation')
plt.title('Actual vs Predicted Resource Allocation')
plt.grid(True)
plt.tight_layout()
plt.show()
