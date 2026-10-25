import joblib
import numpy as np

# Load saved components
model = joblib.load('power_allocation_model.pkl')
scaler = joblib.load('scaler.pkl')
le_app = joblib.load('app_encoder.pkl')
le_user = joblib.load('user_encoder.pkl')

# -------- USER INPUTS --------
user_id_input = input("Enter User ID (e.g., User_1): ")
application_type_input = input("Enter Application Type (e.g., Video_Call): ")
signal_strength = float(input("Enter Signal Strength (e.g., -75): "))
latency = float(input("Enter Latency (e.g., 30): "))
required_bandwidth = float(input("Enter Required Bandwidth (e.g., 10): "))
allocated_bandwidth = float(input("Enter Allocated Bandwidth (e.g., 15): "))
resource_allocation = float(input("Enter Resource Allocation (e.g., 70): "))

# Encode inputs
user_id_encoded = le_user.transform([user_id_input])[0]
app_type_encoded = le_app.transform([application_type_input])[0]

# Create feature vector
input_data = np.array([[user_id_encoded, app_type_encoded, signal_strength,
                        latency, required_bandwidth, allocated_bandwidth, resource_allocation]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict power allocation
predicted_power = model.predict(input_scaled)[0]
print(f"\n🔋 Predicted Power Allocation: {predicted_power:.2f} units")
