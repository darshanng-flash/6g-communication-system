import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('resource_allocation_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input and make prediction
def get_user_input_and_predict():
    # Taking user input for the required parameters
    try:
        required_bandwidth = float(input("Enter Required Bandwidth (in Mbps or Kbps, no units): "))
        allocated_bandwidth = float(input("Enter Allocated Bandwidth (in Mbps or Kbps, no units): "))
        signal_strength = float(input("Enter Signal Strength (in dBm, no units): "))
        latency = float(input("Enter Latency (in ms, no units): "))
        
        # Create a DataFrame for the input values
        user_input = np.array([[required_bandwidth, allocated_bandwidth, signal_strength, latency]])
        
        # Scale the input data using the same scaler that was used during training
        scaled_input = scaler.transform(user_input)
        
        # Make the prediction using the trained model
        predicted_resource_allocation = model.predict(scaled_input)
        
        # Display the predicted Resource Allocation
        print(f"Predicted Resource Allocation: {predicted_resource_allocation[0]:.2f}")
    
    except ValueError:
        print("Invalid input. Please enter numeric values only.")
        
# Run the prediction function
get_user_input_and_predict()
