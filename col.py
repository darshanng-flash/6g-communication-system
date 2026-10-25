# ============================================
# CHECK_COLUMNS_SCRIPT.py - Run this first
# ============================================

import pandas as pd

# Load and check sensor data
print("Checking SENSOR DATA (test_final.csv):")
sensor_data = pd.read_csv('test_final.csv')
print(f"Shape: {sensor_data.shape}")
print(f"Columns: {sensor_data.columns.tolist()}")
print("\nFirst 5 rows:")
print(sensor_data.head())
print("\nData types:")
print(sensor_data.dtypes)

print("\n" + "="*60)
print("Checking WSN DATA (WSN-DS.csv):")
wsn_data = pd.read_csv('WSN-DS.csv')
print(f"Shape: {wsn_data.shape}")
print(f"Columns: {wsn_data.columns.tolist()}")
print("\nFirst 5 rows:")
print(wsn_data.head())
print("\nData types:")
print(wsn_data.dtypes)
print("\nUnique values in 'Attack type' column (if exists):")
if 'Attack type' in wsn_data.columns:
    print(wsn_data['Attack type'].unique())
elif 'Attack_type' in wsn_data.columns:
    print(wsn_data['Attack_type'].unique())
else:
    print("No attack type column found")
