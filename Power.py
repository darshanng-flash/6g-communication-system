import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# === Load Dataset ===
df = pd.read_csv("cleaned_5G.csv")

# === Simulate Power_Allocation ===
df['Power_Allocation'] = (
    0.5 * (1 / (df['Signal_Strength'].abs() + 1)) +
    0.3 * df['Latency'] +
    0.2 * df['Required_Bandwidth']
)

# === Encode Categorical Columns ===
le_user = LabelEncoder()
le_app = LabelEncoder()
df['User_ID'] = le_user.fit_transform(df['User_ID'])
df['Application_Type'] = le_app.fit_transform(df['Application_Type'])

# === Feature Selection ===
features = ['User_ID', 'Application_Type', 'Signal_Strength', 'Latency',
            'Required_Bandwidth', 'Allocated_Bandwidth', 'Resource_Allocation']
target = 'Power_Allocation'

X = df[features]
y = df[target]

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Save Model and Preprocessors ===
joblib.dump(model, "power_allocation_model.pkl")
joblib.dump(scaler, "scaler1.pkl")
joblib.dump(le_app, "app_encoder.pkl")
joblib.dump(le_user, "user_encoder.pkl")

print("✅ Model and encoders saved successfully!")
