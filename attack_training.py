# ============================================
# OPTIMIZED ATTACK PREDICTION TRAINING SCRIPT
# Fast attack detection for 6G network
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            precision_recall_fscore_support, roc_curve, auc,
                            precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("6G NETWORK ATTACK PREDICTION SYSTEM")
print("OPTIMIZED TRAINING AND EVALUATION")
print("="*60)

# Load dataset
print("\n1. Loading WSN-DS dataset...")
start_time = time.time()
wsn_data = pd.read_csv('WSN-DS.csv')

# Clean column names
wsn_data.columns = wsn_data.columns.str.strip()
print(f"   Dataset shape: {wsn_data.shape}")
print(f"   Time taken: {time.time()-start_time:.2f} seconds")

# Display attack types
print(f"\n   Attack types in dataset:")
attack_counts = wsn_data['Attack type'].value_counts()
for attack, count in attack_counts.items():
    print(f"   • {attack}: {count} samples ({count/len(wsn_data)*100:.1f}%)")

# ============================================
# Feature Engineering for Attack Detection
# ============================================

def engineer_attack_features(data):
    """
    Engineer features specifically for attack detection
    """
    df = data.copy()
    
    # Network behavior features
    df['data_rate'] = df['DATA_S'] / (df['Time'] + 1)
    df['adv_to_join_ratio'] = df['ADV_S'] / (df['JOIN_S'] + 1)
    df['sch_efficiency'] = df['SCH_S'] / (df['SCH_R'] + 1)
    df['communication_overhead'] = (df['ADV_S'] + df['JOIN_S'] + df['SCH_S']) / (df['Time'] + 1)
    
    # Energy-related features
    df['energy_per_data'] = df['Expaned Energy'] / (df['Data_Sent_To_BS'] + 1)
    df['energy_efficiency'] = df['Data_Sent_To_BS'] / (df['Expaned Energy'] + 0.001)
    
    # Distance and topology features
    df['relative_distance'] = df['Dist_To_CH'] / (df['dist_CH_To_BS'] + 1)
    df['network_density'] = 1 / (df['Dist_To_CH'] + 0.001)
    
    # Protocol behavior features
    df['control_packet_ratio'] = (df['ADV_S'] + df['JOIN_S']) / (df['DATA_S'] + 1)
    df['response_latency'] = df['JOIN_R'] / (df['JOIN_S'] + 1)
    
    # Anomaly indicators
    df['packet_drop_rate'] = (df['ADV_S'] - df['ADV_R']) / (df['ADV_S'] + 1)
    df['join_success_rate'] = df['JOIN_R'] / (df['JOIN_S'] + 1)
    
    return df

print("\n2. Engineering features for attack detection...")
start_time = time.time()
wsn_features = engineer_attack_features(wsn_data)
print(f"   New feature set size: {len(wsn_features.columns)} features")
print(f"   Time taken: {time.time()-start_time:.2f} seconds")

# ============================================
# Prepare features for training
# ============================================

print("\n3. Preparing training data...")

# Define feature columns for attack detection
attack_feature_columns = [
    'Is_CH', 'Dist_To_CH', 'ADV_S', 'ADV_R', 'JOIN_S', 'JOIN_R',
    'SCH_S', 'SCH_R', 'Rank', 'DATA_S', 'DATA_R', 'Data_Sent_To_BS',
    'dist_CH_To_BS', 'send_code', 'Expaned Energy',
    'data_rate', 'adv_to_join_ratio', 'sch_efficiency', 'communication_overhead',
    'energy_per_data', 'energy_efficiency', 'relative_distance', 'network_density',
    'control_packet_ratio', 'response_latency', 'packet_drop_rate', 'join_success_rate'
]

# Ensure all features exist
available_features = [col for col in attack_feature_columns if col in wsn_features.columns]
print(f"   Using {len(available_features)} features for attack detection")

# Prepare feature matrix (use sampling for faster training)
sample_size = min(100000, len(wsn_features))  # Use 100k samples for faster training
wsn_sampled = wsn_features.sample(n=sample_size, random_state=42)
X = wsn_sampled[available_features].copy()
y = wsn_sampled['Attack type'].copy()

print(f"   Using {len(X)} samples for training (sampled for efficiency)")

# Handle missing values
X = X.fillna(X.mean())

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder for later use
joblib.dump(label_encoder, 'attack_label_encoder.pkl')
print(f"   Attack classes: {label_encoder.classes_.tolist()}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, 'attack_feature_scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ============================================
# Train optimized models for comparison
# ============================================

print("\n4. Training optimized models for attack detection...")

# Optimized models (removed Gradient Boosting for speed)
models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=50, random_state=42, verbose=False),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42, C=1.0)
}

trained_models = {}
model_performances = {}
training_times = {}

for model_name, model in models.items():
    print(f"\n   Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    training_times[model_name] = training_time
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store performance
    model_performances[model_name] = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'model': model,
        'training_time': training_time
    }
    
    print(f"   ✓ {model_name} - Accuracy: {accuracy:.4f} (Time: {training_time:.2f}s)")

# ============================================
# Select best model
# ============================================

best_model_name = max(model_performances, key=lambda x: model_performances[x]['accuracy'])
best_model = model_performances[best_model_name]['model']
best_accuracy = model_performances[best_model_name]['accuracy']

print(f"\n5. Best model selected: {best_model_name} (Accuracy: {best_accuracy:.4f})")

# Save best model
joblib.dump(best_model, 'attack_prediction_best_model.pkl')
joblib.dump(available_features, 'attack_features_list.pkl')
print("   ✓ Best model saved as 'attack_prediction_best_model.pkl'")

# ============================================
# Generate comprehensive visualizations
# ============================================

print("\n6. Generating visualizations...")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Model Comparison Bar Chart
ax1 = plt.subplot(3, 3, 1)
model_names = list(model_performances.keys())
accuracies = [model_performances[m]['accuracy'] for m in model_names]
colors = ['#2ecc71' if m == best_model_name else '#3498db' for m in model_names]
bars = ax1.bar(model_names, accuracies, color=colors, alpha=0.8)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.tick_params(axis='x', rotation=45)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Training Time Comparison
ax2 = plt.subplot(3, 3, 2)
training_times_list = [model_performances[m]['training_time'] for m in model_names]
colors_time = ['#e74c3c' if t > 10 else '#2ecc71' for t in training_times_list]
bars2 = ax2.bar(model_names, training_times_list, color=colors_time, alpha=0.8)
ax2.set_ylabel('Training Time (seconds)', fontsize=12)
ax2.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
for bar, t in zip(bars2, training_times_list):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')

# 3. Confusion Matrix for Best Model
ax3 = plt.subplot(3, 3, 3)
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
ax3.set_xlabel('Predicted', fontsize=12)
ax3.set_ylabel('Actual', fontsize=12)
ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')

# 4. Attack Type Distribution
ax4 = plt.subplot(3, 3, 4)
attack_dist = wsn_data['Attack type'].value_counts()
colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
wedges, texts, autotexts = ax4.pie(attack_dist.values, labels=attack_dist.index, 
                                     autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax4.set_title('Attack Type Distribution in Dataset', fontsize=14, fontweight='bold')

# 5. Feature Importance (Top 15)
ax5 = plt.subplot(3, 3, 5)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)
    
    ax5.barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
    ax5.set_xlabel('Importance', fontsize=12)
    ax5.set_title('Top 15 Features for Attack Detection', fontsize=14, fontweight='bold')
    ax5.tick_params(axis='y', labelsize=8)

# 6. Cross-Validation Scores
ax6 = plt.subplot(3, 3, 6)
cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=5)
ax6.bar(range(1, 6), cv_scores, color='#2ecc71', alpha=0.7, edgecolor='black')
ax6.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
ax6.set_xlabel('Fold', fontsize=12)
ax6.set_ylabel('Accuracy', fontsize=12)
ax6.set_title('5-Fold Cross-Validation Performance', fontsize=14, fontweight='bold')
ax6.set_ylim([0.7, 1.0])
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Per-Class Accuracy
ax7 = plt.subplot(3, 3, 7)
class_accuracy = {}
for i, class_name in enumerate(label_encoder.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        class_accuracy[class_name] = accuracy_score(y_test[mask], y_pred_best[mask])
    else:
        class_accuracy[class_name] = 0
    
classes = list(class_accuracy.keys())
acc_values = list(class_accuracy.values())
colors_bar = ['#e74c3c' if acc < 0.8 else '#2ecc71' for acc in acc_values]
ax7.bar(classes, acc_values, color=colors_bar, alpha=0.8)
ax7.set_ylabel('Accuracy', fontsize=12)
ax7.set_title('Per-Class Detection Accuracy', fontsize=14, fontweight='bold')
ax7.tick_params(axis='x', rotation=45)
ax7.set_ylim([0, 1])

# 8. Accuracy vs Training Time Trade-off
ax8 = plt.subplot(3, 3, 8)
for model_name in model_names:
    acc = model_performances[model_name]['accuracy']
    time = model_performances[model_name]['training_time']
    color = '#e74c3c' if model_name == best_model_name else '#3498db'
    size = 200 if model_name == best_model_name else 100
    ax8.scatter(time, acc, s=size, c=color, alpha=0.6, label=model_name if model_name != best_model_name else f"{model_name} (Best)")
ax8.set_xlabel('Training Time (seconds)', fontsize=12)
ax8.set_ylabel('Accuracy', fontsize=12)
ax8.set_title('Accuracy vs Training Time Trade-off', fontsize=14, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Model Ranking by Accuracy
ax9 = plt.subplot(3, 3, 9)
model_comparison = []
for name, perf in model_performances.items():
    model_comparison.append({
        'Model': name,
        'Accuracy': perf['accuracy']
    })
df_comparison = pd.DataFrame(model_comparison)
df_comparison_sorted = df_comparison.sort_values('Accuracy', ascending=True)
ax9.barh(df_comparison_sorted['Model'], df_comparison_sorted['Accuracy'], 
         color='#3498db', alpha=0.8)
ax9.set_xlabel('Accuracy', fontsize=12)
ax9.set_title('Model Ranking by Accuracy', fontsize=14, fontweight='bold')
ax9.set_xlim([0, 1])

plt.tight_layout()
plt.savefig('attack_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: attack_prediction_analysis.png")
plt.show()

# ============================================
# Additional Detailed Visualizations
# ============================================

# Figure 2: Detailed Attack Analysis
fig2, axes = plt.subplots(2, 2, figsize=(15, 12))

# 2a: Attack patterns by network parameters
ax = axes[0, 0]
attack_energy = wsn_data.groupby('Attack type')['Expaned Energy'].mean()
attack_energy.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
ax.set_title('Average Energy Consumption by Attack Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Expaned Energy')
ax.tick_params(axis='x', rotation=45)

# 2b: Attack vs Normal communication patterns
ax = axes[0, 1]
attack_data_rate = wsn_data.groupby('Attack type')['DATA_S'].mean()
attack_data_rate.plot(kind='bar', ax=ax, color='#e74c3c', alpha=0.7)
ax.set_title('Data Transmission Rate by Attack Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Data Sent')
ax.tick_params(axis='x', rotation=45)

# 2c: Distance patterns in attacks
ax = axes[1, 0]
attack_distance = wsn_data.groupby('Attack type')['Dist_To_CH'].mean()
attack_distance.plot(kind='bar', ax=ax, color='#2ecc71', alpha=0.7)
ax.set_title('Average Distance to Cluster Head by Attack Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance to CH')
ax.tick_params(axis='x', rotation=45)

# 2d: Packet drop patterns
ax = axes[1, 1]
wsn_data['packet_drop'] = wsn_data['ADV_S'] - wsn_data['ADV_R']
attack_drop = wsn_data.groupby('Attack type')['packet_drop'].mean()
attack_drop.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
ax.set_title('Packet Drop Patterns by Attack Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Packet Drop')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('attack_patterns_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: attack_patterns_analysis.png")
plt.show()

# ============================================
# Save training report
# ============================================

print("\n7. Saving training report...")

report_content = f"""
========================================
6G ATTACK PREDICTION SYSTEM - TRAINING REPORT
========================================

Dataset Information:
- Total samples: {len(wsn_data)}
- Training samples used: {len(X)}
- Features used: {len(available_features)}
- Attack types: {len(label_encoder.classes_)}

Attack Distribution:
{'-'*40}
{attack_counts.to_string()}

Model Performance Summary:
{'-'*40}
Best Model: {best_model_name}
Best Accuracy: {best_accuracy:.4f}

Model Details:
{'-'*40}
"""

for model_name, perf in model_performances.items():
    report_content += f"""
{model_name}:
- Accuracy: {perf['accuracy']:.4f}
- Training Time: {perf['training_time']:.2f} seconds
"""

report_content += f"""
Cross-Validation Performance (Best Model):
- Mean CV Accuracy: {cv_scores.mean():.4f}
- Std CV Accuracy: {cv_scores.std():.4f}
- CV Scores: {cv_scores}

Per-Class Performance (Best Model):
{'-'*40}
{classification_report(y_test, y_pred_best, target_names=label_encoder.classes_)}

Saved Files:
1. attack_prediction_best_model.pkl - Trained best model
2. attack_label_encoder.pkl - Label encoder for attack types
3. attack_feature_scaler.pkl - Feature scaler
4. attack_features_list.pkl - List of features used
5. attack_prediction_analysis.png - Model performance graphs
6. attack_patterns_analysis.png - Attack pattern analysis
7. attack_training_report.txt - This report

========================================
"""

with open('attack_training_report.txt', 'w') as f:
    f.write(report_content)

print("   ✓ Saved: attack_training_report.txt")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
print("="*60)
