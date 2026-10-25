# ============================================
# FAST ATTACK PREDICTION TESTING SCRIPT
# User-friendly interface for predictions
# ============================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class AttackPredictionSystem:
    def __init__(self, model_path='attack_prediction_best_model.pkl',
                 encoder_path='attack_label_encoder.pkl',
                 scaler_path='attack_feature_scaler.pkl',
                 features_path='attack_features_list.pkl'):
        
        print("="*60)
        print("6G ATTACK PREDICTION SYSTEM")
        print("LOADING TRAINED MODEL...")
        print("="*60)
        
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.scaler = joblib.load(scaler_path)
            self.features = joblib.load(features_path)
            print("✓ Model loaded successfully!")
            print(f"✓ Attack types: {self.label_encoder.classes_.tolist()}")
            print(f"✓ Features used: {len(self.features)}")
            print(f"✓ Model type: {type(self.model).__name__}")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Please run attack_prediction_training.py first to train the model.")
            raise
    
    def engineer_features_single(self, data_dict):
        """
        Engineer features for a single data point
        """
        # Create DataFrame from input
        df = pd.DataFrame([data_dict])
        
        # Add engineered features
        df['data_rate'] = df['DATA_S'] / (df['Time'] + 1)
        df['adv_to_join_ratio'] = df['ADV_S'] / (df['JOIN_S'] + 1)
        df['sch_efficiency'] = df['SCH_S'] / (df['SCH_R'] + 1)
        df['communication_overhead'] = (df['ADV_S'] + df['JOIN_S'] + df['SCH_S']) / (df['Time'] + 1)
        df['energy_per_data'] = df['Expaned Energy'] / (df['Data_Sent_To_BS'] + 1)
        df['energy_efficiency'] = df['Data_Sent_To_BS'] / (df['Expaned Energy'] + 0.001)
        df['relative_distance'] = df['Dist_To_CH'] / (df['dist_CH_To_BS'] + 1)
        df['network_density'] = 1 / (df['Dist_To_CH'] + 0.001)
        df['control_packet_ratio'] = (df['ADV_S'] + df['JOIN_S']) / (df['DATA_S'] + 1)
        df['response_latency'] = df['JOIN_R'] / (df['JOIN_S'] + 1)
        df['packet_drop_rate'] = (df['ADV_S'] - df['ADV_R']) / (df['ADV_S'] + 1)
        df['join_success_rate'] = df['JOIN_R'] / (df['JOIN_S'] + 1)
        
        # Select required features
        features_df = df[self.features]
        
        return features_df
    
    def predict_attack(self, node_data):
        """
        Predict attack type for a given node
        """
        # Engineer features
        features = self.engineer_features_single(node_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        attack_type = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get probability for predicted class
        confidence = prediction_proba[prediction_encoded]
        
        # Create detailed results
        results = {
            'predicted_attack': attack_type,
            'confidence': confidence,
            'all_probabilities': dict(zip(self.label_encoder.classes_, prediction_proba)),
            'risk_level': self._get_risk_level(attack_type, confidence)
        }
        
        return results
    
    def _get_risk_level(self, attack_type, confidence):
        """
        Determine risk level based on attack type and confidence
        """
        if attack_type == 'Normal':
            return "✅ LOW - Normal Operation"
        elif attack_type == 'Flooding':
            return "⚠️ MEDIUM - Network Flooding Risk"
        elif attack_type == 'TDMA':
            return "⚠️ MEDIUM - TDMA Disruption Risk"
        elif attack_type == 'Grayhole':
            return "🔴 HIGH - Packet Dropping Risk"
        elif attack_type == 'Blackhole':
            return "🔴 CRITICAL - Complete Data Loss Risk"
        else:
            return "❓ UNKNOWN"
    
    def quick_predict(self):
        """
        Quick prediction with minimal inputs
        """
        print("\n" + "="*60)
        print("QUICK PREDICTION MODE")
        print("="*60)
        print("\nEnter the following parameters (press Enter for defaults):")
        
        defaults = {
            'Time': 50,
            'Is_CH': 0,
            'Dist_To_CH': 50.0,
            'ADV_S': 5,
            'ADV_R': 5,
            'JOIN_S': 3,
            'JOIN_R': 3,
            'SCH_S': 2,
            'SCH_R': 2,
            'Rank': 10,
            'DATA_S': 100,
            'DATA_R': 95,
            'Data_Sent_To_BS': 100,
            'dist_CH_To_BS': 100.0,
            'send_code': 2,
            'Expaned Energy': 0.07
        }
        
        node_data = {}
        print("\nBasic Parameters (most important):")
        
        # Only ask for most important parameters
        important_params = ['Dist_To_CH', 'DATA_S', 'Data_Sent_To_BS', 'Expaned Energy', 'ADV_S']
        
        for param in important_params:
            default = defaults.get(param, 0)
            user_input = input(f"{param} [{default}]: ").strip()
            if user_input == "":
                node_data[param] = default
            else:
                try:
                    node_data[param] = float(user_input) if isinstance(default, float) else int(user_input)
                except:
                    node_data[param] = default
        
        # Fill remaining with defaults
        for key, value in defaults.items():
            if key not in node_data:
                node_data[key] = value
        
        # Make prediction
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        
        prediction = self.predict_attack(node_data)
        
        print(f"\n🎯 Predicted Attack Type: {prediction['predicted_attack']}")
        print(f"📊 Confidence: {prediction['confidence']:.2%}")
        print(f"⚠️ Risk Level: {prediction['risk_level']}")
        
        # Show top 3 probabilities
        print(f"\n📈 Top 3 Possible Attacks:")
        sorted_probs = sorted(prediction['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for attack, prob in sorted_probs:
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {attack:12s}: {bar} {prob:.2%}")
        
        return prediction
    
    def batch_test(self, num_samples=20):
        """
        Fast batch testing on dataset
        """
        print("\n" + "="*60)
        print(f"BATCH TESTING ({num_samples} samples)")
        print("="*60)
        
        # Load test data
        df = pd.read_csv('WSN-DS.csv')
        df.columns = df.columns.str.strip()
        
        # Take random samples
        samples = df.sample(n=min(num_samples, len(df)))
        
        results = []
        for idx, row in samples.iterrows():
            node_data = row.to_dict()
            prediction = self.predict_attack(node_data)
            actual = row['Attack type']
            
            results.append({
                'Actual': actual,
                'Predicted': prediction['predicted_attack'],
                'Confidence': f"{prediction['confidence']:.2%}",
                'Correct': '✓' if actual == prediction['predicted_attack'] else '✗'
            })
        
        results_df = pd.DataFrame(results)
        print("\nResults:")
        print(results_df.to_string(index=False))
        
        accuracy = (results_df['Correct'] == '✓').mean()
        print(f"\n✅ Batch Test Accuracy: {accuracy:.2%}")
        
        return results_df

def main():
    """
    Main testing interface
    """
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█    6G NETWORK ATTACK PREDICTION SYSTEM     █")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    try:
        # Initialize system
        system = AttackPredictionSystem()
        
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. ⚡ Quick Prediction (Recommended)")
            print("2. 📊 Batch Test on Dataset")
            print("3. 📈 View Model Performance")
            print("4. 🚪 Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                system.quick_predict()
                
            elif choice == '2':
                num = input("Number of samples to test (default 20): ").strip()
                num = int(num) if num else 20
                system.batch_test(num_samples=num)
                
            elif choice == '3':
                print("\n" + "="*60)
                print("MODEL PERFORMANCE SUMMARY")
                print("="*60)
                try:
                    with open('attack_training_report.txt', 'r') as f:
                        # Show only key information
                        lines = f.readlines()
                        for line in lines[:30]:  # Show first 30 lines
                            print(line, end='')
                        print("\n... (Full report saved in attack_training_report.txt)")
                except:
                    print("Training report not found. Please run training first.")
                    
            elif choice == '4':
                print("\n✅ Thank you for using 6G Attack Prediction System!")
                print("Stay secure! 🛡️")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the model is trained first by running attack_prediction_training.py")

if __name__ == "__main__":
    main()
