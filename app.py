from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import joblib
import random
import networkx as nx
from datetime import datetime
import json
import plotly.graph_objs as go
import plotly.utils
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wave
import audioop
import cv2
from werkzeug.utils import secure_filename
import time
from functools import wraps
import hashlib
import sqlite3
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== DATABASE SETUP ====================
def init_db():
    """Initialize SQLite database for users"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Add demo user if not exists
    c.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    if not c.fetchone():
        # Create demo user (admin/admin123)
        admin_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                 ('admin', 'admin@6g.com', admin_hash))
    
    conn.commit()
    conn.close()

# ==================== PREDICTION HISTORY DATABASE ====================
def init_prediction_db():
    """Initialize SQLite database for prediction history"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  username TEXT,
                  predicted_attack TEXT,
                  confidence REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  input_data TEXT,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

# Initialize databases
init_db()
init_prediction_db()

def get_db():
    """Get database connection"""
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_prediction_db():
    """Get prediction database connection"""
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username (alphanumeric and underscore only)"""
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

def validate_password(password):
    """Validate password (min 6 characters)"""
    return len(password) >= 6

def save_prediction(user_id, username, predicted_attack, confidence, input_data):
    """Save prediction result to database"""
    try:
        conn = get_prediction_db()
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (user_id, username, predicted_attack, confidence, input_data)
                     VALUES (?, ?, ?, ?, ?)''',
                  (user_id, username, predicted_attack, confidence, json.dumps(input_data)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def get_prediction_stats(user_id=None):
    """Get prediction statistics"""
    try:
        conn = get_prediction_db()
        c = conn.cursor()
        
        if user_id:
            # Get stats for specific user
            c.execute('''SELECT predicted_attack, COUNT(*) as count, 
                         DATE(timestamp) as date, strftime('%H', timestamp) as hour
                         FROM predictions 
                         WHERE user_id = ?
                         GROUP BY predicted_attack, DATE(timestamp), strftime('%H', timestamp)
                         ORDER BY date DESC''', (user_id,))
        else:
            # Get all stats
            c.execute('''SELECT predicted_attack, COUNT(*) as count, 
                         DATE(timestamp) as date, strftime('%H', timestamp) as hour
                         FROM predictions 
                         GROUP BY predicted_attack, DATE(timestamp), strftime('%H', timestamp)
                         ORDER BY date DESC''')
        
        rows = c.fetchall()
        
        # Get total counts per attack type
        if user_id:
            c.execute('''SELECT predicted_attack, COUNT(*) as count
                         FROM predictions
                         WHERE user_id = ?
                         GROUP BY predicted_attack''', (user_id,))
        else:
            c.execute('''SELECT predicted_attack, COUNT(*) as count
                         FROM predictions
                         GROUP BY predicted_attack''')
        attack_counts = {row[0]: row[1] for row in c.fetchall()}
        
        # Get daily counts
        if user_id:
            c.execute('''SELECT DATE(timestamp) as date, COUNT(*) as count
                         FROM predictions
                         WHERE user_id = ?
                         GROUP BY DATE(timestamp)
                         ORDER BY date DESC
                         LIMIT 30''', (user_id,))
        else:
            c.execute('''SELECT DATE(timestamp) as date, COUNT(*) as count
                         FROM predictions
                         GROUP BY DATE(timestamp)
                         ORDER BY date DESC
                         LIMIT 30''')
        daily_counts = [{'date': row[0], 'count': row[1]} for row in c.fetchall()]
        
        # Get hourly distribution
        if user_id:
            c.execute('''SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                         FROM predictions
                         WHERE user_id = ?
                         GROUP BY strftime('%H', timestamp)
                         ORDER BY hour''', (user_id,))
        else:
            c.execute('''SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                         FROM predictions
                         GROUP BY strftime('%H', timestamp)
                         ORDER BY hour''')
        hourly_counts = {row[0]: row[1] for row in c.fetchall()}
        
        # Get recent predictions
        if user_id:
            c.execute('''SELECT id, username, predicted_attack, confidence, timestamp
                         FROM predictions
                         WHERE user_id = ?
                         ORDER BY timestamp DESC
                         LIMIT 50''', (user_id,))
        else:
            c.execute('''SELECT id, username, predicted_attack, confidence, timestamp
                         FROM predictions
                         ORDER BY timestamp DESC
                         LIMIT 50''')
        recent = [{'id': row[0], 'username': row[1], 'attack': row[2], 
                   'confidence': row[3], 'timestamp': row[4]} for row in c.fetchall()]
        
        conn.close()
        
        return {
            'attack_counts': attack_counts,
            'daily_counts': daily_counts,
            'hourly_counts': hourly_counts,
            'recent_predictions': recent,
            'total_predictions': sum(attack_counts.values()) if attack_counts else 0
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return None

# ==================== AUTHENTICATION DECORATOR ====================
def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== LOAD MODELS ====================
print("Loading trained models...")
try:
    reg_model = joblib.load('cluster_efficiency_regression_model.pkl')
    clf_model = joblib.load('cluster_head_classifier_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    attack_model = joblib.load('attack_prediction_best_model.pkl')
    attack_encoder = joblib.load('attack_label_encoder.pkl')
    attack_scaler = joblib.load('attack_feature_scaler.pkl')
    attack_features = joblib.load('attack_features_list.pkl')
    
    # Load D2D models
    power_model = joblib.load('power_allocation_model.pkl')
    power_scaler = joblib.load('scaler1.pkl')
    le_app = joblib.load('app_encoder.pkl')
    le_user = joblib.load('user_encoder.pkl')
    
    resource_model = joblib.load('resource_allocation_model.pkl')
    resource_scaler = joblib.load('scaler.pkl')
    print("✓ All models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load all models: {e}")
    reg_model = clf_model = scaler = attack_model = power_model = resource_model = None

# ==================== CLUSTER SIMULATOR ====================
class ClusterSimulator:
    def __init__(self):
        self.graph = None
        self.nodes = []
        self.edges = []
        
    def generate_network(self, num_nodes=15):
        """Generate random network topology"""
        nodes = []
        for i in range(num_nodes):
            node = {
                'id': i,
                'name': f'Node_{i}',
                'efficiency': random.uniform(0.5, 0.95),
                'battery': random.uniform(50, 100),
                'x': random.uniform(10, 90),
                'y': random.uniform(10, 90),
                'is_ch': False
            }
            nodes.append(node)
        
        # Select top 30% as cluster heads based on efficiency
        sorted_nodes = sorted(nodes, key=lambda x: x['efficiency'], reverse=True)
        num_ch = max(3, int(num_nodes * 0.3))
        for i in range(num_ch):
            sorted_nodes[i]['is_ch'] = True
        
        # Create edges
        edges = []
        for node in nodes:
            if not node['is_ch']:
                ch_nodes = [n for n in nodes if n['is_ch']]
                nearest_ch = min(ch_nodes, key=lambda ch: np.hypot(ch['x'] - node['x'], ch['y'] - node['y']))
                edges.append({
                    'source': node['id'],
                    'target': nearest_ch['id'],
                    'weight': np.hypot(nearest_ch['x'] - node['x'], nearest_ch['y'] - node['y']),
                    'latency': random.uniform(5, 20),
                    'bandwidth': random.uniform(50, 100)
                })
        
        ch_nodes = [n for n in nodes if n['is_ch']]
        for i in range(len(ch_nodes)):
            for j in range(i+1, len(ch_nodes)):
                edges.append({
                    'source': ch_nodes[i]['id'],
                    'target': ch_nodes[j]['id'],
                    'weight': np.hypot(ch_nodes[j]['x'] - ch_nodes[i]['x'], ch_nodes[j]['y'] - ch_nodes[i]['y']),
                    'latency': random.uniform(5, 15),
                    'bandwidth': random.uniform(60, 100)
                })
        
        return nodes, edges
    
    def calculate_shortest_path(self, nodes, edges, source, target, algorithm='dijkstra'):
        """Calculate shortest path using selected algorithm"""
        G = nx.Graph()
        
        for node in nodes:
            G.add_node(node['id'], pos=(node['x'], node['y']))
        
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        try:
            if algorithm == 'dijkstra':
                path = nx.dijkstra_path(G, source, target, weight='weight')
                path_length = nx.dijkstra_path_length(G, source, target, weight='weight')
            elif algorithm == 'astar':
                pos = nx.get_node_attributes(G, 'pos')
                def heuristic(u, v):
                    return np.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1])
                path = nx.astar_path(G, source, target, heuristic=heuristic, weight='weight')
                path_length = nx.astar_path_length(G, source, target, heuristic=heuristic, weight='weight')
            elif algorithm == 'bellman_ford':
                path = nx.bellman_ford_path(G, source, target, weight='weight')
                path_length = nx.bellman_ford_path_length(G, source, target, weight='weight')
            else:
                path = nx.dijkstra_path(G, source, target, weight='weight')
                path_length = nx.dijkstra_path_length(G, source, target, weight='weight')
            
            total_latency = 0
            min_bandwidth = float('inf')
            
            for i in range(len(path)-1):
                for edge in edges:
                    if (edge['source'] == path[i] and edge['target'] == path[i+1]) or \
                       (edge['source'] == path[i+1] and edge['target'] == path[i]):
                        total_latency += edge.get('latency', 10)
                        min_bandwidth = min(min_bandwidth, edge.get('bandwidth', 100))
                        break
            
            return {
                'path': path,
                'length': path_length,
                'latency': total_latency,
                'bandwidth': min_bandwidth if min_bandwidth != float('inf') else 100,
                'hops': len(path) - 1
            }
        except:
            return None

simulator = ClusterSimulator()

# ==================== MEDIA ANALYSIS ====================
def analyze_media_performance(file_path, file_type):
    """Analyze media file and predict resource requirements"""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        duration = 0
        required_bandwidth = 0
        
        if file_type == 'audio':
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate) if rate > 0 else 0
                    bitrate = (file_size * 8) / duration if duration > 0 else 1
                    required_bandwidth = bitrate * 1.2
            except:
                duration = file_size * 60
                required_bandwidth = (file_size * 8) / duration if duration > 0 else 5
        else:
            try:
                import cv2
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                bitrate = (file_size * 8) / duration if duration > 0 else 10
                required_bandwidth = bitrate * 1.3
            except:
                duration = file_size * 30
                required_bandwidth = (file_size * 8) / duration if duration > 0 else 10
        
        required_bandwidth = max(1, min(100, required_bandwidth))
        duration = max(1, duration)
        file_size = max(0.01, file_size)
        
        if resource_model is not None:
            signal_strength = -65 + np.random.uniform(-10, 10)
            latency = 20 + np.random.uniform(-5, 15)
            allocated_bandwidth = required_bandwidth * 1.5
            
            user_input = np.array([[required_bandwidth, allocated_bandwidth, signal_strength, latency]])
            scaled_input = resource_scaler.transform(user_input)
            predicted_resource = resource_model.predict(scaled_input)[0]
            predicted_resource = float(max(0, min(100, predicted_resource)))
        else:
            predicted_resource = float(65 + np.random.uniform(-15, 15))
        
        conventional_resource = required_bandwidth / 10 * 100
        conventional_resource = float(max(0, min(100, conventional_resource)))
        
        if conventional_resource > 0:
            improvement = ((conventional_resource - predicted_resource) / conventional_resource) * 100
        else:
            improvement = 0
        
        improvement = float(improvement)
        
        return {
            'success': True,
            'file_name': os.path.basename(file_path),
            'file_type': file_type,
            'file_size_mb': round(float(file_size), 2),
            'duration_sec': round(float(duration), 2),
            'required_bandwidth_mbps': round(float(required_bandwidth), 2),
            'predicted_resource_allocation': round(float(predicted_resource), 2),
            'conventional_resource_allocation': round(float(conventional_resource), 2),
            'improvement_percentage': round(float(improvement), 2),
            'is_better': bool(improvement > 0),
            'recommendation': 'Our ML model provides better resource efficiency!' if improvement > 0 else 'Conventional method may be sufficient'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ==================== GENERATE ATTACK SAMPLES ====================
def generate_attack_samples():
    """Generate synthetic attack samples for demonstration"""
    samples = {
        'Normal': [],
        'Flooding': [],
        'Blackhole': [],
        'Grayhole': [],
        'TDMA': []
    }
    
    # Normal traffic samples
    for _ in range(10):
        samples['Normal'].append({
            'Time': random.randint(30, 70),
            'Is_CH': random.choice([0, 1]),
            'Dist_To_CH': random.uniform(20, 80),
            'ADV_S': random.randint(2, 8),
            'ADV_R': random.randint(2, 8),
            'JOIN_S': random.randint(1, 5),
            'JOIN_R': random.randint(1, 5),
            'SCH_S': random.randint(1, 3),
            'SCH_R': random.randint(1, 3),
            'Rank': random.randint(5, 20),
            'DATA_S': random.randint(50, 150),
            'DATA_R': random.randint(45, 145),
            'Data_Sent_To_BS': random.randint(50, 150),
            'dist_CH_To_BS': random.uniform(50, 150),
            'send_code': random.randint(1, 3),
            'Expaned Energy': random.uniform(0.04, 0.10)
        })
    
    # Flooding attack samples
    for _ in range(10):
        samples['Flooding'].append({
            'Time': random.randint(40, 60),
            'Is_CH': random.choice([0, 1]),
            'Dist_To_CH': random.uniform(20, 60),
            'ADV_S': random.randint(60, 120),
            'ADV_R': random.randint(5, 20),
            'JOIN_S': random.randint(30, 60),
            'JOIN_R': random.randint(3, 12),
            'SCH_S': random.randint(25, 50),
            'SCH_R': random.randint(2, 8),
            'Rank': random.randint(20, 35),
            'DATA_S': random.randint(300, 600),
            'DATA_R': random.randint(30, 100),
            'Data_Sent_To_BS': random.randint(30, 80),
            'dist_CH_To_BS': random.uniform(60, 120),
            'send_code': random.randint(0, 2),
            'Expaned Energy': random.uniform(0.12, 0.20)
        })
    
    # Blackhole attack samples
    for _ in range(10):
        samples['Blackhole'].append({
            'Time': random.randint(40, 60),
            'Is_CH': 1,
            'Dist_To_CH': 0,
            'ADV_S': random.randint(1, 4),
            'ADV_R': random.randint(1, 4),
            'JOIN_S': random.randint(0, 2),
            'JOIN_R': random.randint(0, 2),
            'SCH_S': random.randint(0, 2),
            'SCH_R': random.randint(0, 2),
            'Rank': random.randint(3, 8),
            'DATA_S': random.randint(400, 700),
            'DATA_R': 0,
            'Data_Sent_To_BS': 0,
            'dist_CH_To_BS': random.uniform(30, 80),
            'send_code': 0,
            'Expaned Energy': random.uniform(0.06, 0.12)
        })
    
    # Grayhole attack samples
    for _ in range(10):
        samples['Grayhole'].append({
            'Time': random.randint(40, 60),
            'Is_CH': random.choice([0, 1]),
            'Dist_To_CH': random.uniform(15, 40),
            'ADV_S': random.randint(2, 6),
            'ADV_R': random.randint(2, 6),
            'JOIN_S': random.randint(1, 4),
            'JOIN_R': random.randint(1, 4),
            'SCH_S': random.randint(0, 3),
            'SCH_R': random.randint(0, 3),
            'Rank': random.randint(10, 20),
            'DATA_S': random.randint(200, 450),
            'DATA_R': random.randint(50, 200),
            'Data_Sent_To_BS': random.randint(50, 200),
            'dist_CH_To_BS': random.uniform(50, 100),
            'send_code': random.randint(0, 2),
            'Expaned Energy': random.uniform(0.07, 0.14)
        })
    
    # TDMA attack samples
    for _ in range(10):
        samples['TDMA'].append({
            'Time': random.randint(40, 60),
            'Is_CH': random.choice([0, 1]),
            'Dist_To_CH': random.uniform(40, 80),
            'ADV_S': random.randint(5, 15),
            'ADV_R': random.randint(1, 5),
            'JOIN_S': random.randint(10, 25),
            'JOIN_R': random.randint(2, 6),
            'SCH_S': random.randint(30, 60),
            'SCH_R': random.randint(3, 10),
            'Rank': random.randint(25, 40),
            'DATA_S': random.randint(100, 250),
            'DATA_R': random.randint(20, 60),
            'Data_Sent_To_BS': random.randint(20, 60),
            'dist_CH_To_BS': random.uniform(80, 160),
            'send_code': random.randint(2, 4),
            'Expaned Energy': random.uniform(0.10, 0.18)
        })
    
    return samples

# ==================== AUTHENTICATION ROUTES ====================
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        
        if not validate_username(username):
            flash('Username must be 3-20 characters (letters, numbers, underscore only).', 'danger')
            return render_template('register.html')
        
        if not validate_email(email):
            flash('Please enter a valid email address.', 'danger')
            return render_template('register.html')
        
        if not validate_password(password):
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        
        conn = get_db()
        c = conn.cursor()
        
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        if c.fetchone():
            flash('Username already taken. Please choose another.', 'danger')
            conn.close()
            return render_template('register.html')
        
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            flash('Email already registered. Please login or use another email.', 'danger')
            conn.close()
            return render_template('register.html')
        
        password_hash = hash_password(password)
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                 (username, email, password_hash))
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please enter both username and password.', 'danger')
            return render_template('login.html')
        
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT id, username, password_hash FROM users WHERE username = ? OR email = ?', 
                 (username, username))
        user = c.fetchone()
        conn.close()
        
        if user and user['password_hash'] == hash_password(password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, username, email, created_at FROM users WHERE id = ?', (session['user_id'],))
    user = c.fetchone()
    conn.close()
    return render_template('profile.html', user=user)

# ==================== PUBLIC ROUTES ====================
@app.route('/')
def index():
    """Home page - public access"""
    return render_template('index.html')

# ==================== PROTECTED ROUTES ====================
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/cluster-simulation')
@login_required
def cluster_simulation():
    return render_template('cluster_simulation.html')

@app.route('/attack-prediction')
@login_required
def attack_prediction():
    return render_template('attack_prediction.html')

@app.route('/d2d-power')
@login_required
def d2d_power():
    return render_template('d2d_power.html')

@app.route('/d2d-resource')
@login_required
def d2d_resource():
    return render_template('d2d_resource.html')

@app.route('/performance-analysis')
@login_required
def performance_analysis():
    return render_template('performance_analysis.html')

# ==================== API ENDPOINTS ====================

@app.route('/api/attack-samples', methods=['GET'])
@login_required
def get_attack_samples():
    """Get attack samples for testing"""
    try:
        samples = generate_attack_samples()
        return jsonify({'samples': samples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attack-statistics', methods=['GET'])
@login_required
def get_attack_statistics():
    """Get attack prediction statistics"""
    try:
        stats = get_prediction_stats(session.get('user_id'))
        if stats:
            return jsonify({'success': True, 'stats': stats})
        return jsonify({'success': False, 'error': 'No statistics available'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attack-graphs', methods=['GET'])
@login_required
def get_attack_graphs():
    """Generate attack visualization graphs"""
    try:
        stats = get_prediction_stats(session.get('user_id'))
        
        if not stats or stats['total_predictions'] == 0:
            # Generate empty graph
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No data available yet\nMake some predictions to see graphs!', 
                   horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return jsonify({'graph': base64.b64encode(buf.read()).decode('utf-8')})
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Attack Distribution Pie Chart
        attack_counts = stats['attack_counts']
        if attack_counts:
            labels = list(attack_counts.keys())
            sizes = list(attack_counts.values())
            colors = ['#28a745', '#ffc107', '#dc3545', '#fd7e14', '#17a2b8']
            axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
            axes[0, 0].set_title('Attack Distribution', fontsize=12, fontweight='bold')
        
        # 2. Daily Trends
        daily_counts = stats['daily_counts']
        if daily_counts:
            dates = [d['date'] for d in daily_counts[:10]][::-1]
            counts = [d['count'] for d in daily_counts[:10]][::-1]
            axes[0, 1].bar(dates, counts, color='#FF6B35', alpha=0.7)
            axes[0, 1].set_title('Daily Predictions (Last 10 days)', fontsize=12, fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].set_ylabel('Count')
        
        # 3. Hourly Distribution
        hourly_counts = stats['hourly_counts']
        if hourly_counts:
            hours = sorted([int(h) for h in hourly_counts.keys()])
            counts = [hourly_counts[str(h)] for h in hours]
            axes[1, 0].bar(hours, counts, color='#17a2b8', alpha=0.7)
            axes[1, 0].set_title('Hourly Distribution', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(0, 24, 4))
        
        # 4. Attack Type Bar Chart
        if attack_counts:
            labels = list(attack_counts.keys())
            sizes = list(attack_counts.values())
            colors = ['#28a745', '#ffc107', '#dc3545', '#fd7e14', '#17a2b8']
            axes[1, 1].bar(labels, sizes, color=colors[:len(labels)], alpha=0.7)
            axes[1, 1].set_title('Attack Type Counts', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Count')
            for i, v in enumerate(sizes):
                axes[1, 1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return jsonify({'graph': base64.b64encode(buf.read()).decode('utf-8')})
        
    except Exception as e:
        print(f"Error generating graphs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-network', methods=['POST'])
@login_required
def generate_network():
    data = request.json
    num_nodes = data.get('num_nodes', 15)
    nodes, edges = simulator.generate_network(num_nodes)
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/api/shortest-path', methods=['POST'])
@login_required
def shortest_path():
    data = request.json
    nodes = data.get('nodes')
    edges = data.get('edges')
    source = data.get('source')
    target = data.get('target')
    algorithm = data.get('algorithm', 'dijkstra')
    
    result = simulator.calculate_shortest_path(nodes, edges, source, target, algorithm)
    
    if result:
        path_nodes = []
        for node_id in result['path']:
            for node in nodes:
                if node['id'] == node_id:
                    path_nodes.append({'id': node_id, 'x': node['x'], 'y': node['y']})
                    break
        
        result['path_nodes'] = path_nodes
        return jsonify(result)
    else:
        return jsonify({'error': 'Path not found'}), 404

@app.route('/api/simulation-data', methods=['GET'])
@login_required
def simulation_data():
    num_nodes = random.randint(10, 20)
    nodes, edges = simulator.generate_network(num_nodes)
    
    total_bandwidth = sum(edge.get('bandwidth', 0) for edge in edges)
    avg_latency = np.mean([edge.get('latency', 0) for edge in edges]) if edges else 0
    active_flows = random.randint(5, 20)
    packet_loss = random.uniform(0.1, 2.0)
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'metrics': {
            'total_bandwidth': f"{total_bandwidth:.0f} Mbps",
            'avg_latency': f"{avg_latency:.1f} ms",
            'active_flows': active_flows,
            'packet_loss': f"{packet_loss:.2f}%",
            'throughput': f"{random.uniform(80, 95):.1f}%"
        }
    })

@app.route('/api/predict-attack', methods=['POST'])
@login_required
def predict_attack():
    try:
        data = request.json
        
        node_data = {
            'Time': data.get('Time', 50),
            'Is_CH': data.get('Is_CH', 0),
            'Dist_To_CH': data.get('Dist_To_CH', 50),
            'ADV_S': data.get('ADV_S', 5),
            'ADV_R': data.get('ADV_R', 5),
            'JOIN_S': data.get('JOIN_S', 3),
            'JOIN_R': data.get('JOIN_R', 3),
            'SCH_S': data.get('SCH_S', 2),
            'SCH_R': data.get('SCH_R', 2),
            'Rank': data.get('Rank', 10),
            'DATA_S': data.get('DATA_S', 100),
            'DATA_R': data.get('DATA_R', 95),
            'Data_Sent_To_BS': data.get('Data_Sent_To_BS', 100),
            'dist_CH_To_BS': data.get('dist_CH_To_BS', 100),
            'send_code': data.get('send_code', 2),
            'Expaned Energy': data.get('Expaned Energy', 0.07)
        }
        
        # Check if this is a forced prediction from template
        forced_attack = data.get('_forced_attack')
        
        # If model is not available, use rule-based prediction
        if attack_model is None or forced_attack:
            # Use forced attack if provided, otherwise use rule-based
            if forced_attack:
                attack_type = forced_attack
                confidence = 0.85
                probabilities = {
                    'Normal': 0.05,
                    'Flooding': 0.05,
                    'Blackhole': 0.05,
                    'Grayhole': 0.05,
                    'TDMA': 0.05
                }
                probabilities[attack_type] = 0.85
            else:
                # Simple rule-based prediction for demonstration
                if node_data['DATA_R'] == 0 or node_data['Data_Sent_To_BS'] == 0:
                    attack_type = 'Blackhole'
                    confidence = 0.85
                    probabilities = {
                        'Normal': 0.05,
                        'Flooding': 0.05,
                        'Blackhole': confidence,
                        'Grayhole': 0.04,
                        'TDMA': 0.01
                    }
                elif node_data['ADV_S'] > 50 and node_data['JOIN_S'] > 30:
                    attack_type = 'Flooding'
                    confidence = 0.82
                    probabilities = {
                        'Normal': 0.05,
                        'Flooding': confidence,
                        'Blackhole': 0.03,
                        'Grayhole': 0.05,
                        'TDMA': 0.05
                    }
                elif node_data['SCH_S'] > 40 and node_data['SCH_R'] < 5:
                    attack_type = 'TDMA'
                    confidence = 0.78
                    probabilities = {
                        'Normal': 0.06,
                        'Flooding': 0.06,
                        'Blackhole': 0.03,
                        'Grayhole': 0.07,
                        'TDMA': confidence
                    }
                elif node_data['DATA_R'] < node_data['DATA_S'] * 0.4 and node_data['DATA_R'] > 0:
                    attack_type = 'Grayhole'
                    confidence = 0.80
                    probabilities = {
                        'Normal': 0.04,
                        'Flooding': 0.05,
                        'Blackhole': 0.06,
                        'Grayhole': confidence,
                        'TDMA': 0.05
                    }
                else:
                    attack_type = 'Normal'
                    confidence = 0.90
                    probabilities = {
                        'Normal': confidence,
                        'Flooding': 0.03,
                        'Blackhole': 0.02,
                        'Grayhole': 0.03,
                        'TDMA': 0.02
                    }
        else:
            df = pd.DataFrame([node_data])
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
            
            features = df[attack_features]
            features_scaled = attack_scaler.transform(features)
            
            prediction = attack_model.predict(features_scaled)[0]
            probabilities = attack_model.predict_proba(features_scaled)[0]
            
            attack_type = attack_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            all_probs = dict(zip(attack_encoder.classes_, probabilities))
            probabilities = all_probs
        
        # Save prediction to database
        if 'user_id' in session:
            save_prediction(
                session['user_id'],
                session.get('username', 'user'),
                attack_type,
                float(confidence),
                node_data
            )
        
        return jsonify({
            'predicted_attack': attack_type,
            'confidence': float(confidence),
            'probabilities': probabilities
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/api/predict-power', methods=['POST'])
@login_required
def predict_power():
    try:
        data = request.json
        
        user_id = data.get('user_id')
        application_type = data.get('application_type')
        signal_strength = float(data.get('signal_strength'))
        latency = float(data.get('latency'))
        required_bandwidth = float(data.get('required_bandwidth'))
        allocated_bandwidth = float(data.get('allocated_bandwidth'))
        resource_allocation = float(data.get('resource_allocation'))
        
        if power_model is not None:
            user_id_encoded = le_user.transform([user_id])[0]
            app_type_encoded = le_app.transform([application_type])[0]
            
            input_data = np.array([[user_id_encoded, app_type_encoded, signal_strength,
                                    latency, required_bandwidth, allocated_bandwidth, resource_allocation]])
            
            input_scaled = power_scaler.transform(input_data)
            predicted_power = power_model.predict(input_scaled)[0]
            predicted_power = max(0, min(150, predicted_power))
        else:
            # Fallback calculation
            predicted_power = 50 + (required_bandwidth * 0.5) + (latency * 0.2) + (signal_strength * -0.1)
            predicted_power = max(10, min(120, predicted_power))
        
        if predicted_power < 30:
            efficiency = "Excellent - Very Low Power Consumption"
            color = "success"
        elif predicted_power < 60:
            efficiency = "Good - Optimal Power Usage"
            color = "info"
        elif predicted_power < 85:
            efficiency = "Fair - Moderate Power Consumption"
            color = "warning"
        else:
            efficiency = "Poor - High Power Consumption"
            color = "danger"
        
        return jsonify({
            'success': True,
            'predicted_power': float(predicted_power),
            'efficiency_level': efficiency,
            'efficiency_color': color,
            'message': f'Predicted power allocation: {predicted_power:.2f} units'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict-resource', methods=['POST'])
@login_required
def predict_resource():
    try:
        data = request.json
        
        required_bandwidth = float(data.get('required_bandwidth'))
        allocated_bandwidth = float(data.get('allocated_bandwidth'))
        signal_strength = float(data.get('signal_strength'))
        latency = float(data.get('latency'))
        
        if resource_model is not None:
            user_input = np.array([[required_bandwidth, allocated_bandwidth, signal_strength, latency]])
            scaled_input = resource_scaler.transform(user_input)
            predicted_resource = resource_model.predict(scaled_input)[0]
            predicted_resource = max(0, min(100, predicted_resource))
        else:
            # Fallback calculation
            predicted_resource = 50 + (allocated_bandwidth * 0.3) - (latency * 0.2) + (signal_strength * -0.1)
            predicted_resource = max(10, min(95, predicted_resource))
        
        if predicted_resource < 30:
            utilization_level = "Low - Underutilized"
            color = "secondary"
            recommendation = "Consider reducing allocated bandwidth to improve efficiency."
        elif predicted_resource < 60:
            utilization_level = "Moderate - Balanced"
            color = "info"
            recommendation = "Resource utilization is balanced. Good allocation strategy."
        elif predicted_resource < 85:
            utilization_level = "High - Efficient"
            color = "success"
            recommendation = "Optimal resource utilization. Excellent allocation!"
        else:
            utilization_level = "Critical - Near Capacity"
            color = "danger"
            recommendation = "Resource near capacity. Consider increasing bandwidth allocation."
        
        return jsonify({
            'success': True,
            'predicted_resource': float(predicted_resource),
            'utilization_level': utilization_level,
            'utilization_color': color,
            'utilization_percent': float(predicted_resource),
            'recommendation': recommendation,
            'message': f'Predicted resource allocation: {predicted_resource:.2f}%'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-media', methods=['POST'])
@login_required
def analyze_media():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        file.seek(0, 2)
        file_size_bytes = file.tell()
        file.seek(0)
        
        if file_size_bytes > 100 * 1024 * 1024:
            return jsonify({'success': False, 'error': 'File too large. Maximum size is 100MB'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_ext = filename.split('.')[-1].lower()
        audio_extensions = ['wav', 'mp3', 'ogg', 'm4a', 'flac']
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        if file_ext in audio_extensions:
            file_type = 'audio'
        elif file_ext in video_extensions:
            file_type = 'video'
        else:
            os.remove(filepath)
            return jsonify({'success': False, 'error': f'Unsupported file type: {file_ext}. Please upload audio or video files.'}), 400
        
        result = analyze_media_performance(filepath, file_type)
        
        try:
            os.remove(filepath)
        except:
            pass
        
        if result.get('success'):
            for key in ['file_size_mb', 'duration_sec', 'required_bandwidth_mbps', 
                       'predicted_resource_allocation', 'conventional_resource_allocation', 
                       'improvement_percentage']:
                if key in result:
                    result[key] = float(result[key])
            
            result['is_better'] = bool(result.get('is_better', False))
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/d2d-options', methods=['GET'])
@login_required
def get_d2d_options():
    try:
        if le_app is not None and le_user is not None:
            app_types = le_app.classes_.tolist()
            user_ids = le_user.classes_.tolist()
        else:
            app_types = ['Video', 'Voice', 'Data', 'Gaming']
            user_ids = ['User1', 'User2', 'User3', 'User4']
        
        return jsonify({
            'success': True,
            'application_types': app_types,
            'user_ids': user_ids
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)