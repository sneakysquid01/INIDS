"""
Real-Time Intrusion Detection Simulator
Simulates live detection by processing packets with temporal delay
"""
import pandas as pd
import joblib
import os
import time
import logging
from datetime import datetime
try:
    from src.schema import COLUMNS, LABEL_COLUMNS
except ImportError:
    from schema import COLUMNS, LABEL_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

class RealtimeDetector:
    def __init__(self, model_path=None):
        """Initialize real-time detector with trained model"""
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "rf_nsl_kdd.pkl")
        
        logging.info(f"Loading model: {model_path}")
        self.model = joblib.load(model_path)
        self.alerts = []
        self.processed_count = 0
        self.attack_count = 0
    
    def detect(self, packet_features):
        """Analyze single packet and return prediction"""
        try:
            # Convert Series to DataFrame with single row
            if isinstance(packet_features, pd.Series):
                packet_features = packet_features.to_frame().T
            
            prediction = self.model.predict(packet_features)[0]
            confidence = max(self.model.predict_proba(packet_features)[0]) * 100
            
            self.processed_count += 1
            if prediction == 1:
                self.attack_count += 1
            
            return {
                'prediction': 'Attack' if prediction == 1 else 'Normal',
                'confidence': round(confidence, 2),
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
        except Exception as e:
            logging.error(f"Detection error: {e}")
            return None
    
    def simulate_realtime(self, data_file=TEST_FILE, num_packets=100, delay=0.3):
        """
        Simulate real-time detection on dataset
        Args:
            data_file: Path to network data CSV/TXT
            num_packets: Number of packets to process
            delay: Delay between packets (seconds) for realistic simulation
        """
        logging.info("=" * 70)
        logging.info("🚀 INIDS REAL-TIME DETECTION SIMULATION")
        logging.info("=" * 70)
        
        # Load data
        try:
            df = pd.read_csv(data_file, names=COLUMNS)
            df = df.head(num_packets)
            X = df.drop(columns=LABEL_COLUMNS)
            y_true = df['label']
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            return
        
        logging.info(f"📊 Processing {len(df)} packets with {delay}s delay per packet\n")
        time.sleep(1)
        
        # Process packets with delay
        for idx in range(len(X)):
            time.sleep(delay)
            
            packet = X.iloc[idx]
            true_label = y_true.iloc[idx]
            
            result = self.detect(packet)
            if not result:
                continue
            
            # Format output
            timestamp = result['timestamp']
            prediction = result['prediction']
            confidence = result['confidence']
            
            if prediction == 'Attack':
                alert = f"[{timestamp}] 🚨 ATTACK DETECTED | Packet #{idx} | Confidence: {confidence}% | True: {true_label}"
                print(alert)
                self.alerts.append({
                    'packet_id': idx,
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'confidence': confidence,
                    'true_label': true_label
                })
            else:
                print(f"[{timestamp}] ✅ Normal Traffic  | Packet #{idx} | Confidence: {confidence}%")
        
        # Summary
        print("\n" + "=" * 70)
        print("📈 DETECTION SUMMARY")
        print("=" * 70)
        print(f"Total Packets Analyzed: {self.processed_count}")
        print(f"Attacks Detected: {self.attack_count}")
        print(f"Normal Traffic: {self.processed_count - self.attack_count}")
        
        if self.processed_count > 0:
            print(f"Detection Rate: {(self.attack_count / self.processed_count * 100):.2f}%")
        else:
            print("Detection Rate: N/A (No packets processed)")
        
        print(f"Alerts Generated: {len(self.alerts)}")
        print("=" * 70)
        
        return self.alerts

if __name__ == "__main__":
    detector = RealtimeDetector()
    detector.simulate_realtime(num_packets=50, delay=0.2)
