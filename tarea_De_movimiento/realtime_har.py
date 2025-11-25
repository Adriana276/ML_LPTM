"""
Real-Time Activity Recognition from Arduino/ESP32
Uses serial communication and sliding window for prediction
Scikit-learn version (compatible with Python 3.14)
"""

import serial
import numpy as np
import pickle
import os
from config import *
import time
from collections import deque

class RealTimeHAR:
    def __init__(self, port=SERIAL_PORT, baud_rate=BAUD_RATE):
        """Initialize real-time HAR system"""
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.model = None
        self.input_buffer = deque(maxlen=STEP_SIZE)
        
        # Load model
        print("Loading trained model...")
        model_file = os.path.join(MODEL_PATH, 'model.pkl')
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully!")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Input features: {STEP_SIZE * SENSOR_NUM}")
            print(f"Output classes: {NUM_CLASSES}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Make sure to train the model first using train_model.py")
            raise
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        
    def connect_serial(self):
        """Connect to Arduino/ESP32"""
        print(f"Connecting to {self.port} at {self.baud_rate} baud...")
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=TIMEOUT
            )
            time.sleep(2)  # Wait for connection to stabilize
            print("Serial connection established!")
            
            # Clear initial messages
            for _ in range(5):
                if self.serial_conn.in_waiting:
                    print(self.serial_conn.readline().decode('utf-8').strip())
            
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            print(f"Available ports: Check Device Manager (Windows) or ls /dev/tty* (Linux)")
            return False
    
    def parse_data_packet(self, line):
        """Parse data packet from Arduino"""
        try:
            # Format: !ax,ay,az,gx,gy,gz@
            line = line.strip()
            
            if not line.startswith('!') or not line.endswith('@'):
                return None
            
            # Remove ! and @
            data_str = line[1:-1]
            
            # Split by comma
            values = data_str.split(',')
            
            if len(values) != SENSOR_NUM:
                return None
            
            # Convert to float
            values = [float(v) for v in values]
            
            return values
            
        except Exception as e:
            # print(f"Error parsing: {e}")
            return None
    
    def predict(self):
        """Make prediction using current buffer"""
        if len(self.input_buffer) < STEP_SIZE:
            return None, None
        
        # Prepare input - flatten to (1, STEP_SIZE * SENSOR_NUM)
        input_data = np.array(list(self.input_buffer)).reshape(1, -1)
        
        # Predict
        result = self.model.predict(input_data)[0]
        pred_proba = self.model.predict_proba(input_data)[0]
        confidence = pred_proba[result] * 100
        
        # Get activity name
        activity = CLASS_NAMES[result]
        
        # Smooth predictions (majority vote over last 5 predictions)
        self.prediction_history.append(result)
        if len(self.prediction_history) >= 3:
            smoothed_result = max(self.prediction_history, key=list(self.prediction_history).count)
            smoothed_activity = CLASS_NAMES[smoothed_result]
        else:
            smoothed_activity = activity
        
        return smoothed_activity, confidence
    
    def run(self):
        """Main loop for real-time prediction"""
        if not self.connect_serial():
            return
        
        print("\n" + "=" * 60)
        print("REAL-TIME ACTIVITY RECOGNITION")
        print("=" * 60)
        print("Waiting for sensor data...")
        print("Activities: WAL=Walking, JUM=Jumping, FALL=Falling, LYI=Lying")
        print("-" * 60)
        
        packet_count = 0
        
        try:
            while True:
                # Read line from serial
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore')
                    
                    # Parse data
                    data = self.parse_data_packet(line)
                    
                    if data is not None:
                        # Add to buffer
                        self.input_buffer.append(data)
                        packet_count += 1
                        
                        # Make prediction when buffer is full
                        if len(self.input_buffer) == STEP_SIZE:
                            activity, confidence = self.predict()
                            
                            if activity:
                                # Print activity with color coding
                                activity_symbols = {
                                    'WAL': '🚶',
                                    'JUM': '🤾',
                                    'FALL': '⚠️',
                                    'LYI': '🛌'
                                }
                                
                                symbol = activity_symbols.get(activity, '❓')
                                
                                # Print status line
                                status = f"[{packet_count:5d}] {symbol} Activity: {activity:4s} | Confidence: {confidence:5.1f}%"
                                
                                # Highlight falls
                                if activity == 'FALL':
                                    print(f"\n{'!' * 60}")
                                    print(f">>> FALL DETECTED! <<<")
                                    print(f"{'!' * 60}\n")
                                elif activity == 'LYI':
                                    print(f"{status} (Person is lying down)")
                                else:
                                    print(status)
                        
                        # Show progress every 10 packets before buffer fills
                        elif packet_count % 10 == 0:
                            print(f"Collecting data... ({len(self.input_buffer)}/{STEP_SIZE})")
        
        except KeyboardInterrupt:
            print("\n\nStopping real-time recognition...")
        
        finally:
            if self.serial_conn:
                self.serial_conn.close()
                print("Serial connection closed")
            
            print("\n" + "=" * 60)
            print(f"Total packets processed: {packet_count}")
            print("=" * 60)

if __name__ == "__main__":
    import sys
    
    # Allow custom port via command line
    port = SERIAL_PORT
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    print("=" * 60)
    print("Real-Time Human Activity Recognition System")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"Baud Rate: {BAUD_RATE}")
    print(f"Window Size: {STEP_SIZE} timesteps")
    print("=" * 60)
    
    har = RealTimeHAR(port=port)
    har.run()
