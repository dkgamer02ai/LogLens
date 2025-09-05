#!/usr/bin/env python3
"""
LogLens Demo Script

This script demonstrates the capabilities of the LogLens system
with a complete end-to-end workflow.
"""

import asyncio
import json
import time
import requests
import subprocess
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogLensDemo:
    """
    Comprehensive demo of LogLens functionality.
    """
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.api_process: Optional[subprocess.Popen] = None
        
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "="*60)
        print(f"🔍 {title}")
        print("="*60)
    
    def print_step(self, step: str):
        """Print a step description."""
        print(f"\n📋 {step}")
        print("-" * 40)
    
    def setup_environment(self):
        """Set up the demo environment."""
        self.print_header("LogLens System Demo")
        self.print_step("Setting up demo environment")
        
        # Create necessary directories
        dirs = ["data/samples", "models", "logs"]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        
        print("✅ Environment setup complete")
    
    def generate_sample_data(self):
        """Generate sample data for demonstration."""
        self.print_step("Generating sample data")
        
        try:
            # Generate synthetic dataset
            from src.data.generator import SyntheticLogGenerator
            
            generator = SyntheticLogGenerator(seed=42)
            logs = generator.generate_dataset(
                num_samples=1000,
                anomaly_ratio=0.15,
                include_scenarios=True
            )
            
            # Save dataset
            output_path = "data/samples/demo_dataset.json"
            with open(output_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print(f"✅ Generated {len(logs)} log entries")
            print(f"✅ Saved to {output_path}")
            
            # Create sample log files
            from data.samples import save_sample_logs
            save_sample_logs("data/samples")
            print("✅ Created sample log files")
            
        except Exception as e:
            print(f"❌ Error generating data: {e}")
            return False
        
        return True
    
    def train_model(self):
        """Train the anomaly detection model."""
        self.print_step("Training anomaly detection model")
        
        try:
            from src.models.trainer import ModelTrainer
            
            trainer = ModelTrainer()
            
            print("🤖 Training BERT-based anomaly detector...")
            print("   (This may take several minutes)")
            
            # Use smaller dataset for demo
            results = trainer.run_training_pipeline(
                data_path=None,  # Use synthetic data
                num_synthetic=1000,  # Smaller for demo
                output_dir="models"
            )
            
            print(f"✅ Training completed!")
            print(f"   Accuracy: {results['evaluation']['accuracy']:.3f}")
            print(f"   F1-Score: {results['evaluation']['f1']:.3f}")
            print(f"   Model saved to: {results['training']['model_path']}")
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            print("   Using pre-trained BERT model for demo")
            return False
        
        return True
    
    def start_api_server(self):
        """Start the API server."""
        self.print_step("Starting LogLens API server")
        
        try:
            # Start API server in background
            self.api_process = subprocess.Popen([
                sys.executable, "-m", "src.api.main"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            print("🚀 Starting API server...")
            time.sleep(10)
            
            # Test connection
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API server running at {self.api_url}")
                return True
            else:
                print(f"❌ API server not responding (status: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    def test_anomaly_detection(self):
        """Test anomaly detection with sample logs."""
        self.print_step("Testing anomaly detection")
        
        # Sample logs to test
        test_logs = [
            {
                "message": "User admin logged in successfully from 192.168.1.50",
                "expected": "Normal"
            },
            {
                "message": "Failed login attempt from 192.168.1.100 - Invalid credentials",
                "expected": "Anomaly"
            },
            {
                "message": "SQL injection attempt: SELECT * FROM users WHERE id='1' OR '1'='1'",
                "expected": "Anomaly"
            },
            {
                "message": "Database query executed successfully in 45ms",
                "expected": "Normal"
            },
            {
                "message": "XSS attack detected: <script>alert('XSS')</script>",
                "expected": "Anomaly"
            }
        ]
        
        print(f"🧪 Testing {len(test_logs)} log entries")
        
        for i, test_log in enumerate(test_logs, 1):
            try:
                response = requests.post(
                    f"{self.api_url}/detect",
                    json={"message": test_log["message"]},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = "🚨 ANOMALY" if result["is_anomaly"] else "✅ NORMAL"
                    score = result["anomaly_score"]
                    severity = result["severity"]
                    
                    print(f"{i}. {status} (Score: {score:.3f}, Severity: {severity})")
                    print(f"   Message: {test_log['message'][:60]}...")
                    print(f"   Expected: {test_log['expected']}")
                    
                    # Check if prediction matches expectation
                    is_correct = (
                        (result["is_anomaly"] and test_log["expected"] == "Anomaly") or
                        (not result["is_anomaly"] and test_log["expected"] == "Normal")
                    )
                    
                    if is_correct:
                        print("   ✅ Correct prediction!")
                    else:
                        print("   ⚠️  Unexpected prediction")
                    
                else:
                    print(f"{i}. ❌ API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"{i}. ❌ Error: {e}")
            
            print()
    
    def test_batch_detection(self):
        """Test batch anomaly detection."""
        self.print_step("Testing batch anomaly detection")
        
        # Load sample data
        try:
            with open("data/samples/labeled_dataset.json", 'r') as f:
                dataset = json.load(f)
            
            # Take first 10 entries for demo
            sample_logs = dataset[:10]
            
            batch_request = {
                "logs": [
                    {"message": log["message"]} for log in sample_logs
                ],
                "threshold": 0.7
            }
            
            print(f"📦 Processing batch of {len(sample_logs)} log entries")
            
            response = requests.post(
                f"{self.api_url}/detect/batch",
                json=batch_request,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Batch processing completed!")
                print(f"   Total processed: {result['total_processed']}")
                print(f"   Anomalies detected: {result['anomalies_detected']}")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                
                # Show detailed results
                print("\n📊 Detailed results:")
                for i, (log, res) in enumerate(zip(sample_logs, result["results"])):
                    status = "🚨" if res["is_anomaly"] else "✅"
                    expected = "🚨" if log["is_anomaly"] else "✅"
                    print(f"   {i+1}. {status} Expected: {expected} Score: {res['anomaly_score']:.3f}")
                
            else:
                print(f"❌ Batch processing failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
    
    def demonstrate_real_time_monitoring(self):
        """Demonstrate real-time log monitoring."""
        self.print_step("Demonstrating real-time monitoring")
        
        print("🔄 Starting real-time log monitoring simulation")
        print("   (Will process sample logs in real-time)")
        
        try:
            # Read sample logs
            with open("data/samples/mixed_logs.txt", 'r') as f:
                log_lines = f.readlines()
            
            print(f"📝 Processing {len(log_lines)} log entries in real-time...")
            
            anomalies_detected = 0
            
            for i, line in enumerate(log_lines, 1):
                # Extract message from log line
                if "] " in line:
                    message = line.split("] ", 1)[1].strip()
                else:
                    message = line.strip()
                
                if not message:
                    continue
                
                try:
                    response = requests.post(
                        f"{self.api_url}/detect",
                        json={"message": message},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result["is_anomaly"]:
                            anomalies_detected += 1
                            print(f"   🚨 ANOMALY #{anomalies_detected}: {message[:50]}...")
                            print(f"      Score: {result['anomaly_score']:.3f}, Severity: {result['severity']}")
                        else:
                            print(f"   ✅ Normal log #{i}")
                        
                        # Simulate real-time processing delay
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"   ❌ Error processing log {i}: {e}")
            
            print(f"\n✅ Real-time monitoring complete!")
            print(f"   Total logs processed: {len(log_lines)}")
            print(f"   Anomalies detected: {anomalies_detected}")
            
        except Exception as e:
            print(f"❌ Error in real-time monitoring: {e}")
    
    def show_system_status(self):
        """Show system status and statistics."""
        self.print_step("System status and statistics")
        
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            
            if response.status_code == 200:
                status = response.json()
                
                print("📊 System Status:")
                print(f"   Status: {status['status']}")
                print(f"   Model Loaded: {'✅' if status['model_loaded'] else '❌'}")
                print(f"   Uptime: {status['uptime']}")
                print(f"   Total Requests: {status['total_requests']}")
                print(f"   Anomalies Detected: {status['anomalies_detected']}")
                
                if status.get('last_alert'):
                    print(f"   Last Alert: {status['last_alert']}")
                else:
                    print("   Last Alert: None")
                    
            else:
                print(f"❌ Could not get status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    def cleanup(self):
        """Clean up demo resources."""
        self.print_step("Cleaning up")
        
        # Stop API server
        if self.api_process:
            print("🛑 Stopping API server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=10)
                print("✅ API server stopped")
            except subprocess.TimeoutExpired:
                self.api_process.kill()
                print("⚠️  API server force-killed")
        
        print("✅ Cleanup complete")
    
    def run_demo(self):
        """Run the complete demo."""
        try:
            # Setup
            self.setup_environment()
            
            # Generate data
            if not self.generate_sample_data():
                print("⚠️  Continuing demo with limited functionality")
            
            # Train model (optional - can be skipped for faster demo)
            print("\n🤔 Do you want to train a new model? (This takes several minutes)")
            train_choice = input("   Enter 'y' to train, or any other key to skip: ").lower()
            
            if train_choice == 'y':
                self.train_model()
            else:
                print("⏭️  Skipping model training (will use pre-trained model)")
            
            # Start API server
            if not self.start_api_server():
                print("❌ Cannot continue demo without API server")
                return
            
            # Run tests
            self.test_anomaly_detection()
            self.test_batch_detection()
            self.demonstrate_real_time_monitoring()
            self.show_system_status()
            
            # Demo complete
            self.print_header("Demo Complete!")
            print("🎉 LogLens demonstration completed successfully!")
            print("\n📚 Next steps:")
            print("   - Check the API documentation at http://localhost:8000/docs")
            print("   - Try the CLI commands: python -m src.cli --help")
            print("   - Explore the monitoring capabilities")
            print("   - Configure alerting for your environment")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Demo interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n\n🛑 Received shutdown signal")
    sys.exit(0)


def main():
    """Main demo entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run demo
    demo = LogLensDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
