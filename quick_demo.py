#!/usr/bin/env python3
"""
Quick Demo of LogLens Core Functionality

This script demonstrates the basic functionality without requiring
heavy dependencies like PyTorch or transformers.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_log_parser():
    """Test the log parser functionality."""
    print("🔍 Testing Log Parser")
    print("-" * 40)
    
    try:
        from preprocessing.log_parser import LogParser
        
        parser = LogParser()
        
        # Test different log formats
        test_logs = [
            "2024-01-15 10:00:00 [INFO] User admin logged in successfully",
            '{"timestamp": "2024-01-15T10:00:00Z", "level": "ERROR", "message": "Database connection failed"}',
            "192.168.1.50 - - [15/Jan/2024:10:00:00 +0000] \"GET /index.html HTTP/1.1\" 200 1024",
        ]
        
        for i, log in enumerate(test_logs, 1):
            print(f"{i}. Testing: {log[:50]}...")
            
            entry = parser.parse_line(log)
            if entry:
                print(f"   ✅ Parsed - Level: {entry.level}, Source: {entry.source}")
                print(f"      Message: {entry.message[:60]}...")
            else:
                print("   ❌ Failed to parse")
            print()
        
        print("✅ Log Parser test completed!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import log parser: {e}\n")
        return False
    except Exception as e:
        print(f"❌ Error testing log parser: {e}\n")
        return False


def test_feature_extractor():
    """Test basic feature extraction (without ML dependencies)."""
    print("🔧 Testing Feature Extractor (Basic Features)")
    print("-" * 40)
    
    try:
        from preprocessing.log_parser import LogParser, LogEntry
        
        # Create a test log entry
        entry = LogEntry(
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            level="ERROR",
            source="auth_service",
            message="Failed login attempt from 192.168.1.100 - Invalid credentials",
            raw_log="2024-01-15 10:00:00 [ERROR] Failed login attempt from 192.168.1.100",
            metadata={}
        )
        
        # Manual feature extraction (without heavy dependencies)
        print("📊 Extracting basic features...")
        
        # Basic features
        message = entry.message
        basic_features = {
            "hour": entry.timestamp.hour,
            "day_of_week": entry.timestamp.weekday(),
            "message_length": len(message),
            "word_count": len(message.split()),
            "log_level_error": 1 if entry.level == "ERROR" else 0,
            "has_ip_address": 1 if any(c.isdigit() and '.' in message for c in message.split()) else 0,
            "has_failed_login": 1 if "failed login" in message.lower() else 0,
        }
        
        print("   Basic Features:")
        for key, value in basic_features.items():
            print(f"     {key}: {value}")
        
        print("✅ Feature extraction test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error testing feature extractor: {e}\n")
        return False


def test_synthetic_data_generator():
    """Test synthetic data generation."""
    print("🎲 Testing Synthetic Data Generator")
    print("-" * 40)
    
    try:
        from data.generator import SyntheticLogGenerator
        
        generator = SyntheticLogGenerator(seed=42)
        
        print("📝 Generating sample log entries...")
        
        # Generate a few sample entries
        timestamp = datetime.now()
        
        normal_entry = generator.generate_log_entry(timestamp, force_anomaly=False)
        anomaly_entry = generator.generate_log_entry(timestamp, force_anomaly=True)
        
        print("   Normal log entry:")
        print(f"     Level: {normal_entry['level']}")
        print(f"     Message: {normal_entry['message']}")
        print(f"     Is Anomaly: {normal_entry['is_anomaly']}")
        
        print("\n   Anomalous log entry:")
        print(f"     Level: {anomaly_entry['level']}")
        print(f"     Message: {anomaly_entry['message']}")
        print(f"     Is Anomaly: {anomaly_entry['is_anomaly']}")
        
        print("\n🎯 Generating small dataset...")
        
        # Generate small dataset
        dataset = generator.generate_dataset(num_samples=10, anomaly_ratio=0.3)
        
        anomalies = sum(1 for log in dataset if log["is_anomaly"])
        normal = len(dataset) - anomalies
        
        print(f"   Generated {len(dataset)} entries:")
        print(f"     Normal: {normal}")
        print(f"     Anomalies: {anomalies}")
        
        print("✅ Synthetic data generation test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error testing synthetic data generator: {e}\n")
        return False


def test_alert_manager():
    """Test alert manager (without external dependencies)."""
    print("🚨 Testing Alert Manager")
    print("-" * 40)
    
    try:
        import tempfile
        import yaml
        
        # Create minimal config
        config = {
            "alerting": {
                "channels": {
                    "email": {
                        "enabled": False,  # Disabled for testing
                        "smtp_host": "localhost",
                        "recipients": ["test@example.com"]
                    }
                },
                "rules": {
                    "high_severity": {
                        "threshold": 0.7,
                        "cooldown": 300,
                        "channels": ["email"]
                    }
                },
                "templates": {
                    "email": "Alert: {anomaly_details}"
                }
            }
        }
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        from alerting.alert_manager import AlertManager
        
        alert_manager = AlertManager(config_path)
        
        print("📧 Creating test alert...")
        
        alert = alert_manager.create_alert(
            log_message="Failed login attempt from suspicious IP",
            anomaly_score=0.85,
            anomaly_details="High confidence security anomaly detected",
            source="auth_service"
        )
        
        print(f"   Alert ID: {alert.id}")
        print(f"   Severity: {alert.severity}")
        print(f"   Confidence: {alert.confidence:.2f}")
        print(f"   Message: {alert.log_message}")
        
        # Clean up
        Path(config_path).unlink()
        
        print("✅ Alert manager test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error testing alert manager: {e}\n")
        return False


def create_sample_files():
    """Create sample data files for demonstration."""
    print("📁 Creating Sample Files")
    print("-" * 40)
    
    try:
        # Create data directory
        data_dir = Path("data/samples")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample log entries
        sample_logs = [
            "2024-01-15 10:00:00 [INFO] User admin logged in successfully from 192.168.1.50",
            "2024-01-15 10:00:05 [INFO] HTTP GET /api/users/profile - Status: 200 - 45ms",
            "2024-01-15 10:01:00 [ERROR] Failed login attempt from 192.168.1.100 - Invalid credentials",
            "2024-01-15 10:01:05 [ERROR] SQL injection attempt detected: SELECT * FROM users WHERE id='1' OR '1'='1'",
            "2024-01-15 10:01:10 [CRITICAL] Unauthorized access attempt to /admin/panel",
        ]
        
        # Save as text file
        with open(data_dir / "sample_logs.txt", "w") as f:
            for log in sample_logs:
                f.write(log + "\n")
        
        # Create labeled dataset
        labeled_data = []
        for log in sample_logs:
            # Simple heuristic to label anomalies
            is_anomaly = any(keyword in log.lower() for keyword in 
                           ["failed", "error", "critical", "injection", "unauthorized"])
            
            labeled_data.append({
                "message": log.split("] ", 1)[1] if "] " in log else log,
                "raw_log": log,
                "is_anomaly": is_anomaly,
                "timestamp": "2024-01-15T10:00:00Z"
            })
        
        with open(data_dir / "labeled_dataset.json", "w") as f:
            json.dump(labeled_data, f, indent=2)
        
        print(f"✅ Sample files created in {data_dir}/")
        print(f"   - sample_logs.txt ({len(sample_logs)} entries)")
        print(f"   - labeled_dataset.json ({len(labeled_data)} entries)")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample files: {e}\n")
        return False


def main():
    """Run the basic demonstration."""
    print("🔍 LogLens Basic Functionality Demo")
    print("=" * 60)
    print("This demo tests core functionality without heavy ML dependencies.")
    print("For full AI features, install: pip install -r requirements.txt")
    print("=" * 60)
    print()
    
    # Track test results
    results = []
    
    # Run tests
    results.append(("Log Parser", test_log_parser()))
    results.append(("Feature Extractor", test_feature_extractor()))
    results.append(("Synthetic Data Generator", test_synthetic_data_generator()))
    results.append(("Alert Manager", test_alert_manager()))
    results.append(("Sample Files", create_sample_files()))
    
    # Summary
    print("📊 Test Summary")
    print("-" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed!")
        print("\n📚 Next steps:")
        print("   1. Install full dependencies: pip install -r requirements.txt")
        print("   2. Train a model: python -m src.models.trainer")
        print("   3. Start the API: python -m src.api.main")
        print("   4. Run full demo: python demo.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check error messages above.")
    
    print(f"\n📁 Sample data available in: {Path('data/samples').absolute()}")


if __name__ == "__main__":
    main()
