"""
Log Generator Script for Testing

Generates realistic log entries to test the anomaly detection system.
"""

import time
import random
import os
from datetime import datetime
import sys

# Add src to path
sys.path.append('/app')
from src.data.synthetic_logs import SyntheticLogGenerator


def write_log_entry(log_file, entry):
    """Write a log entry to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_entry = f"{timestamp} - {entry.level} - {entry.source} - {entry.message}\n"
    
    with open(log_file, 'a') as f:
        f.write(formatted_entry)
    
    print(f"Generated: {formatted_entry.strip()}")


def main():
    """Generate logs continuously."""
    log_file = "/app/logs/application.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    generator = SyntheticLogGenerator()
    
    print("Starting log generation...")
    
    while True:
        # Generate normal logs most of the time
        if random.random() < 0.9:  # 90% normal logs
            entry = generator.generate_log_entry(is_anomaly=False)
        else:  # 10% anomalous logs
            entry = generator.generate_log_entry(is_anomaly=True)
        
        write_log_entry(log_file, entry)
        
        # Random delay between 1-10 seconds
        time.sleep(random.uniform(1, 10))


if __name__ == "__main__":
    main()