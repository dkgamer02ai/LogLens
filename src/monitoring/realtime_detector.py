"""
Real-time Log Anomaly Detection System

Monitors log files and streams for anomalous patterns in real-time.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml
from datetime import datetime, timedelta
import threading
import queue
import logging
from collections import deque

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.bert_anomaly_detector import LogAnomalyTrainer
from src.preprocessing.log_parser import LogParser
from src.preprocessing.embeddings import LogEmbeddingsGenerator, SemanticLogAnalyzer


class LogFileHandler(FileSystemEventHandler):
    """Handle file system events for log files."""
    
    def __init__(self, log_queue: queue.Queue, monitored_extensions: List[str] = None):
        self.log_queue = log_queue
        self.monitored_extensions = monitored_extensions or ['.log', '.txt']
        self.file_positions = {}  # Track file read positions
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not any(file_path.endswith(ext) for ext in self.monitored_extensions):
            return
        
        # Read new lines from the file
        self._read_new_lines(file_path)
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not any(file_path.endswith(ext) for ext in self.monitored_extensions):
            return
        
        # Initialize position for new file
        self.file_positions[file_path] = 0
        self._read_new_lines(file_path)
    
    def _read_new_lines(self, file_path: str):
        """Read new lines from a file since last position."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to last known position
                last_position = self.file_positions.get(file_path, 0)
                f.seek(last_position)
                
                # Read new lines
                new_lines = f.readlines()
                
                # Update position
                self.file_positions[file_path] = f.tell()
                
                # Queue new lines for processing
                for line in new_lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        self.log_queue.put({
                            'file_path': file_path,
                            'line': line,
                            'timestamp': datetime.now().isoformat()
                        })
        
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")


class RealTimeLogDetector:
    """Real-time log anomaly detection system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.log_queue = queue.Queue()
        self.observer = Observer()
        self.log_parser = LogParser()
        self.embeddings_generator = LogEmbeddingsGenerator()
        self.semantic_analyzer = SemanticLogAnalyzer(self.embeddings_generator)
        self.bert_trainer = LogAnomalyTrainer(config_path)
        
        # State
        self.is_running = False
        self.detection_thread = None
        self.stats = {
            'total_logs_processed': 0,
            'anomalies_detected': 0,
            'start_time': None,
            'last_anomaly_time': None
        }
        
        # Recent logs buffer for pattern analysis
        self.recent_logs = deque(maxlen=1000)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback function to be called when anomalies are detected."""
        self.alert_callbacks.append(callback)
    
    def load_model(self, model_path: str):
        """Load the trained BERT model."""
        self.bert_trainer.load_model(model_path)
        self.logger.info(f"Loaded BERT model from {model_path}")
    
    def fit_semantic_baseline(self, normal_logs: List[str]):
        """Fit semantic analyzer on normal log patterns."""
        self.semantic_analyzer.fit_normal_patterns(normal_logs)
        self.logger.info("Fitted semantic analyzer on normal patterns")
    
    def start_monitoring(self, log_paths: List[str]):
        """Start monitoring specified log paths."""
        if self.is_running:
            self.logger.warning("Monitoring is already running")
            return
        
        self.logger.info("Starting log monitoring...")
        self.stats['start_time'] = datetime.now()
        
        # Setup file watchers
        for log_path in log_paths:
            if os.path.exists(log_path):
                event_handler = LogFileHandler(self.log_queue)
                self.observer.schedule(event_handler, log_path, recursive=True)
                self.logger.info(f"Monitoring path: {log_path}")
            else:
                self.logger.warning(f"Path does not exist: {log_path}")
        
        # Start file watcher
        self.observer.start()
        
        # Start detection thread
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()
        
        self.logger.info("Log monitoring started")
    
    def stop_monitoring(self):
        """Stop log monitoring."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping log monitoring...")
        
        # Stop detection thread
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join()
        
        # Stop file watcher
        self.observer.stop()
        self.observer.join()
        
        self.logger.info("Log monitoring stopped")
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        batch_size = self.config['monitoring']['batch_size']
        check_interval = self.config['monitoring']['check_interval']
        
        log_batch = []
        last_process_time = time.time()
        
        while self.is_running:
            try:
                # Collect logs from queue
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                    log_batch.append(log_entry)
                    self.recent_logs.append(log_entry)
                except queue.Empty:
                    pass
                
                # Process batch if ready
                current_time = time.time()
                if (len(log_batch) >= batch_size or 
                    (log_batch and current_time - last_process_time >= check_interval)):
                    
                    self._process_log_batch(log_batch)
                    log_batch = []
                    last_process_time = current_time
                
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
    
    def _process_log_batch(self, log_batch: List[Dict]):
        """Process a batch of log entries for anomaly detection."""
        if not log_batch:
            return
        
        messages = [entry['line'] for entry in log_batch]
        
        try:
            # Method 1: BERT-based detection (if model is loaded)
            bert_anomalies = []
            if hasattr(self.bert_trainer, 'model') and self.bert_trainer.model is not None:
                bert_results = self.bert_trainer.predict(messages)
                bert_anomalies = [
                    (i, result) for i, result in enumerate(bert_results)
                    if result['is_anomaly']
                ]
            
            # Method 2: Semantic-based detection
            semantic_anomalies = []
            if self.semantic_analyzer.normal_embeddings is not None:
                semantic_results = self.semantic_analyzer.batch_detect_anomalies(messages)
                semantic_anomalies = [
                    (i, result) for i, result in enumerate(semantic_results)
                    if result[0]  # is_anomaly
                ]
            
            # Method 3: Rule-based detection (using log parser)
            rule_anomalies = []
            for i, message in enumerate(messages):
                features = self.log_parser.extract_features(message)
                
                # Simple rule-based anomaly detection
                is_anomaly = (
                    features['anomaly_keyword_count'] > 0 or
                    features['log_level_severity'] >= 6  # ERROR, CRITICAL, FATAL, ALERT
                )
                
                if is_anomaly:
                    rule_anomalies.append((i, features))
            
            # Combine results and trigger alerts
            all_anomaly_indices = set()
            all_anomaly_indices.update([i for i, _ in bert_anomalies])
            all_anomaly_indices.update([i for i, _ in semantic_anomalies])
            all_anomaly_indices.update([i for i, _ in rule_anomalies])
            
            # Process detected anomalies
            for i in all_anomaly_indices:
                log_entry = log_batch[i]
                
                # Create anomaly alert
                alert = {
                    'timestamp': log_entry['timestamp'],
                    'file_path': log_entry['file_path'],
                    'message': log_entry['line'],
                    'detection_methods': [],
                    'details': {}
                }
                
                # Add detection method details
                if i in [idx for idx, _ in bert_anomalies]:
                    bert_result = next(result for idx, result in bert_anomalies if idx == i)
                    alert['detection_methods'].append('BERT')
                    alert['details']['bert'] = {
                        'confidence': bert_result['confidence'],
                        'anomaly_prob': bert_result['anomaly_prob']
                    }
                
                if i in [idx for idx, _ in semantic_anomalies]:
                    semantic_result = next(result for idx, result in semantic_anomalies if idx == i)
                    alert['detection_methods'].append('Semantic')
                    alert['details']['semantic'] = {
                        'max_similarity': semantic_result[1]
                    }
                
                if i in [idx for idx, _ in rule_anomalies]:
                    rule_result = next(result for idx, result in rule_anomalies if idx == i)
                    alert['detection_methods'].append('Rule-based')
                    alert['details']['rule_based'] = rule_result
                
                # Trigger alert
                self._trigger_alert(alert)
            
            # Update statistics
            self.stats['total_logs_processed'] += len(log_batch)
            if all_anomaly_indices:
                self.stats['anomalies_detected'] += len(all_anomaly_indices)
                self.stats['last_anomaly_time'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error processing log batch: {e}")
    
    def _trigger_alert(self, alert: Dict):
        """Trigger alert for detected anomaly."""
        self.logger.warning(f"ANOMALY DETECTED: {alert['message']}")
        self.logger.info(f"Detection methods: {', '.join(alert['detection_methods'])}")
        
        # Call all registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        current_stats = self.stats.copy()
        
        if current_stats['start_time']:
            runtime = datetime.now() - current_stats['start_time']
            current_stats['runtime_seconds'] = runtime.total_seconds()
            
            if current_stats['total_logs_processed'] > 0:
                current_stats['logs_per_second'] = (
                    current_stats['total_logs_processed'] / runtime.total_seconds()
                )
                current_stats['anomaly_rate'] = (
                    current_stats['anomalies_detected'] / current_stats['total_logs_processed']
                )
        
        return current_stats
    
    def get_recent_logs(self, count: int = 100) -> List[Dict]:
        """Get recent log entries."""
        return list(self.recent_logs)[-count:]


def main():
    """Main function for real-time monitoring."""
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description="Real-time Log Anomaly Detection")
    parser.add_argument("--log-paths", nargs="+", default=["./logs/"], help="Paths to monitor")
    parser.add_argument("--model", type=str, help="Path to trained BERT model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealTimeLogDetector(args.config)
    
    # Load model if provided
    if args.model and os.path.exists(args.model):
        detector.load_model(args.model)
    
    # Simple console alert callback
    def console_alert(alert):
        print(f"\n🚨 ANOMALY ALERT 🚨")
        print(f"Time: {alert['timestamp']}")
        print(f"File: {alert['file_path']}")
        print(f"Message: {alert['message']}")
        print(f"Methods: {', '.join(alert['detection_methods'])}")
        print("-" * 50)
    
    detector.add_alert_callback(console_alert)
    
    # Graceful shutdown handler
    def signal_handler(signum, frame):
        print("\nShutting down...")
        detector.stop_monitoring()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    detector.start_monitoring(args.log_paths)
    
    # Keep running and show stats periodically
    try:
        while True:
            time.sleep(30)  # Show stats every 30 seconds
            stats = detector.get_stats()
            print(f"\n📊 Stats: {stats['total_logs_processed']} logs processed, "
                  f"{stats['anomalies_detected']} anomalies detected")
    except KeyboardInterrupt:
        detector.stop_monitoring()


if __name__ == "__main__":
    main()