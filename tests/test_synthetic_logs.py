"""
Unit tests for the synthetic log generator.
"""

import pytest
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.synthetic_logs import SyntheticLogGenerator, LogEntry


class TestSyntheticLogGenerator:
    """Test cases for SyntheticLogGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SyntheticLogGenerator()
    
    def test_log_entry_creation(self):
        """Test creation of log entries."""
        # Test normal log entry
        normal_entry = self.generator.generate_log_entry(is_anomaly=False)
        assert isinstance(normal_entry, LogEntry)
        assert not normal_entry.is_anomaly
        assert normal_entry.level in ['INFO', 'DEBUG', 'WARN']
        assert normal_entry.source in ['access', 'error', 'security', 'system']
        
        # Test anomalous log entry
        anomaly_entry = self.generator.generate_log_entry(is_anomaly=True)
        assert isinstance(anomaly_entry, LogEntry)
        assert anomaly_entry.is_anomaly
        assert anomaly_entry.level in ['WARNING', 'CRITICAL', 'ALERT', 'FATAL']
    
    def test_dataset_generation(self):
        """Test dataset generation with correct proportions."""
        total_count = 1000
        anomaly_rate = 0.1
        
        dataset = self.generator.generate_dataset(total_count, anomaly_rate)
        
        # Check total count
        assert len(dataset) == total_count
        
        # Check anomaly rate (allow for small variance due to rounding)
        anomaly_count = sum(1 for entry in dataset if entry.is_anomaly)
        expected_anomalies = int(total_count * anomaly_rate)
        assert abs(anomaly_count - expected_anomalies) <= 1
    
    def test_variable_substitution(self):
        """Test that variables are properly substituted in templates."""
        entry = self.generator.generate_log_entry()
        
        # Check that no template variables remain
        assert '{user}' not in entry.message
        assert '{ip}' not in entry.message
        assert '{id}' not in entry.message
    
    def test_timestamp_format(self):
        """Test timestamp format."""
        entry = self.generator.generate_log_entry()
        
        # Should be in format YYYY-MM-DD HH:MM:SS
        from datetime import datetime
        try:
            datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pytest.fail("Invalid timestamp format")
    
    def test_save_dataset(self, tmp_path):
        """Test saving dataset to CSV file."""
        dataset = self.generator.generate_dataset(100, 0.1)
        filepath = tmp_path / "test_logs.csv"
        
        self.generator.save_dataset(dataset, str(filepath))
        
        # Check file exists
        assert filepath.exists()
        
        # Check file contents
        import pandas as pd
        df = pd.read_csv(filepath)
        assert len(df) == 100
        assert 'timestamp' in df.columns
        assert 'level' in df.columns
        assert 'source' in df.columns
        assert 'message' in df.columns
        assert 'is_anomaly' in df.columns