"""
Unit tests for the log parser.
"""

import pytest
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.log_parser import LogParser, LogFeatureExtractor


class TestLogParser:
    """Test cases for LogParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = LogParser()
    
    def test_extract_features(self):
        """Test feature extraction from log messages."""
        message = "User alice logged in from 192.168.1.10 with 5 attempts"
        features = self.parser.extract_features(message)
        
        # Check basic features
        assert 'message_length' in features
        assert features['message_length'] == len(message)
        assert features['word_count'] == len(message.split())
        
        # Check pattern-based features
        assert features['ip_count'] == 1  # 192.168.1.10
        assert features['number_count'] == 4  # 192, 168, 1, 10, 5
    
    def test_normalize_log_message(self):
        """Test log message normalization."""
        message = "User alice logged in from 192.168.1.10 at 2024-01-15 10:30:45"
        normalized = self.parser.normalize_log_message(message)
        
        # IP should be replaced
        assert '192.168.1.10' not in normalized
        assert '<IP>' in normalized
        
        # Timestamp should be replaced
        assert '2024-01-15 10:30:45' not in normalized
        assert '<TIMESTAMP>' in normalized
    
    def test_parse_structured_log(self):
        """Test parsing of structured log lines."""
        log_line = "2024-01-15 10:30:45 - INFO - User login successful"
        parsed = self.parser.parse_structured_log(log_line)
        
        assert parsed['timestamp'] == "2024-01-15 10:30:45"
        assert parsed['level'] == "INFO"
        assert parsed['message'] == "User login successful"
        assert parsed['raw'] == log_line
    
    def test_anomaly_keyword_detection(self):
        """Test detection of anomaly keywords."""
        normal_message = "User successfully logged in"
        anomaly_message = "CRITICAL: Failed login attempt detected"
        
        normal_features = self.parser.extract_features(normal_message)
        anomaly_features = self.parser.extract_features(anomaly_message)
        
        assert normal_features['anomaly_keyword_count'] == 0
        assert anomaly_features['anomaly_keyword_count'] > 0
    
    def test_log_level_severity(self):
        """Test log level severity detection."""
        info_message = "INFO: User logged in"
        critical_message = "CRITICAL: System failure"
        
        info_features = self.parser.extract_features(info_message)
        critical_features = self.parser.extract_features(critical_message)
        
        assert info_features['log_level_severity'] > 0
        assert critical_features['log_level_severity'] > info_features['log_level_severity']


class TestLogFeatureExtractor:
    """Test cases for LogFeatureExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = LogFeatureExtractor()
        self.sample_messages = [
            "User alice logged in successfully",
            "User bob failed to authenticate",
            "System health check completed",
            "CRITICAL: Database connection failed"
        ]
    
    def test_fit_transform(self):
        """Test fit and transform functionality."""
        features = self.extractor.fit_transform(self.sample_messages)
        
        # Check shape
        assert features.shape[0] == len(self.sample_messages)
        assert features.shape[1] > 0  # Should have some features
        
        # Check that vocabulary was built
        assert len(self.extractor.vocabulary) > 0
        assert self.extractor.fitted
    
    def test_separate_fit_transform(self):
        """Test separate fit and transform calls."""
        # Fit on training data
        train_messages = self.sample_messages[:2]
        self.extractor.fit(train_messages)
        
        # Transform test data
        test_messages = self.sample_messages[2:]
        features = self.extractor.transform(test_messages)
        
        assert features.shape[0] == len(test_messages)
        assert self.extractor.fitted
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        with pytest.raises(ValueError, match="Feature extractor must be fitted"):
            self.extractor.transform(self.sample_messages)