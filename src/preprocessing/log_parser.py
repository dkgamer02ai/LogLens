"""
Log Parser Module

Handles log parsing, feature extraction, and text preprocessing.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np


class LogParser:
    """Parses and extracts features from log entries."""
    
    def __init__(self):
        self.ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        self.timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.url_pattern = re.compile(r'/[a-zA-Z0-9/_-]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Common log levels
        self.log_levels = ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL', 'ALERT']
        
        # Keywords that often indicate anomalies
        self.anomaly_keywords = [
            'failed', 'error', 'exception', 'critical', 'fatal', 'alert', 'warning',
            'unauthorized', 'blocked', 'denied', 'timeout', 'crash', 'breach',
            'attack', 'malware', 'injection', 'exploit', 'suspicious'
        ]
    
    def extract_features(self, log_message: str) -> Dict[str, any]:
        """Extract various features from a log message."""
        features = {}
        
        # Basic text features
        features['message_length'] = len(log_message)
        features['word_count'] = len(log_message.split())
        features['char_diversity'] = len(set(log_message.lower())) / len(log_message) if log_message else 0
        
        # Pattern-based features
        features['ip_count'] = len(self.ip_pattern.findall(log_message))
        features['number_count'] = len(self.number_pattern.findall(log_message))
        features['url_count'] = len(self.url_pattern.findall(log_message))
        features['email_count'] = len(self.email_pattern.findall(log_message))
        
        # Log level features
        message_upper = log_message.upper()
        features['log_level_severity'] = 0
        for i, level in enumerate(self.log_levels):
            if level in message_upper:
                features['log_level_severity'] = i + 1
                break
        
        # Anomaly indicator features
        features['anomaly_keyword_count'] = sum(
            1 for keyword in self.anomaly_keywords 
            if keyword.lower() in log_message.lower()
        )
        
        # Special character features
        features['special_char_ratio'] = sum(
            1 for char in log_message 
            if not char.isalnum() and not char.isspace()
        ) / len(log_message) if log_message else 0
        
        # Uppercase ratio
        features['uppercase_ratio'] = sum(
            1 for char in log_message if char.isupper()
        ) / len(log_message) if log_message else 0
        
        return features
    
    def normalize_log_message(self, log_message: str) -> str:
        """Normalize log message by replacing variable parts with placeholders."""
        normalized = log_message
        
        # Replace IP addresses
        normalized = self.ip_pattern.sub('<IP>', normalized)
        
        # Replace timestamps
        normalized = self.timestamp_pattern.sub('<TIMESTAMP>', normalized)
        
        # Replace numbers (but preserve some context)
        normalized = self.number_pattern.sub('<NUM>', normalized)
        
        # Replace URLs
        normalized = self.url_pattern.sub('<URL>', normalized)
        
        # Replace email addresses
        normalized = self.email_pattern.sub('<EMAIL>', normalized)
        
        return normalized
    
    def parse_structured_log(self, log_line: str) -> Dict[str, str]:
        """Parse a structured log line into components."""
        # Try to extract timestamp, level, and message
        parts = log_line.split(' - ', 2)
        
        result = {
            'timestamp': '',
            'level': '',
            'message': log_line,
            'raw': log_line
        }
        
        if len(parts) >= 3:
            result['timestamp'] = parts[0].strip()
            result['level'] = parts[1].strip()
            result['message'] = parts[2].strip()
        
        return result
    
    def extract_template(self, log_messages: List[str], threshold: float = 0.8) -> Dict[str, List[str]]:
        """Extract log templates by grouping similar normalized messages."""
        templates = {}
        
        for message in log_messages:
            normalized = self.normalize_log_message(message)
            
            # Find if this message matches an existing template
            matched = False
            for template, messages in templates.items():
                similarity = self._calculate_similarity(normalized, template)
                if similarity >= threshold:
                    messages.append(message)
                    matched = True
                    break
            
            if not matched:
                templates[normalized] = [message]
        
        return templates
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Jaccard similarity."""
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0


class LogFeatureExtractor:
    """Advanced feature extraction for machine learning models."""
    
    def __init__(self):
        self.parser = LogParser()
        self.vocabulary = {}
        self.fitted = False
    
    def fit(self, log_messages: List[str]):
        """Fit the feature extractor on training data."""
        # Build vocabulary from all messages
        all_words = set()
        for message in log_messages:
            normalized = self.parser.normalize_log_message(message)
            words = normalized.lower().split()
            all_words.update(words)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.fitted = True
    
    def transform(self, log_messages: List[str]) -> np.ndarray:
        """Transform log messages into feature vectors."""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        features_list = []
        
        for message in log_messages:
            # Extract basic features
            basic_features = self.parser.extract_features(message)
            
            # Create bag-of-words features
            normalized = self.parser.normalize_log_message(message)
            words = normalized.lower().split()
            
            bow_features = np.zeros(len(self.vocabulary))
            for word in words:
                if word in self.vocabulary:
                    bow_features[self.vocabulary[word]] += 1
            
            # Combine all features
            feature_vector = np.concatenate([
                list(basic_features.values()),
                bow_features
            ])
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def fit_transform(self, log_messages: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(log_messages)
        return self.transform(log_messages)


def main():
    """Example usage of the log parser."""
    sample_logs = [
        "2024-01-15 10:30:45 - INFO - User alice successfully logged in from 192.168.1.10",
        "2024-01-15 10:31:12 - ERROR - Failed login attempt from 203.0.113.45",
        "2024-01-15 10:31:30 - CRITICAL - Multiple failed login attempts from 203.0.113.45 - 15 attempts",
        "2024-01-15 10:32:00 - INFO - GET /api/users/1234 - 200 OK - 150ms"
    ]
    
    parser = LogParser()
    
    print("Log Parsing Example:")
    print("=" * 50)
    
    for log in sample_logs:
        print(f"\nOriginal: {log}")
        
        # Parse structured log
        parsed = parser.parse_structured_log(log)
        print(f"Parsed: {parsed}")
        
        # Extract features
        features = parser.extract_features(parsed['message'])
        print(f"Features: {features}")
        
        # Normalize
        normalized = parser.normalize_log_message(parsed['message'])
        print(f"Normalized: {normalized}")


if __name__ == "__main__":
    main()