"""
Feature Extraction Module

This module extracts features from parsed log entries for machine learning
and semantic similarity analysis.
"""

import re
import math
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import logging

from .log_parser import LogEntry

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Comprehensive feature extraction for log entries.
    
    Extracts both traditional ML features and modern embeddings
    for anomaly detection.
    """
    
    def __init__(
        self,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_cache: bool = True
    ):
        self.embeddings_model_name = embeddings_model
        self.use_cache = use_cache
        self.embeddings_model = None
        self.tfidf_vectorizer = None
        self.feature_cache = {}
        
        # Precompiled regex patterns for efficiency
        self.patterns = {
            "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "url": re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            "file_path": re.compile(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*|/(?:[^/\0]+/)*[^/\0]*'),
            "hash": re.compile(r'\b[a-fA-F0-9]{32,64}\b'),
            "sql_keywords": re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM)\b', re.IGNORECASE),
            "script_tags": re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        }
        
        # Security-related keywords
        self.security_keywords = {
            "attack": ["injection", "xss", "csrf", "exploit", "malware", "virus", "trojan"],
            "access": ["unauthorized", "forbidden", "denied", "privilege", "escalation"],
            "authentication": ["login", "logout", "password", "credential", "token", "session"],
            "network": ["scan", "probe", "brute force", "ddos", "flood"],
            "system": ["crash", "error", "exception", "failure", "timeout"]
        }
        
    def _load_embeddings_model(self):
        """Lazy loading of embeddings model."""
        if self.embeddings_model is None:
            try:
                self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
                logger.info(f"Loaded embeddings model: {self.embeddings_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embeddings model: {e}")
                raise
    
    def extract_basic_features(self, entry: LogEntry) -> Dict[str, Any]:
        """
        Extract basic statistical and structural features.
        
        Args:
            entry: Parsed log entry
            
        Returns:
            Dictionary of basic features
        """
        message = entry.message
        
        features = {
            # Temporal features
            "hour": entry.timestamp.hour,
            "day_of_week": entry.timestamp.weekday(),
            "is_weekend": entry.timestamp.weekday() >= 5,
            
            # Text length features
            "message_length": len(message),
            "word_count": len(message.split()),
            "char_count": len(message),
            "line_length": len(entry.raw_log),
            
            # Level encoding
            "log_level_info": 1 if entry.level == "INFO" else 0,
            "log_level_warn": 1 if entry.level in ["WARN", "WARNING"] else 0,
            "log_level_error": 1 if entry.level in ["ERROR", "CRITICAL", "FATAL"] else 0,
            "log_level_debug": 1 if entry.level == "DEBUG" else 0,
            
            # Character distribution
            "digit_ratio": sum(c.isdigit() for c in message) / len(message) if message else 0,
            "alpha_ratio": sum(c.isalpha() for c in message) / len(message) if message else 0,
            "upper_ratio": sum(c.isupper() for c in message) / len(message) if message else 0,
            "special_char_ratio": sum(not c.isalnum() and not c.isspace() for c in message) / len(message) if message else 0,
        }
        
        return features
    
    def extract_entropy_features(self, entry: LogEntry) -> Dict[str, Any]:
        """
        Extract entropy-based features for anomaly detection.
        
        Args:
            entry: Parsed log entry
            
        Returns:
            Dictionary of entropy features
        """
        message = entry.message
        
        # Character-level entropy
        char_counts = Counter(message.lower())
        char_probs = [count / len(message) for count in char_counts.values()]
        char_entropy = -sum(p * math.log2(p) for p in char_probs if p > 0)
        
        # Word-level entropy
        words = message.lower().split()
        word_counts = Counter(words)
        word_probs = [count / len(words) for count in word_counts.values()] if words else [0]
        word_entropy = -sum(p * math.log2(p) for p in word_probs if p > 0)
        
        return {
            "char_entropy": char_entropy,
            "word_entropy": word_entropy,
            "unique_chars": len(char_counts),
            "unique_words": len(word_counts),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
        }
    
    def extract_pattern_features(self, entry: LogEntry) -> Dict[str, Any]:
        """
        Extract pattern-based features using regex matching.
        
        Args:
            entry: Parsed log entry
            
        Returns:
            Dictionary of pattern features
        """
        message = entry.message
        
        features = {}
        
        # Pattern matching
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(message)
            features[f"has_{pattern_name}"] = 1 if matches else 0
            features[f"count_{pattern_name}"] = len(matches)
        
        # Security keyword detection
        message_lower = message.lower()
        for category, keywords in self.security_keywords.items():
            keyword_count = sum(message_lower.count(keyword) for keyword in keywords)
            features[f"security_{category}_count"] = keyword_count
            features[f"has_security_{category}"] = 1 if keyword_count > 0 else 0
        
        # Additional pattern features
        features.update({
            "has_numbers": 1 if re.search(r'\d', message) else 0,
            "has_uppercase": 1 if re.search(r'[A-Z]', message) else 0,
            "has_brackets": 1 if re.search(r'[\[\]{}()]', message) else 0,
            "has_quotes": 1 if re.search(r'["\']', message) else 0,
            "repeated_chars": len(re.findall(r'(.)\1{2,}', message)),
            "consecutive_spaces": len(re.findall(r' {2,}', message)),
        })
        
        return features
    
    def extract_semantic_features(self, entry: LogEntry) -> Dict[str, Any]:
        """
        Extract semantic features using sentence embeddings.
        
        Args:
            entry: Parsed log entry
            
        Returns:
            Dictionary of semantic features
        """
        if self.use_cache:
            message_hash = hashlib.md5(entry.message.encode()).hexdigest()
            if message_hash in self.feature_cache:
                return self.feature_cache[message_hash]
        
        self._load_embeddings_model()
        
        # Generate embeddings
        embedding = self.embeddings_model.encode([entry.message])[0]
        
        features = {
            f"embedding_{i}": float(embedding[i]) 
            for i in range(len(embedding))
        }
        
        # Additional semantic features
        features.update({
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding_mean": float(np.mean(embedding)),
            "embedding_std": float(np.std(embedding)),
            "embedding_max": float(np.max(embedding)),
            "embedding_min": float(np.min(embedding)),
        })
        
        if self.use_cache:
            self.feature_cache[message_hash] = features
        
        return features
    
    def extract_contextual_features(self, entries: List[LogEntry], current_index: int) -> Dict[str, Any]:
        """
        Extract contextual features based on surrounding log entries.
        
        Args:
            entries: List of log entries
            current_index: Index of current entry
            
        Returns:
            Dictionary of contextual features
        """
        features = {}
        window_size = 5
        
        # Get surrounding entries
        start_idx = max(0, current_index - window_size)
        end_idx = min(len(entries), current_index + window_size + 1)
        window_entries = entries[start_idx:end_idx]
        
        if len(window_entries) > 1:
            # Temporal features
            timestamps = [entry.timestamp for entry in window_entries if entry.timestamp]
            if len(timestamps) > 1:
                time_deltas = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                              for i in range(len(timestamps)-1)]
                features["avg_time_delta"] = np.mean(time_deltas)
                features["std_time_delta"] = np.std(time_deltas)
                features["max_time_delta"] = np.max(time_deltas)
            
            # Level distribution in window
            levels = [entry.level for entry in window_entries]
            level_counts = Counter(levels)
            features["error_ratio_in_window"] = level_counts.get("ERROR", 0) / len(levels)
            features["warn_ratio_in_window"] = level_counts.get("WARN", 0) / len(levels)
            
            # Source diversity
            sources = [entry.source for entry in window_entries]
            features["unique_sources_in_window"] = len(set(sources))
            
            # Message similarity
            current_entry = entries[current_index]
            similarities = []
            
            for entry in window_entries:
                if entry != current_entry:
                    # Simple Jaccard similarity
                    words1 = set(current_entry.message.lower().split())
                    words2 = set(entry.message.lower().split())
                    if words1 or words2:
                        similarity = len(words1 & words2) / len(words1 | words2)
                        similarities.append(similarity)
            
            if similarities:
                features["avg_similarity_in_window"] = np.mean(similarities)
                features["max_similarity_in_window"] = np.max(similarities)
        
        return features
    
    def extract_all_features(
        self, 
        entry: LogEntry, 
        entries: Optional[List[LogEntry]] = None,
        current_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract all available features for a log entry.
        
        Args:
            entry: Log entry to extract features from
            entries: Full list of entries for contextual features
            current_index: Index of current entry in the list
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        # Extract different types of features
        features.update(self.extract_basic_features(entry))
        features.update(self.extract_entropy_features(entry))
        features.update(self.extract_pattern_features(entry))
        
        # Add semantic features if model is available
        try:
            features.update(self.extract_semantic_features(entry))
        except Exception as e:
            logger.warning(f"Could not extract semantic features: {e}")
        
        # Add contextual features if context is provided
        if entries and current_index is not None:
            features.update(self.extract_contextual_features(entries, current_index))
        
        return features
    
    def fit_tfidf(self, messages: List[str], max_features: int = 1000):
        """
        Fit TF-IDF vectorizer on a collection of messages.
        
        Args:
            messages: List of log messages
            max_features: Maximum number of TF-IDF features
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_vectorizer.fit(messages)
        logger.info(f"Fitted TF-IDF vectorizer with {len(self.tfidf_vectorizer.vocabulary_)} features")
    
    def extract_tfidf_features(self, entry: LogEntry) -> Dict[str, float]:
        """
        Extract TF-IDF features for a log entry.
        
        Args:
            entry: Log entry
            
        Returns:
            Dictionary of TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            return {}
        
        tfidf_vector = self.tfidf_vectorizer.transform([entry.message]).toarray()[0]
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return {
            f"tfidf_{name}": float(value) 
            for name, value in zip(feature_names, tfidf_vector)
            if value > 0  # Only include non-zero features
        }
    
    def extract_features_batch(self, entries: List[LogEntry]) -> pd.DataFrame:
        """
        Extract features for a batch of log entries.
        
        Args:
            entries: List of log entries
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for i, entry in enumerate(entries):
            try:
                features = self.extract_all_features(entry, entries, i)
                features["entry_index"] = i
                features["timestamp"] = entry.timestamp
                features["raw_message"] = entry.message
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error extracting features for entry {i}: {e}")
                continue
        
        return pd.DataFrame(features_list)
    
    def calculate_similarity(self, message1: str, message2: str) -> float:
        """
        Calculate semantic similarity between two messages.
        
        Args:
            message1: First message
            message2: Second message
            
        Returns:
            Similarity score between 0 and 1
        """
        self._load_embeddings_model()
        
        embeddings = self.embeddings_model.encode([message1, message2])
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
