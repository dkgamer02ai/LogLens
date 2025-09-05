"""
Embeddings Module

Handles creation of semantic embeddings for log messages using pre-trained models.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import pickle
import os


class LogEmbeddingsGenerator:
    """Generate semantic embeddings for log messages."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a pre-trained sentence transformer model."""
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        self.load_model()
    
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            # Fallback to a simpler model
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
    
    def generate_embeddings(self, messages: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of log messages."""
        if not messages:
            return np.array([])
        
        # Check cache first
        cached_embeddings = []
        uncached_messages = []
        uncached_indices = []
        
        for i, message in enumerate(messages):
            if message in self.embeddings_cache:
                cached_embeddings.append((i, self.embeddings_cache[message]))
            else:
                uncached_messages.append(message)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached messages
        if uncached_messages:
            new_embeddings = self.model.encode(
                uncached_messages, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Cache new embeddings
            for message, embedding in zip(uncached_messages, new_embeddings):
                self.embeddings_cache[message] = embedding
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings in original order
        all_embeddings = [None] * len(messages)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for i, idx in enumerate(uncached_indices):
            all_embeddings[idx] = new_embeddings[i]
        
        return np.array(all_embeddings)
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix for embeddings."""
        return cosine_similarity(embeddings)
    
    def find_similar_logs(self, query_embedding: np.ndarray, 
                         reference_embeddings: np.ndarray,
                         threshold: float = 0.8) -> List[Tuple[int, float]]:
        """Find logs similar to a query embedding."""
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            reference_embeddings
        )[0]
        
        similar_indices = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                similar_indices.append((i, similarity))
        
        # Sort by similarity (descending)
        similar_indices.sort(key=lambda x: x[1], reverse=True)
        return similar_indices
    
    def cluster_logs(self, embeddings: np.ndarray, eps: float = 0.3, 
                    min_samples: int = 2) -> Dict[int, List[int]]:
        """Cluster log embeddings using DBSCAN."""
        # Use 1 - cosine similarity as distance metric
        distances = 1 - cosine_similarity(embeddings)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distances)
        
        # Group by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        return clusters
    
    def save_cache(self, filepath: str):
        """Save embeddings cache to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        print(f"Embeddings cache saved to {filepath}")
    
    def load_cache(self, filepath: str):
        """Load embeddings cache from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"Embeddings cache loaded from {filepath}")
        else:
            print(f"Cache file {filepath} not found")


class SemanticLogAnalyzer:
    """Analyze logs using semantic embeddings."""
    
    def __init__(self, embeddings_generator: LogEmbeddingsGenerator):
        self.embeddings_generator = embeddings_generator
        self.normal_embeddings = None
        self.anomaly_threshold = 0.7
    
    def fit_normal_patterns(self, normal_logs: List[str]):
        """Fit on normal log patterns to establish baseline."""
        print("Generating embeddings for normal log patterns...")
        self.normal_embeddings = self.embeddings_generator.generate_embeddings(normal_logs)
        print(f"Generated {len(self.normal_embeddings)} normal pattern embeddings")
    
    def detect_anomaly(self, log_message: str) -> Tuple[bool, float]:
        """Detect if a log message is anomalous based on semantic similarity."""
        if self.normal_embeddings is None:
            raise ValueError("Must fit normal patterns first")
        
        # Generate embedding for the query message
        query_embedding = self.embeddings_generator.generate_embeddings([log_message])
        
        # Find similarity to normal patterns
        similarities = cosine_similarity(query_embedding, self.normal_embeddings)[0]
        max_similarity = np.max(similarities)
        
        # Anomaly if similarity to all normal patterns is below threshold
        is_anomaly = max_similarity < self.anomaly_threshold
        
        return is_anomaly, max_similarity
    
    def batch_detect_anomalies(self, log_messages: List[str]) -> List[Tuple[bool, float]]:
        """Detect anomalies in a batch of log messages."""
        if self.normal_embeddings is None:
            raise ValueError("Must fit normal patterns first")
        
        # Generate embeddings for all messages
        query_embeddings = self.embeddings_generator.generate_embeddings(log_messages)
        
        results = []
        for query_embedding in query_embeddings:
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                self.normal_embeddings
            )[0]
            max_similarity = np.max(similarities)
            is_anomaly = max_similarity < self.anomaly_threshold
            results.append((is_anomaly, max_similarity))
        
        return results
    
    def set_anomaly_threshold(self, threshold: float):
        """Set the anomaly detection threshold."""
        self.anomaly_threshold = threshold
        print(f"Anomaly threshold set to {threshold}")
    
    def analyze_log_patterns(self, log_messages: List[str]) -> Dict:
        """Analyze patterns in log messages."""
        embeddings = self.embeddings_generator.generate_embeddings(log_messages)
        
        # Cluster logs
        clusters = self.embeddings_generator.cluster_logs(embeddings)
        
        # Calculate statistics
        analysis = {
            'total_logs': len(log_messages),
            'num_clusters': len([c for c in clusters.keys() if c != -1]),  # Exclude noise cluster
            'noise_logs': len(clusters.get(-1, [])),
            'clusters': {}
        }
        
        for cluster_id, log_indices in clusters.items():
            if cluster_id == -1:
                continue  # Skip noise cluster
            
            cluster_embeddings = embeddings[log_indices]
            cluster_messages = [log_messages[i] for i in log_indices]
            
            # Calculate cluster cohesion (average pairwise similarity)
            if len(cluster_embeddings) > 1:
                similarity_matrix = cosine_similarity(cluster_embeddings)
                # Get upper triangle (excluding diagonal)
                upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                cohesion = np.mean(upper_triangle)
            else:
                cohesion = 1.0
            
            analysis['clusters'][cluster_id] = {
                'size': len(log_indices),
                'cohesion': cohesion,
                'sample_messages': cluster_messages[:3]  # First 3 messages as samples
            }
        
        return analysis


def main():
    """Example usage of the embeddings module."""
    # Sample log messages
    sample_logs = [
        "User alice successfully logged in from 192.168.1.10",
        "User bob successfully logged in from 192.168.1.11", 
        "GET /api/users/1234 - 200 OK - 150ms",
        "GET /api/users/5678 - 200 OK - 200ms",
        "CRITICAL: Multiple failed login attempts from 203.0.113.45",
        "ALERT: Brute force attack detected from 203.0.113.45",
        "System health check completed successfully",
        "Database connection established"
    ]
    
    print("Embeddings Example:")
    print("=" * 50)
    
    # Initialize embeddings generator
    embeddings_gen = LogEmbeddingsGenerator()
    
    # Generate embeddings
    embeddings = embeddings_gen.generate_embeddings(sample_logs)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Semantic analysis
    analyzer = SemanticLogAnalyzer(embeddings_gen)
    
    # Use first 4 logs as normal patterns
    normal_logs = sample_logs[:4]
    analyzer.fit_normal_patterns(normal_logs)
    
    # Test anomaly detection on remaining logs
    test_logs = sample_logs[4:]
    for log in test_logs:
        is_anomaly, similarity = analyzer.detect_anomaly(log)
        print(f"Log: {log}")
        print(f"Anomaly: {is_anomaly}, Max Similarity: {similarity:.3f}")
        print()
    
    # Analyze patterns
    analysis = analyzer.analyze_log_patterns(sample_logs)
    print("Pattern Analysis:")
    print(f"Total logs: {analysis['total_logs']}")
    print(f"Clusters found: {analysis['num_clusters']}")
    print(f"Noise logs: {analysis['noise_logs']}")


if __name__ == "__main__":
    main()