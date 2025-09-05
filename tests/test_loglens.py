"""
Test suite for LogLens anomaly detection system.
"""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path

from src.preprocessing.log_parser import LogParser, LogEntry
from src.preprocessing.feature_extractor import FeatureExtractor
from src.data.generator import SyntheticLogGenerator
from src.models.bert_classifier import BERTAnomalyDetector
from src.alerting.alert_manager import AlertManager


class TestLogParser:
    """Test cases for log parser."""
    
    def test_parse_application_log(self):
        """Test parsing application log format."""
        parser = LogParser()
        log_line = "2024-01-01 10:00:00 [INFO] User user_0001 logged in successfully"
        
        entry = parser.parse_line(log_line)
        
        assert entry is not None
        assert entry.level == "INFO"
        assert "user_0001" in entry.message
        assert entry.source is not None
    
    def test_parse_json_log(self):
        """Test parsing JSON log format."""
        parser = LogParser()
        log_data = {
            "timestamp": "2024-01-01T10:00:00Z",
            "level": "ERROR",
            "message": "Database connection failed",
            "service": "api"
        }
        log_line = json.dumps(log_data)
        
        entry = parser.parse_line(log_line)
        
        assert entry is not None
        assert entry.level == "ERROR"
        assert entry.message == "Database connection failed"
        assert entry.source == "api"
    
    def test_parse_batch(self):
        """Test parsing multiple log lines."""
        parser = LogParser()
        lines = [
            "2024-01-01 10:00:00 [INFO] Normal log entry",
            "2024-01-01 10:00:01 [ERROR] Error occurred",
            "invalid log line",
            ""
        ]
        
        entries = parser.parse_batch(lines)
        
        assert len(entries) == 2  # Only valid entries
        assert entries[0].level == "INFO"
        assert entries[1].level == "ERROR"


class TestFeatureExtractor:
    """Test cases for feature extractor."""
    
    def test_basic_features(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            level="ERROR",
            source="test",
            message="Failed login attempt from 192.168.1.100",
            raw_log="test log",
            metadata={}
        )
        
        features = extractor.extract_basic_features(entry)
        
        assert "hour" in features
        assert "message_length" in features
        assert "log_level_error" in features
        assert features["log_level_error"] == 1
        assert features["hour"] == 10
    
    def test_pattern_features(self):
        """Test pattern-based feature extraction."""
        extractor = FeatureExtractor()
        entry = LogEntry(
            timestamp=datetime.now(),
            level="ERROR",
            source="test",
            message="SQL injection attempt: SELECT * FROM users WHERE id='1' OR '1'='1'",
            raw_log="test",
            metadata={}
        )
        
        features = extractor.extract_pattern_features(entry)
        
        assert "has_sql_keywords" in features
        assert features["has_sql_keywords"] == 1
        assert "security_attack_count" in features
    
    def test_entropy_features(self):
        """Test entropy feature extraction."""
        extractor = FeatureExtractor()
        entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            source="test",
            message="Regular log message with normal text",
            raw_log="test",
            metadata={}
        )
        
        features = extractor.extract_entropy_features(entry)
        
        assert "char_entropy" in features
        assert "word_entropy" in features
        assert "unique_chars" in features
        assert features["char_entropy"] > 0


class TestSyntheticLogGenerator:
    """Test cases for synthetic data generator."""
    
    def test_generate_single_entry(self):
        """Test generating a single log entry."""
        generator = SyntheticLogGenerator(seed=42)
        timestamp = datetime.now()
        
        entry = generator.generate_log_entry(timestamp)
        
        assert "timestamp" in entry
        assert "level" in entry
        assert "message" in entry
        assert "is_anomaly" in entry
        assert isinstance(entry["is_anomaly"], bool)
    
    def test_generate_dataset(self):
        """Test generating a full dataset."""
        generator = SyntheticLogGenerator(seed=42)
        
        logs = generator.generate_dataset(num_samples=100, anomaly_ratio=0.2)
        
        assert len(logs) >= 100  # May have additional scenario logs
        
        # Check anomaly ratio is approximately correct
        anomalies = sum(1 for log in logs[:100] if log["is_anomaly"])
        assert 15 <= anomalies <= 25  # Allow some variance
    
    def test_attack_scenario(self):
        """Test generating attack scenarios."""
        generator = SyntheticLogGenerator(seed=42)
        start_time = datetime.now()
        
        logs = generator.generate_attack_scenario(start_time, "brute_force")
        
        assert len(logs) > 0
        assert all(log["is_anomaly"] for log in logs)
        assert any("failed login" in log["message"].lower() for log in logs)


class TestBERTAnomalyDetector:
    """Test cases for BERT anomaly detector."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return BERTAnomalyDetector(model_name="distilbert-base-uncased", max_length=128)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "distilbert-base-uncased"
        assert model.max_length == 128
        assert model.tokenizer is not None
        assert model.bert is not None
    
    def test_predict_single(self, model):
        """Test single prediction."""
        text = "User logged in successfully"
        
        result = model.predict_single(text)
        
        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "confidence" in result
        assert isinstance(result["is_anomaly"], bool)
        assert 0 <= result["anomaly_score"] <= 1
        assert 0 <= result["confidence"] <= 1
    
    def test_predict_batch(self, model):
        """Test batch prediction."""
        texts = [
            "User logged in successfully",
            "SQL injection attempt detected",
            "Normal database query executed"
        ]
        
        predictions, scores = model.predict(texts)
        
        assert len(predictions) == 3
        assert len(scores) == 3
        assert all(isinstance(p, (int, bool)) for p in predictions)
        assert all(0 <= s <= 1 for s in scores)


class TestAlertManager:
    """Test cases for alert manager."""
    
    def test_alert_creation(self):
        """Test creating alerts."""
        # Create a minimal config
        config = {
            "alerting": {
                "channels": {},
                "rules": {
                    "test_rule": {
                        "threshold": 0.5,
                        "cooldown": 300,
                        "channels": []
                    }
                },
                "templates": {}
            }
        }
        
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            alert_manager = AlertManager(config_path)
            
            alert = alert_manager.create_alert(
                log_message="Test log message",
                anomaly_score=0.8,
                anomaly_details="Test anomaly",
                source="test"
            )
            
            assert alert.anomaly_score == 0.8
            assert alert.severity == "high"  # 0.8 should be high severity
            assert alert.log_message == "Test log message"
            assert alert.source == "test"
            
        finally:
            Path(config_path).unlink()  # Clean up temp file


@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests for the API."""
    
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from src.api.main import create_app
        from httpx import AsyncClient
        
        app = create_app()
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__])
