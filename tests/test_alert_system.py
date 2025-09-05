"""
Unit tests for the alerting system.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.alerting.alert_system import AlertSystem, EmailAlertHandler, WebhookAlertHandler


class TestEmailAlertHandler:
    """Test cases for EmailAlertHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'sender': 'test@example.com',
            'recipients': ['admin@example.com'],
            'use_tls': True
        }
        self.handler = EmailAlertHandler(self.config)
    
    def test_create_email_body(self):
        """Test email body creation."""
        alert = {
            'timestamp': '2024-01-15T10:30:45',
            'file_path': '/var/log/test.log',
            'message': 'Test alert message',
            'detection_methods': ['BERT', 'Rule-based'],
            'details': {'confidence': 0.95}
        }
        
        body = self.handler._create_email_body(alert)
        
        # Check that key information is included
        assert 'LogLens Anomaly Detected' in body
        assert alert['timestamp'] in body
        assert alert['file_path'] in body
        assert alert['message'] in body
        assert 'BERT, Rule-based' in body
    
    @patch('smtplib.SMTP')
    def test_send_alert_success(self, mock_smtp):
        """Test successful email sending."""
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        alert = {
            'timestamp': '2024-01-15T10:30:45',
            'message': 'Test alert',
            'detection_methods': ['BERT']
        }
        
        result = self.handler.send_alert(alert)
        
        assert result is True
        mock_smtp.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.sendmail.assert_called_once()
        mock_server.quit.assert_called_once()


class TestWebhookAlertHandler:
    """Test cases for WebhookAlertHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'url': 'https://hooks.slack.com/test',
            'template': 'slack'
        }
        self.handler = WebhookAlertHandler(self.config)
    
    def test_create_slack_payload(self):
        """Test Slack payload creation."""
        alert = {
            'timestamp': '2024-01-15T10:30:45',
            'file_path': '/var/log/test.log',
            'message': 'Test alert message',
            'detection_methods': ['BERT']
        }
        
        payload = self.handler._create_payload(alert)
        
        # Check Slack-specific structure
        assert 'text' in payload
        assert 'blocks' in payload
        assert payload['text'] == "🚨 LogLens Anomaly Alert"
        assert len(payload['blocks']) == 2
    
    def test_create_teams_payload(self):
        """Test Teams payload creation."""
        self.handler.template = 'teams'
        
        alert = {
            'timestamp': '2024-01-15T10:30:45',
            'message': 'Test alert',
            'detection_methods': ['BERT']
        }
        
        payload = self.handler._create_payload(alert)
        
        # Check Teams-specific structure
        assert '@type' in payload
        assert payload['@type'] == 'MessageCard'
        assert 'sections' in payload
    
    @patch('requests.post')
    def test_send_alert_success(self, mock_post):
        """Test successful webhook sending."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        alert = {
            'timestamp': '2024-01-15T10:30:45',
            'message': 'Test alert',
            'detection_methods': ['BERT']
        }
        
        result = self.handler.send_alert(alert)
        
        assert result is True
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_alert_failure(self, mock_post):
        """Test webhook sending failure."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        
        alert = {
            'timestamp': '2024-01-15T10:30:45',
            'message': 'Test alert',
            'detection_methods': ['BERT']
        }
        
        result = self.handler.send_alert(alert)
        
        assert result is False


class TestAlertSystem:
    """Test cases for AlertSystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create minimal config
        self.config = {
            'alerting': {
                'email': {'enabled': False},
                'webhook': {'enabled': False},
                'prometheus': {'enabled': False}
            },
            'logging': {'level': 'INFO', 'format': '%(message)s'}
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = str(self.config)
            # Use temporary config for testing
            with patch('yaml.safe_load', return_value=self.config):
                self.alert_system = AlertSystem('dummy_config.yaml')
    
    def test_rate_limiting(self):
        """Test alert rate limiting."""
        alert1 = {
            'timestamp': datetime.now().isoformat(),
            'message': 'Test alert message',
            'detection_methods': ['BERT']
        }
        
        alert2 = {
            'timestamp': datetime.now().isoformat(),
            'message': 'Test alert message',  # Same message
            'detection_methods': ['BERT']
        }
        
        # First alert should not be rate limited
        assert not self.alert_system._is_rate_limited(alert1)
        
        # Update rate limiter
        self.alert_system._update_rate_limiter(alert1)
        
        # Second identical alert should be rate limited
        assert self.alert_system._is_rate_limited(alert2)
    
    def test_message_similarity(self):
        """Test message similarity calculation."""
        msg1 = "User alice failed to login"
        msg2 = "User bob failed to login"
        msg3 = "System health check completed"
        
        # Similar messages
        assert self.alert_system._messages_similar(msg1, msg2, threshold=0.5)
        
        # Dissimilar messages
        assert not self.alert_system._messages_similar(msg1, msg3, threshold=0.5)
    
    def test_alert_history(self):
        """Test alert history tracking."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': 'Test alert',
            'detection_methods': ['BERT']
        }
        
        initial_count = len(self.alert_system.alert_history)
        self.alert_system.send_alert(alert)
        
        # Should have one more alert in history
        assert len(self.alert_system.alert_history) == initial_count + 1
        
        # Recent alert should be the one we sent
        recent_alert = self.alert_system.alert_history[-1]
        assert recent_alert['message'] == alert['message']
        assert 'sent_at' in recent_alert
    
    def test_get_alert_stats(self):
        """Test alert statistics."""
        stats = self.alert_system.get_alert_stats()
        
        # Check required fields
        assert 'total_alerts' in stats
        assert 'recent_alerts_1h' in stats
        assert 'alerts_by_method' in stats
        assert 'handlers_configured' in stats
        assert 'metrics_enabled' in stats