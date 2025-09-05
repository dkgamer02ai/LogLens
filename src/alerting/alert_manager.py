"""
Alert Manager Module

This module handles real-time alerting for detected anomalies,
supporting multiple notification channels.
"""

import smtplib
import json
import requests
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import threading
from queue import Queue, Empty
import yaml
import os

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents a security alert."""
    id: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    confidence: float
    log_message: str
    anomaly_details: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    name: str
    threshold: float
    cooldown: int  # seconds
    channels: List[str]
    enabled: bool = True


class AlertChannel:
    """Base class for alert channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
    
    def send_alert(self, alert: Alert, template: str) -> bool:
        """Send alert through this channel."""
        raise NotImplementedError
    
    def format_message(self, alert: Alert, template: str) -> str:
        """Format alert message using template."""
        return template.format(
            timestamp=alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            severity=alert.severity.upper(),
            confidence=f"{alert.confidence:.1%}",
            log_message=alert.log_message,
            anomaly_details=alert.anomaly_details,
            source=alert.source,
            alert_id=alert.id
        )


class EmailChannel(AlertChannel):
    """Email notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_host = config.get("smtp_host")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.recipients = config.get("recipients", [])
    
    def send_alert(self, alert: Alert, template: str) -> bool:
        """Send email alert."""
        if not self.enabled or not self.recipients:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.username
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"🚨 LogLens Security Alert - {alert.severity.upper()} Severity"
            
            # Format message body
            body = self.format_message(alert, template)
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class SlackChannel(AlertChannel):
    """Slack notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#security-alerts")
    
    def send_alert(self, alert: Alert, template: str) -> bool:
        """Send Slack alert."""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            # Format message
            message = self.format_message(alert, template)
            
            # Create Slack payload
            payload = {
                "channel": self.channel,
                "username": "LogLens",
                "icon_emoji": ":warning:",
                "text": message,
                "attachments": [
                    {
                        "color": self._get_color(alert.severity),
                        "fields": [
                            {
                                "title": "Alert ID",
                                "value": alert.id,
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Confidence",
                                "value": f"{alert.confidence:.1%}",
                                "short": True
                            }
                        ],
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert.id}")
                return True
            else:
                logger.error(f"Slack webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _get_color(self, severity: str) -> str:
        """Get color based on severity."""
        colors = {
            "low": "#36a64f",      # green
            "medium": "#ff9900",   # orange
            "high": "#ff6600",     # red-orange
            "critical": "#ff0000"  # red
        }
        return colors.get(severity.lower(), "#cccccc")


class WebhookChannel(AlertChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url")
        self.headers = config.get("headers", {})
        self.method = config.get("method", "POST")
    
    def send_alert(self, alert: Alert, template: str) -> bool:
        """Send webhook alert."""
        if not self.enabled or not self.url:
            return False
        
        try:
            # Create payload
            payload = {
                "alert": asdict(alert),
                "message": self.format_message(alert, template),
                "timestamp": alert.timestamp.isoformat()
            }
            
            # Convert datetime objects to strings
            payload["alert"]["timestamp"] = alert.timestamp.isoformat()
            
            # Send webhook
            response = requests.request(
                method=self.method,
                url=self.url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code < 400:
                logger.info(f"Webhook alert sent for {alert.id}")
                return True
            else:
                logger.error(f"Webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AlertManager:
    """
    Manages alert processing, deduplication, and routing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.channels = self._initialize_channels()
        self.rules = self._load_rules()
        self.templates = self._load_templates()
        
        # Alert state management
        self.alert_queue = Queue()
        self.alert_history: Dict[str, datetime] = {}  # For cooldown tracking
        self.active_alerts: Set[str] = set()
        self.alert_counts = defaultdict(int)
        
        # Background processing
        self.running = False
        self.worker_thread = None
        
        logger.info("Alert Manager initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load alerting configuration."""
        if config_path is None:
            config_path = "config/alerting.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config_str = yaml.dump(config)
            for key, value in os.environ.items():
                config_str = config_str.replace(f"${{{key}}}", value)
            
            return yaml.safe_load(config_str)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {"alerting": {"channels": {}, "rules": {}, "templates": {}}}
    
    def _initialize_channels(self) -> Dict[str, AlertChannel]:
        """Initialize alert channels."""
        channels = {}
        channel_configs = self.config.get("alerting", {}).get("channels", {})
        
        for name, config in channel_configs.items():
            try:
                if name == "email":
                    channels[name] = EmailChannel(config)
                elif name == "slack":
                    channels[name] = SlackChannel(config)
                elif name == "webhook":
                    channels[name] = WebhookChannel(config)
                else:
                    logger.warning(f"Unknown channel type: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize channel {name}: {e}")
        
        logger.info(f"Initialized {len(channels)} alert channels")
        return channels
    
    def _load_rules(self) -> Dict[str, AlertRule]:
        """Load alert rules."""
        rules = {}
        rule_configs = self.config.get("alerting", {}).get("rules", {})
        
        for name, config in rule_configs.items():
            rules[name] = AlertRule(
                name=name,
                threshold=config.get("threshold", 0.5),
                cooldown=config.get("cooldown", 300),
                channels=config.get("channels", []),
                enabled=config.get("enabled", True)
            )
        
        return rules
    
    def _load_templates(self) -> Dict[str, str]:
        """Load message templates."""
        return self.config.get("alerting", {}).get("templates", {
            "email": "Security alert detected at {timestamp}\nSeverity: {severity}\nDetails: {anomaly_details}",
            "slack": "🚨 Security Alert: {severity} severity detected at {timestamp}",
            "webhook": "Alert: {anomaly_details}"
        })
    
    def create_alert(
        self,
        log_message: str,
        anomaly_score: float,
        anomaly_details: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert."""
        severity = self._calculate_severity(anomaly_score)
        alert_id = self._generate_alert_id(log_message, source)
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            confidence=anomaly_score,
            log_message=log_message,
            anomaly_details=anomaly_details,
            source=source,
            metadata=metadata or {}
        )
        
        return alert
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate severity based on anomaly score."""
        if anomaly_score >= 0.9:
            return "critical"
        elif anomaly_score >= 0.7:
            return "high"
        elif anomaly_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_alert_id(self, log_message: str, source: str) -> str:
        """Generate unique alert ID."""
        import hashlib
        content = f"{log_message}:{source}:{datetime.now().strftime('%Y%m%d%H')}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on rules and cooldowns."""
        # Check if matching rule exists
        matching_rule = None
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            if alert.confidence >= rule.threshold:
                matching_rule = rule
                break
        
        if not matching_rule:
            return False
        
        # Check cooldown
        if alert.id in self.alert_history:
            last_sent = self.alert_history[alert.id]
            if datetime.now() - last_sent < timedelta(seconds=matching_rule.cooldown):
                logger.debug(f"Alert {alert.id} in cooldown period")
                return False
        
        return True
    
    def queue_alert(self, alert: Alert):
        """Queue alert for processing."""
        self.alert_queue.put(alert)
        logger.debug(f"Queued alert {alert.id}")
    
    def process_alert(self, alert: Alert):
        """Process a single alert."""
        if not self.should_send_alert(alert):
            return
        
        # Find matching rule
        matching_rule = None
        for rule in self.rules.values():
            if rule.enabled and alert.confidence >= rule.threshold:
                matching_rule = rule
                break
        
        if not matching_rule:
            return
        
        logger.info(f"Processing alert {alert.id} - {alert.severity} severity")
        
        # Send through configured channels
        success_count = 0
        for channel_name in matching_rule.channels:
            if channel_name in self.channels:
                template = self.templates.get(channel_name, self.templates.get("default", ""))
                success = self.channels[channel_name].send_alert(alert, template)
                if success:
                    success_count += 1
        
        if success_count > 0:
            # Update state
            self.alert_history[alert.id] = alert.timestamp
            self.active_alerts.add(alert.id)
            self.alert_counts[alert.severity] += 1
            
            logger.info(f"Alert {alert.id} sent through {success_count} channels")
        else:
            logger.warning(f"Failed to send alert {alert.id} through any channel")
    
    def start(self):
        """Start the alert processing worker."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Alert Manager started")
    
    def stop(self):
        """Stop the alert processing worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Alert Manager stopped")
    
    def _worker_loop(self):
        """Main worker loop for processing alerts."""
        while self.running:
            try:
                # Get alert from queue with timeout
                alert = self.alert_queue.get(timeout=1)
                self.process_alert(alert)
                self.alert_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def send_alert_sync(
        self,
        log_message: str,
        anomaly_score: float,
        anomaly_details: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send alert synchronously."""
        alert = self.create_alert(log_message, anomaly_score, anomaly_details, source, metadata)
        self.process_alert(alert)
    
    def send_alert_async(
        self,
        log_message: str,
        anomaly_score: float,
        anomaly_details: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send alert asynchronously."""
        alert = self.create_alert(log_message, anomaly_score, anomaly_details, source, metadata)
        self.queue_alert(alert)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return {
            "total_alerts": sum(self.alert_counts.values()),
            "alerts_by_severity": dict(self.alert_counts),
            "active_alerts": len(self.active_alerts),
            "queue_size": self.alert_queue.qsize(),
            "running": self.running,
            "channels_configured": len(self.channels)
        }
    
    def test_channels(self) -> Dict[str, bool]:
        """Test all configured channels."""
        results = {}
        
        test_alert = Alert(
            id="test_alert",
            timestamp=datetime.now(),
            severity="low",
            confidence=0.6,
            log_message="This is a test alert from LogLens",
            anomaly_details="Testing alert system configuration",
            source="test",
            metadata={"test": True}
        )
        
        for name, channel in self.channels.items():
            template = self.templates.get(name, "Test alert: {anomaly_details}")
            try:
                success = channel.send_alert(test_alert, template)
                results[name] = success
            except Exception as e:
                logger.error(f"Channel {name} test failed: {e}")
                results[name] = False
        
        return results
