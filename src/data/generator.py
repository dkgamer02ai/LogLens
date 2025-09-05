"""
Synthetic Log Data Generator

This module generates synthetic log datasets for training and testing
the anomaly detection model.
"""

import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogTemplate:
    """Template for generating log entries."""
    level: str
    message_template: str
    is_anomaly: bool
    weight: float = 1.0


class SyntheticLogGenerator:
    """
    Generator for synthetic log datasets with normal and anomalous patterns.
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Normal log templates
        self.normal_templates = [
            LogTemplate("INFO", "User {user_id} logged in successfully from {ip}", False, 0.3),
            LogTemplate("INFO", "HTTP {method} {path} - Status: {status} - {size}ms", False, 0.25),
            LogTemplate("INFO", "Database query executed: SELECT * FROM {table} - {time}ms", False, 0.15),
            LogTemplate("INFO", "API request processed: {endpoint} - Response time: {time}ms", False, 0.1),
            LogTemplate("DEBUG", "Processing request for user {user_id}", False, 0.05),
            LogTemplate("INFO", "File uploaded: {filename} - Size: {size}KB", False, 0.05),
            LogTemplate("INFO", "Email sent to {email} - Subject: {subject}", False, 0.05),
            LogTemplate("WARN", "High memory usage detected: {percentage}%", False, 0.05),
        ]
        
        # Anomalous log templates
        self.anomaly_templates = [
            LogTemplate("ERROR", "Failed login attempt from {ip} - Invalid credentials", True, 0.2),
            LogTemplate("ERROR", "SQL injection attempt detected: {query}", True, 0.15),
            LogTemplate("CRITICAL", "Unauthorized access attempt to {resource}", True, 0.15),
            LogTemplate("ERROR", "Malicious file upload blocked: {filename}", True, 0.1),
            LogTemplate("WARN", "Suspicious user agent detected: {user_agent}", True, 0.1),
            LogTemplate("ERROR", "XSS attack attempt: {payload}", True, 0.1),
            LogTemplate("CRITICAL", "System compromise detected - Multiple failed logins", True, 0.05),
            LogTemplate("ERROR", "Buffer overflow attempt in {service}", True, 0.05),
            LogTemplate("CRITICAL", "Privilege escalation detected for user {user_id}", True, 0.05),
            LogTemplate("ERROR", "DDoS attack detected from {ip_range}", True, 0.05),
        ]
        
        # Data for placeholder substitution
        self.data_sources = {
            "user_id": [f"user_{i:04d}" for i in range(1, 1000)],
            "ip": self._generate_ips(),
            "method": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            "path": ["/api/users", "/login", "/dashboard", "/admin", "/upload", "/download"],
            "status": [200, 201, 400, 401, 403, 404, 500],
            "table": ["users", "orders", "products", "logs", "sessions"],
            "endpoint": ["/api/v1/users", "/api/v1/orders", "/api/v1/auth"],
            "filename": ["document.pdf", "image.jpg", "script.js", "malware.exe"],
            "email": [f"user{i}@company.com" for i in range(1, 100)],
            "subject": ["Welcome", "Password Reset", "Account Update", "Security Alert"],
            "resource": ["/admin/panel", "/secret/data", "/config/settings"],
            "service": ["authentication", "database", "file_handler", "web_server"],
            "ip_range": ["192.168.1.0/24", "10.0.0.0/16", "172.16.0.0/12"],
            "user_agent": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "sqlmap/1.0",
                "<script>alert('xss')</script>",
                "Nikto/2.1.6"
            ],
            "query": [
                "SELECT * FROM users WHERE id = '1' OR '1'='1'",
                "'; DROP TABLE users; --",
                "UNION SELECT password FROM admin_users"
            ],
            "payload": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>"
            ]
        }
    
    def _generate_ips(self) -> List[str]:
        """Generate a list of IP addresses."""
        ips = []
        
        # Normal IPs
        for _ in range(100):
            ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            ips.append(ip)
        
        # Suspicious IPs
        suspicious_ips = [
            "192.168.1.100",  # Known attacker
            "10.0.0.50",      # Internal suspicious
            "203.0.113.1",    # External attacker
            "198.51.100.1"    # Another external
        ]
        ips.extend(suspicious_ips)
        
        return ips
    
    def _fill_template(self, template: LogTemplate) -> Tuple[str, bool]:
        """
        Fill a template with random data.
        
        Args:
            template: LogTemplate to fill
            
        Returns:
            Tuple of (filled_message, is_anomaly)
        """
        message = template.message_template
        
        # Find all placeholders in the template
        import re
        placeholders = re.findall(r'\{(\w+)\}', message)
        
        # Replace each placeholder with random data
        for placeholder in placeholders:
            if placeholder in self.data_sources:
                if placeholder in ["size", "time", "percentage"]:
                    # Generate numeric values
                    if placeholder == "size":
                        value = random.randint(1, 10000)
                    elif placeholder == "time":
                        value = random.randint(10, 5000)
                    else:  # percentage
                        value = random.randint(50, 95)
                else:
                    value = random.choice(self.data_sources[placeholder])
                
                message = message.replace(f"{{{placeholder}}}", str(value))
        
        return message, template.is_anomaly
    
    def generate_log_entry(self, timestamp: datetime, force_anomaly: bool = False) -> Dict[str, Any]:
        """
        Generate a single log entry.
        
        Args:
            timestamp: Timestamp for the log entry
            force_anomaly: Force generation of anomalous entry
            
        Returns:
            Dictionary representing log entry
        """
        if force_anomaly:
            template = random.choices(
                self.anomaly_templates,
                weights=[t.weight for t in self.anomaly_templates]
            )[0]
        else:
            # Choose between normal and anomaly based on weights
            all_templates = self.normal_templates + self.anomaly_templates
            template = random.choices(
                all_templates,
                weights=[t.weight for t in all_templates]
            )[0]
        
        message, is_anomaly = self._fill_template(template)
        
        return {
            "timestamp": timestamp.isoformat(),
            "level": template.level,
            "source": random.choice(["web_server", "database", "auth_service", "api_gateway"]),
            "message": message,
            "is_anomaly": is_anomaly,
            "raw_log": f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} [{template.level}] {message}"
        }
    
    def generate_time_series(
        self, 
        start_time: datetime,
        duration_hours: int = 24,
        base_rate: int = 100  # logs per hour
    ) -> List[datetime]:
        """
        Generate realistic timestamps for log entries.
        
        Args:
            start_time: Starting timestamp
            duration_hours: Duration in hours
            base_rate: Base number of logs per hour
            
        Returns:
            List of timestamps
        """
        timestamps = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        while current_time < end_time:
            # Vary the rate based on time of day (more activity during business hours)
            hour = current_time.hour
            if 9 <= hour <= 17:  # Business hours
                rate_multiplier = 1.5
            elif 22 <= hour or hour <= 6:  # Night hours
                rate_multiplier = 0.3
            else:
                rate_multiplier = 1.0
            
            # Add some randomness
            rate = int(base_rate * rate_multiplier * random.uniform(0.5, 1.5))
            
            # Generate logs for this hour
            for _ in range(rate):
                # Add random seconds within the hour
                random_seconds = random.randint(0, 3599)
                log_time = current_time + timedelta(seconds=random_seconds)
                timestamps.append(log_time)
            
            current_time += timedelta(hours=1)
        
        return sorted(timestamps)
    
    def generate_attack_scenario(self, start_time: datetime, attack_type: str = "brute_force") -> List[Dict[str, Any]]:
        """
        Generate a specific attack scenario.
        
        Args:
            start_time: When the attack starts
            attack_type: Type of attack to simulate
            
        Returns:
            List of log entries representing the attack
        """
        logs = []
        attacker_ip = "192.168.1.100"
        
        if attack_type == "brute_force":
            # Simulate brute force attack over 10 minutes
            for i in range(50):
                timestamp = start_time + timedelta(seconds=i * 12)
                
                if i < 45:  # Failed attempts
                    log = {
                        "timestamp": timestamp.isoformat(),
                        "level": "ERROR",
                        "source": "auth_service",
                        "message": f"Failed login attempt from {attacker_ip} - Invalid credentials for user admin",
                        "is_anomaly": True,
                        "raw_log": f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Failed login attempt from {attacker_ip}"
                    }
                else:  # Successful breach
                    log = {
                        "timestamp": timestamp.isoformat(),
                        "level": "CRITICAL",
                        "source": "auth_service",
                        "message": f"Successful login for admin from {attacker_ip} after multiple failed attempts",
                        "is_anomaly": True,
                        "raw_log": f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} [CRITICAL] Admin login from {attacker_ip}"
                    }
                
                logs.append(log)
        
        elif attack_type == "sql_injection":
            # SQL injection attempts
            for i in range(10):
                timestamp = start_time + timedelta(seconds=i * 30)
                queries = [
                    "SELECT * FROM users WHERE id = '1' OR '1'='1'",
                    "'; DROP TABLE users; --",
                    "UNION SELECT password FROM admin_users"
                ]
                
                log = {
                    "timestamp": timestamp.isoformat(),
                    "level": "ERROR",
                    "source": "web_server",
                    "message": f"SQL injection attempt detected: {random.choice(queries)}",
                    "is_anomaly": True,
                    "raw_log": f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] SQL injection from {attacker_ip}"
                }
                logs.append(log)
        
        return logs
    
    def generate_dataset(
        self, 
        num_samples: int = 10000,
        anomaly_ratio: float = 0.1,
        include_scenarios: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_samples: Total number of log entries
            anomaly_ratio: Ratio of anomalous entries
            include_scenarios: Whether to include attack scenarios
            
        Returns:
            List of log entries
        """
        logger.info(f"Generating {num_samples} synthetic log entries...")
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=7)
        timestamps = self.generate_time_series(start_time, duration_hours=168, base_rate=num_samples//168)
        
        # Ensure we have enough timestamps
        while len(timestamps) < num_samples:
            additional_time = start_time + timedelta(hours=random.randint(0, 168))
            timestamps.append(additional_time)
        
        timestamps = sorted(timestamps[:num_samples])
        
        logs = []
        num_anomalies = int(num_samples * anomaly_ratio)
        anomaly_indices = set(random.sample(range(num_samples), num_anomalies))
        
        # Generate individual log entries
        for i, timestamp in enumerate(timestamps):
            force_anomaly = i in anomaly_indices
            log_entry = self.generate_log_entry(timestamp, force_anomaly)
            logs.append(log_entry)
        
        # Add attack scenarios if requested
        if include_scenarios:
            scenario_start_times = [
                start_time + timedelta(hours=24),   # Day 2
                start_time + timedelta(hours=72),   # Day 4
                start_time + timedelta(hours=120),  # Day 6
            ]
            
            for i, scenario_time in enumerate(scenario_start_times):
                attack_type = ["brute_force", "sql_injection", "brute_force"][i]
                scenario_logs = self.generate_attack_scenario(scenario_time, attack_type)
                logs.extend(scenario_logs)
        
        # Sort by timestamp
        logs.sort(key=lambda x: x["timestamp"])
        
        # Add some metadata
        normal_count = sum(1 for log in logs if not log["is_anomaly"])
        anomaly_count = sum(1 for log in logs if log["is_anomaly"])
        
        logger.info(f"Generated dataset: {len(logs)} total entries")
        logger.info(f"Normal entries: {normal_count} ({normal_count/len(logs)*100:.1f}%)")
        logger.info(f"Anomalous entries: {anomaly_count} ({anomaly_count/len(logs)*100:.1f}%)")
        
        return logs
    
    def save_dataset(self, logs: List[Dict[str, Any]], filepath: str):
        """Save dataset to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Dataset saved to {filepath}")
    
    def save_dataset_csv(self, logs: List[Dict[str, Any]], filepath: str):
        """Save dataset to CSV file."""
        import pandas as pd
        
        df = pd.DataFrame(logs)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Dataset saved to {filepath}")


def main():
    """Command-line interface for generating datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic log dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--anomaly-ratio", type=float, default=0.1, help="Ratio of anomalous samples")
    parser.add_argument("--output", type=str, default="data/synthetic_logs.json", help="Output file path")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generator = SyntheticLogGenerator(seed=args.seed)
    logs = generator.generate_dataset(
        num_samples=args.samples,
        anomaly_ratio=args.anomaly_ratio
    )
    
    if args.format == "json":
        generator.save_dataset(logs, args.output)
    else:
        generator.save_dataset_csv(logs, args.output)


if __name__ == "__main__":
    main()
