"""
Synthetic Log Data Generation Module

Generates synthetic log data for training the anomaly detection model.
Includes both normal and anomalous log patterns.
"""

import random
import datetime
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class LogEntry:
    timestamp: str
    level: str
    source: str
    message: str
    is_anomaly: bool


class SyntheticLogGenerator:
    """Generates synthetic log data with normal and anomalous patterns."""
    
    def __init__(self):
        self.normal_patterns = {
            "access": [
                "User {user} successfully logged in from {ip}",
                "GET /api/users/{id} - 200 OK - {response_time}ms",
                "POST /api/auth/login - 200 OK - User authenticated",
                "GET /dashboard - 200 OK - {response_time}ms",
                "User {user} accessed resource {resource}",
            ],
            "system": [
                "System health check completed successfully",
                "Database connection established",
                "Cache cleared - {items} items removed",
                "Scheduled backup completed - {size}MB",
                "Service {service} started successfully",
            ],
            "security": [
                "SSL certificate validated for {domain}",
                "Firewall rule updated for {ip}",
                "Security scan completed - no threats detected",
                "Access token refreshed for user {user}",
                "Password policy compliance check passed",
            ],
            "error": [
                "Connection timeout to external service - retrying",
                "Invalid request format - missing required field",
                "Rate limit exceeded for IP {ip} - request throttled",
                "Temporary service unavailable - maintenance mode",
                "Non-critical error in {module} - operation continued",
            ]
        }
        
        self.anomaly_patterns = {
            "access": [
                "CRITICAL: Multiple failed login attempts from {ip} - {count} attempts",
                "ALERT: Unusual access pattern detected for user {user}",
                "WARNING: Brute force attack detected from {ip}",
                "CRITICAL: Privileged escalation attempt by user {user}",
                "ALERT: Access from blacklisted IP {ip}",
            ],
            "system": [
                "CRITICAL: System memory usage at 98% - immediate action required",
                "ALERT: Unusual CPU spike detected - {cpu}% usage",
                "CRITICAL: Database connection pool exhausted",
                "ALERT: Disk space critically low - {space}% remaining",
                "CRITICAL: Service {service} crashed unexpectedly",
            ],
            "security": [
                "CRITICAL: SQL injection attempt detected in query",
                "ALERT: Malware signature detected in uploaded file",
                "CRITICAL: Unauthorized API access attempt",
                "ALERT: Suspicious file modification detected",
                "CRITICAL: Data exfiltration attempt blocked",
            ],
            "error": [
                "FATAL: Application crashed with stack overflow",
                "CRITICAL: Database corruption detected",
                "FATAL: Memory leak causing system instability",
                "CRITICAL: Cascade failure in microservices",
                "FATAL: Security breach - immediate containment required",
            ]
        }
        
        self.users = ["alice", "bob", "charlie", "diana", "eve", "frank"]
        self.ips = ["192.168.1.10", "10.0.0.5", "172.16.0.100", "203.0.113.45"]
        self.services = ["auth-service", "user-service", "payment-service", "notification-service"]
        self.domains = ["api.company.com", "app.company.com", "secure.company.com"]
        self.resources = ["/dashboard", "/api/users", "/settings", "/admin", "/reports"]
    
    def _generate_variables(self) -> Dict[str, str]:
        """Generate random variables for log templates."""
        return {
            "user": random.choice(self.users),
            "ip": random.choice(self.ips),
            "id": str(random.randint(1000, 9999)),
            "response_time": str(random.randint(50, 500)),
            "items": str(random.randint(100, 1000)),
            "size": str(random.randint(10, 100)),
            "service": random.choice(self.services),
            "domain": random.choice(self.domains),
            "count": str(random.randint(5, 50)),
            "cpu": str(random.randint(80, 99)),
            "space": str(random.randint(1, 5)),
            "module": random.choice(["auth", "payment", "user", "notification"]),
            "resource": random.choice(self.resources)
        }
    
    def _generate_timestamp(self) -> str:
        """Generate a realistic timestamp."""
        base_time = datetime.datetime.now()
        random_offset = datetime.timedelta(
            hours=random.randint(-24, 0),
            minutes=random.randint(-59, 59),
            seconds=random.randint(-59, 59)
        )
        return (base_time + random_offset).strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_log_entry(self, is_anomaly: bool = False) -> LogEntry:
        """Generate a single log entry."""
        log_type = random.choice(list(self.normal_patterns.keys()))
        
        if is_anomaly:
            template = random.choice(self.anomaly_patterns[log_type])
            level = random.choice(["WARNING", "CRITICAL", "ALERT", "FATAL"])
        else:
            template = random.choice(self.normal_patterns[log_type])
            level = random.choice(["INFO", "DEBUG", "WARN"])
        
        variables = self._generate_variables()
        message = template.format(**variables)
        
        return LogEntry(
            timestamp=self._generate_timestamp(),
            level=level,
            source=log_type,
            message=message,
            is_anomaly=is_anomaly
        )
    
    def generate_dataset(self, total_count: int, anomaly_rate: float = 0.1) -> List[LogEntry]:
        """Generate a complete dataset of log entries."""
        anomaly_count = int(total_count * anomaly_rate)
        normal_count = total_count - anomaly_count
        
        logs = []
        
        # Generate normal logs
        for _ in range(normal_count):
            logs.append(self.generate_log_entry(is_anomaly=False))
        
        # Generate anomalous logs
        for _ in range(anomaly_count):
            logs.append(self.generate_log_entry(is_anomaly=True))
        
        # Shuffle the dataset
        random.shuffle(logs)
        return logs
    
    def save_dataset(self, logs: List[LogEntry], filepath: str):
        """Save the dataset to a CSV file."""
        data = []
        for log in logs:
            data.append({
                "timestamp": log.timestamp,
                "level": log.level,
                "source": log.source,
                "message": log.message,
                "is_anomaly": int(log.is_anomaly)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Total entries: {len(logs)}")
        print(f"Normal entries: {len([l for l in logs if not l.is_anomaly])}")
        print(f"Anomalous entries: {len([l for l in logs if l.is_anomaly])}")


def main():
    """Main function for generating synthetic logs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic log data")
    parser.add_argument("--count", type=int, default=10000, help="Total number of log entries")
    parser.add_argument("--anomaly-rate", type=float, default=0.1, help="Proportion of anomalous logs")
    parser.add_argument("--output", type=str, default="data/synthetic_logs.csv", help="Output file path")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generator = SyntheticLogGenerator()
    logs = generator.generate_dataset(args.count, args.anomaly_rate)
    generator.save_dataset(logs, args.output)


if __name__ == "__main__":
    main()