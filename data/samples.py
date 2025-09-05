"""
Sample log entries for testing and demonstration.
"""

SAMPLE_NORMAL_LOGS = [
    "2024-01-15 10:00:00 [INFO] User user_0123 logged in successfully from 192.168.1.50",
    "2024-01-15 10:00:05 [INFO] HTTP GET /api/users/profile - Status: 200 - 45ms",
    "2024-01-15 10:00:10 [INFO] Database query executed: SELECT * FROM users WHERE id=123 - 12ms",
    "2024-01-15 10:00:15 [DEBUG] Processing request for user user_0123",
    "2024-01-15 10:00:20 [INFO] File uploaded: document.pdf - Size: 2048KB",
    "2024-01-15 10:00:25 [INFO] Email sent to user123@company.com - Subject: Welcome",
    "2024-01-15 10:00:30 [INFO] API request processed: /api/v1/orders - Response time: 78ms",
    "2024-01-15 10:00:35 [WARN] High memory usage detected: 85%",
]

SAMPLE_ANOMALOUS_LOGS = [
    "2024-01-15 10:01:00 [ERROR] Failed login attempt from 192.168.1.100 - Invalid credentials",
    "2024-01-15 10:01:05 [ERROR] SQL injection attempt detected: SELECT * FROM users WHERE id='1' OR '1'='1'",
    "2024-01-15 10:01:10 [CRITICAL] Unauthorized access attempt to /admin/panel",
    "2024-01-15 10:01:15 [ERROR] Malicious file upload blocked: malware.exe",
    "2024-01-15 10:01:20 [WARN] Suspicious user agent detected: sqlmap/1.0",
    "2024-01-15 10:01:25 [ERROR] XSS attack attempt: <script>alert('XSS')</script>",
    "2024-01-15 10:01:30 [CRITICAL] System compromise detected - Multiple failed logins",
    "2024-01-15 10:01:35 [ERROR] Buffer overflow attempt in authentication service",
    "2024-01-15 10:01:40 [CRITICAL] Privilege escalation detected for user guest_user",
    "2024-01-15 10:01:45 [ERROR] DDoS attack detected from 192.168.1.0/24",
]

SAMPLE_JSON_LOGS = [
    '{"timestamp": "2024-01-15T10:00:00Z", "level": "INFO", "service": "web", "message": "Request processed successfully", "user_id": "user_123", "ip": "192.168.1.50"}',
    '{"timestamp": "2024-01-15T10:00:05Z", "level": "ERROR", "service": "auth", "message": "Authentication failed", "user_id": "unknown", "ip": "192.168.1.100", "reason": "invalid_credentials"}',
    '{"timestamp": "2024-01-15T10:00:10Z", "level": "WARN", "service": "database", "message": "Slow query detected", "query_time": 5000, "table": "users"}',
    '{"timestamp": "2024-01-15T10:00:15Z", "level": "CRITICAL", "service": "security", "message": "Intrusion attempt detected", "source_ip": "203.0.113.1", "attack_type": "brute_force"}',
]

APACHE_ACCESS_LOGS = [
    '192.168.1.50 - - [15/Jan/2024:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 1024',
    '192.168.1.100 - - [15/Jan/2024:10:00:05 +0000] "POST /login HTTP/1.1" 401 512',
    '192.168.1.50 - - [15/Jan/2024:10:00:10 +0000] "GET /api/users/123 HTTP/1.1" 200 2048',
    '203.0.113.1 - - [15/Jan/2024:10:00:15 +0000] "GET /admin/config.php HTTP/1.1" 403 256',
    '192.168.1.100 - - [15/Jan/2024:10:00:20 +0000] "POST /login HTTP/1.1" 401 512',
]

def save_sample_logs(output_dir: str = "data/samples"):
    """Save sample logs to files for testing."""
    import os
    import json
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save normal logs
    with open(f"{output_dir}/normal_logs.txt", "w") as f:
        for log in SAMPLE_NORMAL_LOGS:
            f.write(log + "\n")
    
    # Save anomalous logs
    with open(f"{output_dir}/anomalous_logs.txt", "w") as f:
        for log in SAMPLE_ANOMALOUS_LOGS:
            f.write(log + "\n")
    
    # Save mixed logs
    with open(f"{output_dir}/mixed_logs.txt", "w") as f:
        for log in SAMPLE_NORMAL_LOGS + SAMPLE_ANOMALOUS_LOGS:
            f.write(log + "\n")
    
    # Save JSON logs
    with open(f"{output_dir}/json_logs.jsonl", "w") as f:
        for log in SAMPLE_JSON_LOGS:
            f.write(log + "\n")
    
    # Save Apache logs
    with open(f"{output_dir}/apache_access.log", "w") as f:
        for log in APACHE_ACCESS_LOGS:
            f.write(log + "\n")
    
    # Save labeled dataset
    labeled_data = []
    
    for log in SAMPLE_NORMAL_LOGS:
        labeled_data.append({
            "message": log.split("] ", 1)[1] if "] " in log else log,
            "raw_log": log,
            "is_anomaly": False,
            "timestamp": "2024-01-15T10:00:00Z"
        })
    
    for log in SAMPLE_ANOMALOUS_LOGS:
        labeled_data.append({
            "message": log.split("] ", 1)[1] if "] " in log else log,
            "raw_log": log,
            "is_anomaly": True,
            "timestamp": "2024-01-15T10:01:00Z"
        })
    
    with open(f"{output_dir}/labeled_dataset.json", "w") as f:
        json.dump(labeled_data, f, indent=2)
    
    print(f"Sample logs saved to {output_dir}/")


if __name__ == "__main__":
    save_sample_logs()
