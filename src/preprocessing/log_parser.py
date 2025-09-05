"""
Log Parser Module

This module handles parsing and preprocessing of various log formats
for the anomaly detection pipeline.
"""

import re
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured representation of a log entry."""
    timestamp: datetime
    level: str
    source: str
    message: str
    raw_log: str
    metadata: Dict[str, Any]


class LogParser:
    """
    Universal log parser supporting multiple log formats.
    
    Supports common formats like:
    - Apache/Nginx access logs
    - Syslog format
    - JSON logs
    - Custom application logs
    """
    
    def __init__(self):
        self.patterns = {
            "apache_access": re.compile(
                r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
                r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" '
                r'(?P<status>\d+) (?P<size>\S+)'
            ),
            "syslog": re.compile(
                r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+) '
                r'(?P<hostname>\S+) '
                r'(?P<process>\S+): '
                r'(?P<message>.*)'
            ),
            "application": re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\.]?\d*)\s+'
                r'\[?(?P<level>DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\]?\s*'
                r'(?:\[(?P<thread>[^\]]+)\])?\s*'
                r'(?P<logger>\S+)?\s*[-:]?\s*'
                r'(?P<message>.*)'
            ),
            "nginx_access": re.compile(
                r'(?P<ip>\S+) - \S+ \[(?P<timestamp>[^\]]+)\] '
                r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" '
                r'(?P<status>\d+) (?P<size>\d+) '
                r'"(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"'
            ),
            "json": re.compile(r'^[\s]*{.*}[\s]*$'),
        }
        
        self.timestamp_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S,%f",
            "%d/%b/%Y:%H:%M:%S %z",
            "%b %d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ"
        ]
        
    def detect_format(self, log_line: str) -> Optional[str]:
        """
        Detect the format of a log line.
        
        Args:
            log_line: Raw log line
            
        Returns:
            Detected format name or None
        """
        for format_name, pattern in self.patterns.items():
            if pattern.match(log_line.strip()):
                return format_name
        return None
    
    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse timestamp string to datetime object.
        
        Args:
            timestamp_str: Timestamp string
            
        Returns:
            Parsed datetime object or None
        """
        for fmt in self.timestamp_formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # Try parsing with dateutil as fallback
        try:
            from dateutil import parser
            return parser.parse(timestamp_str)
        except Exception:
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return None
    
    def parse_json_log(self, log_line: str) -> Optional[LogEntry]:
        """Parse JSON formatted log entry."""
        try:
            data = json.loads(log_line.strip())
            
            # Extract common fields
            timestamp_str = data.get("timestamp", data.get("@timestamp", data.get("time", "")))
            level = data.get("level", data.get("severity", "INFO"))
            message = data.get("message", data.get("msg", str(data)))
            source = data.get("source", data.get("logger", data.get("service", "unknown")))
            
            timestamp = self.parse_timestamp(timestamp_str) if timestamp_str else datetime.now()
            
            return LogEntry(
                timestamp=timestamp,
                level=level.upper(),
                source=source,
                message=message,
                raw_log=log_line,
                metadata=data
            )
        except json.JSONDecodeError:
            return None
    
    def parse_apache_log(self, log_line: str) -> Optional[LogEntry]:
        """Parse Apache access log format."""
        match = self.patterns["apache_access"].match(log_line.strip())
        if not match:
            return None
        
        groups = match.groupdict()
        timestamp = self.parse_timestamp(groups["timestamp"])
        
        message = f'{groups["method"]} {groups["path"]} {groups["protocol"]} - {groups["status"]}'
        
        return LogEntry(
            timestamp=timestamp or datetime.now(),
            level="INFO",
            source="apache",
            message=message,
            raw_log=log_line,
            metadata={
                "ip": groups["ip"],
                "method": groups["method"],
                "path": groups["path"],
                "status": int(groups["status"]),
                "size": groups["size"]
            }
        )
    
    def parse_syslog(self, log_line: str) -> Optional[LogEntry]:
        """Parse syslog format."""
        match = self.patterns["syslog"].match(log_line.strip())
        if not match:
            return None
        
        groups = match.groupdict()
        timestamp = self.parse_timestamp(groups["timestamp"])
        
        return LogEntry(
            timestamp=timestamp or datetime.now(),
            level="INFO",  # Syslog doesn't always have explicit levels
            source=groups["hostname"],
            message=groups["message"],
            raw_log=log_line,
            metadata={
                "hostname": groups["hostname"],
                "process": groups["process"]
            }
        )
    
    def parse_application_log(self, log_line: str) -> Optional[LogEntry]:
        """Parse application log format."""
        match = self.patterns["application"].match(log_line.strip())
        if not match:
            return None
        
        groups = match.groupdict()
        timestamp = self.parse_timestamp(groups["timestamp"])
        level = groups.get("level", "INFO")
        
        return LogEntry(
            timestamp=timestamp or datetime.now(),
            level=level.upper() if level else "INFO",
            source=groups.get("logger", "application"),
            message=groups["message"],
            raw_log=log_line,
            metadata={
                "thread": groups.get("thread"),
                "logger": groups.get("logger")
            }
        )
    
    def parse_line(self, log_line: str) -> Optional[LogEntry]:
        """
        Parse a single log line.
        
        Args:
            log_line: Raw log line
            
        Returns:
            Parsed LogEntry or None if parsing fails
        """
        if not log_line.strip():
            return None
        
        # Detect format and parse accordingly
        format_name = self.detect_format(log_line)
        
        if format_name == "json":
            return self.parse_json_log(log_line)
        elif format_name == "apache_access":
            return self.parse_apache_log(log_line)
        elif format_name == "syslog":
            return self.parse_syslog(log_line)
        elif format_name == "application":
            return self.parse_application_log(log_line)
        else:
            # Generic fallback
            return LogEntry(
                timestamp=datetime.now(),
                level="INFO",
                source="unknown",
                message=log_line.strip(),
                raw_log=log_line,
                metadata={}
            )
    
    def parse_file(self, file_path: str) -> List[LogEntry]:
        """
        Parse an entire log file.
        
        Args:
            file_path: Path to log file
            
        Returns:
            List of parsed LogEntry objects
        """
        entries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = self.parse_line(line)
                        if entry:
                            entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"Log file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
        
        logger.info(f"Parsed {len(entries)} log entries from {file_path}")
        return entries
    
    def parse_batch(self, log_lines: List[str]) -> List[LogEntry]:
        """
        Parse a batch of log lines.
        
        Args:
            log_lines: List of raw log lines
            
        Returns:
            List of parsed LogEntry objects
        """
        entries = []
        
        for line in log_lines:
            try:
                entry = self.parse_line(line)
                if entry:
                    entries.append(entry)
            except Exception as e:
                logger.warning(f"Error parsing log line: {e}")
                continue
        
        return entries
    
    def to_dataframe(self, entries: List[LogEntry]) -> pd.DataFrame:
        """
        Convert LogEntry list to pandas DataFrame.
        
        Args:
            entries: List of LogEntry objects
            
        Returns:
            DataFrame with log data
        """
        data = []
        for entry in entries:
            data.append({
                "timestamp": entry.timestamp,
                "level": entry.level,
                "source": entry.source,
                "message": entry.message,
                "raw_log": entry.raw_log,
                "metadata": json.dumps(entry.metadata)
            })
        
        return pd.DataFrame(data)
