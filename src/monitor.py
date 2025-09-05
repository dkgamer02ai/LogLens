"""
Real-time Log Monitoring Module

This module provides real-time monitoring of log files and streams,
sending entries to the LogLens API for anomaly detection.
"""

import asyncio
import aiofiles
import aiohttp
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator
import logging
import signal
import sys

from .preprocessing.log_parser import LogParser

logger = logging.getLogger(__name__)


class LogMonitor:
    """
    Real-time log file monitor with API integration.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        batch_size: int = 10,
        batch_timeout: float = 5.0,
        threshold: float = 0.7
    ):
        self.api_url = api_url.rstrip('/')
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.threshold = threshold
        self.log_parser = LogParser()
        self.running = False
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            "lines_processed": 0,
            "anomalies_detected": 0,
            "api_errors": 0,
            "start_time": None
        }
        
        # Batch processing
        self.batch_buffer = []
        self.last_batch_time = time.time()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def follow_file(self, file_path: str) -> AsyncGenerator[str, None]:
        """
        Follow a log file for new lines (like tail -f).
        
        Args:
            file_path: Path to the log file
            
        Yields:
            New lines from the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")
        
        # Start from the end of the file
        async with aiofiles.open(file_path, 'r') as f:
            await f.seek(0, 2)  # Seek to end
            
            while self.running:
                line = await f.readline()
                
                if line:
                    yield line.strip()
                else:
                    # No new data, wait a bit
                    await asyncio.sleep(0.1)
    
    async def process_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Process a single log line.
        
        Args:
            line: Raw log line
            
        Returns:
            Processed log entry or None if parsing failed
        """
        try:
            entry = self.log_parser.parse_line(line)
            if not entry:
                return None
            
            return {
                "message": entry.message,
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level,
                "source": entry.source,
                "raw_log": entry.raw_log
            }
        except Exception as e:
            logger.warning(f"Failed to parse line: {e}")
            return None
    
    async def send_batch(self, batch: list) -> Dict[str, Any]:
        """
        Send a batch of log entries to the API.
        
        Args:
            batch: List of log entries
            
        Returns:
            API response data
        """
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        try:
            payload = {
                "logs": batch,
                "threshold": self.threshold
            }
            
            async with self.session.post(
                f"{self.api_url}/detect/batch",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    self.stats["api_errors"] += 1
                    return {"error": f"API error {response.status}"}
                    
        except asyncio.TimeoutError:
            logger.error("API request timeout")
            self.stats["api_errors"] += 1
            return {"error": "timeout"}
        except Exception as e:
            logger.error(f"API request failed: {e}")
            self.stats["api_errors"] += 1
            return {"error": str(e)}
    
    async def flush_batch(self):
        """Flush the current batch to the API."""
        if not self.batch_buffer:
            return
        
        logger.debug(f"Flushing batch of {len(self.batch_buffer)} entries")
        
        response = await self.send_batch(self.batch_buffer)
        
        if "error" not in response:
            anomalies = response.get("anomalies_detected", 0)
            self.stats["anomalies_detected"] += anomalies
            
            if anomalies > 0:
                logger.info(f"🚨 Detected {anomalies} anomalies in batch")
                
                # Log detailed anomaly information
                for i, result in enumerate(response.get("results", [])):
                    if result.get("is_anomaly"):
                        entry = self.batch_buffer[i]
                        logger.warning(
                            f"ANOMALY: {entry['message'][:100]}... "
                            f"(Score: {result['anomaly_score']:.3f}, "
                            f"Severity: {result['severity']})"
                        )
        
        self.batch_buffer.clear()
        self.last_batch_time = time.time()
    
    async def add_to_batch(self, entry: Dict[str, Any]):
        """Add an entry to the processing batch."""
        self.batch_buffer.append(entry)
        self.stats["lines_processed"] += 1
        
        # Flush if batch is full or timeout reached
        current_time = time.time()
        should_flush = (
            len(self.batch_buffer) >= self.batch_size or
            current_time - self.last_batch_time >= self.batch_timeout
        )
        
        if should_flush:
            await self.flush_batch()
    
    async def monitor_file(self, file_path: str):
        """
        Monitor a log file for anomalies.
        
        Args:
            file_path: Path to the log file to monitor
        """
        logger.info(f"Starting to monitor {file_path}")
        self.stats["start_time"] = datetime.now()
        self.running = True
        
        try:
            async for line in self.follow_file(file_path):
                if not self.running:
                    break
                
                entry = await self.process_line(line)
                if entry:
                    await self.add_to_batch(entry)
            
        except Exception as e:
            logger.error(f"Error monitoring file: {e}")
        finally:
            # Flush remaining entries
            if self.batch_buffer:
                await self.flush_batch()
    
    async def monitor_stream(self, stream=sys.stdin):
        """
        Monitor log entries from a stream (stdin).
        
        Args:
            stream: Input stream to read from
        """
        logger.info("Starting to monitor stdin")
        self.stats["start_time"] = datetime.now()
        self.running = True
        
        try:
            while self.running:
                line = await asyncio.to_thread(stream.readline)
                
                if not line:  # EOF
                    break
                
                entry = await self.process_line(line.strip())
                if entry:
                    await self.add_to_batch(entry)
            
        except Exception as e:
            logger.error(f"Error monitoring stream: {e}")
        finally:
            if self.batch_buffer:
                await self.flush_batch()
    
    def stop(self):
        """Stop monitoring."""
        logger.info("Stopping log monitor...")
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = self.stats.copy()
        
        if stats["start_time"]:
            uptime = datetime.now() - stats["start_time"]
            stats["uptime"] = str(uptime).split('.')[0]  # Remove microseconds
            
            # Calculate rates
            uptime_seconds = uptime.total_seconds()
            if uptime_seconds > 0:
                stats["lines_per_second"] = stats["lines_processed"] / uptime_seconds
                stats["anomaly_rate"] = stats["anomalies_detected"] / max(stats["lines_processed"], 1)
        
        return stats


async def main():
    """Main entry point for the monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LogLens Real-time Monitor")
    parser.add_argument("--input", "-i", help="Input log file (use '-' for stdin)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="LogLens API URL")
    parser.add_argument("--threshold", type=float, default=0.7, help="Anomaly threshold")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--batch-timeout", type=float, default=5.0, help="Batch timeout (seconds)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Signal handlers for graceful shutdown
    monitor = None
    
    def signal_handler(signum, frame):
        if monitor:
            monitor.stop()
        logger.info("Received shutdown signal")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    async with LogMonitor(
        api_url=args.api_url,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        threshold=args.threshold
    ) as monitor:
        
        try:
            if args.input == "-" or args.input is None:
                await monitor.monitor_stream()
            else:
                await monitor.monitor_file(args.input)
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            # Print final statistics
            stats = monitor.get_stats()
            logger.info("Final Statistics:")
            logger.info(f"  Lines processed: {stats['lines_processed']}")
            logger.info(f"  Anomalies detected: {stats['anomalies_detected']}")
            logger.info(f"  API errors: {stats['api_errors']}")
            logger.info(f"  Uptime: {stats.get('uptime', 'N/A')}")
            if 'anomaly_rate' in stats:
                logger.info(f"  Anomaly rate: {stats['anomaly_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
