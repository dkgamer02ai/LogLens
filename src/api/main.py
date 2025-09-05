"""
FastAPI Web Application for LogLens

This module provides the REST API interface for real-time log anomaly detection.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from starlette.responses import Response
import logging

# Import LogLens modules
from ..models.bert_classifier import BERTAnomalyDetector
from ..preprocessing.log_parser import LogParser, LogEntry
from ..preprocessing.feature_extractor import FeatureExtractor
from ..alerting.alert_manager import AlertManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear any existing metrics to prevent duplicates
try:
    # Find and unregister any existing LogLens metrics
    collectors_to_unregister = []
    for collector in list(REGISTRY._collector_to_names.keys()):
        if hasattr(collector, '_name'):
            if collector._name.startswith('loglens_'):
                collectors_to_unregister.append(collector)
    
    for collector in collectors_to_unregister:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass  # Already unregistered
except Exception as e:
    logger.warning(f"Error clearing existing metrics: {e}")

# Prometheus metrics
REQUEST_COUNT = Counter('loglens_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('loglens_request_duration_seconds', 'Request duration')
ANOMALY_COUNT = Counter('loglens_anomalies_total', 'Total anomalies detected', ['severity'])
PROCESSING_TIME = Histogram('loglens_processing_time_seconds', 'Log processing time')


# Pydantic models for API
class LogEntryRequest(BaseModel):
    """Single log entry for analysis."""
    message: str = Field(..., description="Log message to analyze")
    timestamp: Optional[str] = Field(None, description="Log timestamp (ISO format)")
    level: Optional[str] = Field("INFO", description="Log level")
    source: Optional[str] = Field("unknown", description="Log source")


class BatchLogRequest(BaseModel):
    """Batch of log entries for analysis."""
    logs: List[LogEntryRequest] = Field(..., description="List of log entries")
    threshold: Optional[float] = Field(0.7, description="Anomaly detection threshold")


class AnomalyResponse(BaseModel):
    """Response for anomaly detection."""
    is_anomaly: bool = Field(..., description="Whether the log is anomalous")
    anomaly_score: float = Field(..., description="Anomaly confidence score")
    severity: str = Field(..., description="Severity level")
    confidence: float = Field(..., description="Detection confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    alert_sent: bool = Field(..., description="Whether alert was sent")


class BatchAnomalyResponse(BaseModel):
    """Response for batch anomaly detection."""
    results: List[AnomalyResponse] = Field(..., description="Detection results for each log")
    total_processed: int = Field(..., description="Total logs processed")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    processing_time: float = Field(..., description="Total processing time")


class SystemStatus(BaseModel):
    """System status information."""
    status: str = Field(..., description="System status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime: str = Field(..., description="System uptime")
    total_requests: int = Field(..., description="Total requests processed")
    anomalies_detected: int = Field(..., description="Total anomalies detected")
    last_alert: Optional[str] = Field(None, description="Last alert timestamp")


# Global application state
class AppState:
    def __init__(self):
        self.model: Optional[BERTAnomalyDetector] = None
        self.log_parser = LogParser()
        self.feature_extractor = FeatureExtractor()
        self.alert_manager: Optional[AlertManager] = None
        self.redis_client: Optional[redis.Redis] = None
        self.start_time = datetime.now()
        self.total_requests = 0
        self.anomalies_detected = 0
        self.last_alert: Optional[datetime] = None

app_state = AppState()


async def load_model():
    """Load the trained BERT model."""
    try:
        model_path = os.getenv("MODEL_PATH", "models/final_model.pt")
        if os.path.exists(model_path):
            app_state.model = BERTAnomalyDetector.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            # Load pre-trained model for demo
            app_state.model = BERTAnomalyDetector()
            logger.warning("No trained model found, using pre-trained BERT")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


async def initialize_services():
    """Initialize external services."""
    try:
        # Initialize Redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        app_state.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Test Redis connection
        await app_state.redis_client.ping()
        logger.info("Redis connection established")
        
        # Initialize Alert Manager
        app_state.alert_manager = AlertManager()
        app_state.alert_manager.start()
        logger.info("Alert Manager initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Continue without Redis/alerts if they fail


async def cleanup_services():
    """Cleanup services on shutdown."""
    if app_state.alert_manager:
        app_state.alert_manager.stop()
    
    if app_state.redis_client:
        await app_state.redis_client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await load_model()
    await initialize_services()
    logger.info("LogLens API started successfully")
    
    yield
    
    # Shutdown
    await cleanup_services()
    logger.info("LogLens API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="LogLens API",
    description="AI-Powered Log Anomaly Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_model() -> BERTAnomalyDetector:
    """Dependency to get the loaded model."""
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app_state.model


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic HTML interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LogLens - AI Log Anomaly Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; color: #333; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .code { background: #f5f5f5; padding: 10px; border-radius: 3px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 LogLens</h1>
                <h2>AI-Powered Log Anomaly Detection System</h2>
            </div>
            
            <div class="section">
                <h3>API Endpoints</h3>
                <ul>
                    <li><strong>POST /detect</strong> - Analyze single log entry</li>
                    <li><strong>POST /detect/batch</strong> - Analyze multiple log entries</li>
                    <li><strong>GET /status</strong> - System status</li>
                    <li><strong>GET /metrics</strong> - Prometheus metrics</li>
                    <li><strong>GET /docs</strong> - Interactive API documentation</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>Example Usage</h3>
                <div class="code">
curl -X POST "http://localhost:8000/detect" \\
     -H "Content-Type: application/json" \\
     -d '{"message": "Failed login attempt from 192.168.1.100"}'
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/detect", response_model=AnomalyResponse)
async def detect_anomaly(
    log_entry: LogEntryRequest,
    background_tasks: BackgroundTasks,
    model: BERTAnomalyDetector = Depends(get_model)
):
    """Detect anomaly in a single log entry."""
    start_time = datetime.now()
    
    try:
        REQUEST_COUNT.labels(method='POST', endpoint='/detect').inc()
        
        # Parse log entry
        parsed_entry = LogEntry(
            timestamp=datetime.fromisoformat(log_entry.timestamp) if log_entry.timestamp else datetime.now(),
            level=log_entry.level,
            source=log_entry.source,
            message=log_entry.message,
            raw_log=f"{log_entry.timestamp or datetime.now()} [{log_entry.level}] {log_entry.message}",
            metadata={}
        )
        
        # Predict anomaly
        with PROCESSING_TIME.time():
            result = model.predict_single(log_entry.message)
        
        # Determine severity
        severity = "low"
        if result["anomaly_score"] >= 0.9:
            severity = "critical"
        elif result["anomaly_score"] >= 0.7:
            severity = "high"
        elif result["anomaly_score"] >= 0.5:
            severity = "medium"
        
        # Update metrics
        app_state.total_requests += 1
        if result["is_anomaly"]:
            app_state.anomalies_detected += 1
            ANOMALY_COUNT.labels(severity=severity).inc()
        
        # Send alert if anomaly detected
        alert_sent = False
        if result["is_anomaly"] and app_state.alert_manager:
            app_state.alert_manager.send_alert_async(
                log_message=log_entry.message,
                anomaly_score=result["anomaly_score"],
                anomaly_details=f"Anomaly detected with {result['confidence']:.1%} confidence",
                source=log_entry.source
            )
            alert_sent = True
            app_state.last_alert = datetime.now()
        
        # Cache result in Redis (if available)
        if app_state.redis_client:
            background_tasks.add_task(
                cache_result,
                log_entry.message,
                result,
                severity
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnomalyResponse(
            is_anomaly=result["is_anomaly"],
            anomaly_score=result["anomaly_score"],
            severity=severity,
            confidence=result["confidence"],
            processing_time=processing_time,
            alert_sent=alert_sent
        )
        
    except Exception as e:
        logger.error(f"Error processing log entry: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/detect/batch", response_model=BatchAnomalyResponse)
async def detect_batch_anomalies(
    batch_request: BatchLogRequest,
    background_tasks: BackgroundTasks,
    model: BERTAnomalyDetector = Depends(get_model)
):
    """Detect anomalies in a batch of log entries."""
    start_time = datetime.now()
    
    try:
        REQUEST_COUNT.labels(method='POST', endpoint='/detect/batch').inc()
        
        results = []
        anomalies_detected = 0
        
        # Process each log entry
        messages = [log.message for log in batch_request.logs]
        
        with PROCESSING_TIME.time():
            predictions, scores = model.predict(messages, batch_request.threshold)
        
        for i, (log_entry, prediction, score) in enumerate(zip(batch_request.logs, predictions, scores)):
            # Determine severity
            severity = "low"
            if score >= 0.9:
                severity = "critical"
            elif score >= 0.7:
                severity = "high"
            elif score >= 0.5:
                severity = "medium"
            
            confidence = max(score, 1 - score)
            is_anomaly = bool(prediction)
            
            if is_anomaly:
                anomalies_detected += 1
                ANOMALY_COUNT.labels(severity=severity).inc()
                
                # Send alert
                if app_state.alert_manager:
                    app_state.alert_manager.send_alert_async(
                        log_message=log_entry.message,
                        anomaly_score=score,
                        anomaly_details=f"Batch anomaly detected with {confidence:.1%} confidence",
                        source=log_entry.source
                    )
            
            results.append(AnomalyResponse(
                is_anomaly=is_anomaly,
                anomaly_score=score,
                severity=severity,
                confidence=confidence,
                processing_time=0.0,  # Individual time not measured in batch
                alert_sent=is_anomaly
            ))
        
        # Update global stats
        app_state.total_requests += len(batch_request.logs)
        app_state.anomalies_detected += anomalies_detected
        
        if anomalies_detected > 0:
            app_state.last_alert = datetime.now()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchAnomalyResponse(
            results=results,
            total_processed=len(batch_request.logs),
            anomalies_detected=anomalies_detected,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and statistics."""
    uptime = datetime.now() - app_state.start_time
    
    return SystemStatus(
        status="healthy" if app_state.model else "degraded",
        model_loaded=app_state.model is not None,
        uptime=str(uptime).split('.')[0],  # Remove microseconds
        total_requests=app_state.total_requests,
        anomalies_detected=app_state.anomalies_detected,
        last_alert=app_state.last_alert.isoformat() if app_state.last_alert else None
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/test-alert")
async def test_alert():
    """Test the alerting system."""
    if not app_state.alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not available")
    
    try:
        test_results = app_state.alert_manager.test_channels()
        return {"message": "Test alerts sent", "results": test_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert test failed: {str(e)}")


async def cache_result(message: str, result: Dict[str, Any], severity: str):
    """Cache anomaly detection result in Redis."""
    if not app_state.redis_client:
        return
    
    try:
        import hashlib
        key = f"loglens:result:{hashlib.md5(message.encode()).hexdigest()}"
        
        cache_data = {
            **result,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        await app_state.redis_client.setex(
            key, 
            timedelta(hours=24).total_seconds(),  # Cache for 24 hours
            json.dumps(cache_data, default=str)
        )
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    # Load configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,
        workers=workers if not debug else 1,
        log_level="info"
    )
