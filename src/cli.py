"""
Command Line Interface for LogLens

Provides command-line tools for training, testing, and monitoring.
"""

import click
import json
import sys
import asyncio
from datetime import datetime
from pathlib import Path
import logging

from .models.trainer import ModelTrainer
from .data.generator import SyntheticLogGenerator
from .api.main import create_app
from .alerting.alert_manager import AlertManager
from .preprocessing.log_parser import LogParser
from .models.bert_classifier import BERTAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="LogLens")
def cli():
    """LogLens: AI-Powered Log Anomaly Detection System"""
    pass


@cli.group()
def data():
    """Data generation and management commands."""
    pass


@data.command()
@click.option('--samples', '-n', default=10000, help='Number of samples to generate')
@click.option('--anomaly-ratio', '-r', default=0.1, help='Ratio of anomalous samples')
@click.option('--output', '-o', default='data/synthetic_logs.json', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json', help='Output format')
@click.option('--seed', default=42, help='Random seed')
def generate(samples, anomaly_ratio, output, format, seed):
    """Generate synthetic log dataset."""
    click.echo(f"Generating {samples} synthetic log entries...")
    
    generator = SyntheticLogGenerator(seed=seed)
    logs = generator.generate_dataset(
        num_samples=samples,
        anomaly_ratio=anomaly_ratio
    )
    
    # Ensure output directory exists
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        generator.save_dataset(logs, output)
    else:
        generator.save_dataset_csv(logs, output)
    
    click.echo(f"Dataset saved to {output}")


@cli.group()
def model():
    """Model training and evaluation commands."""
    pass


@model.command()
@click.option('--config', '-c', help='Path to config file')
@click.option('--data', '-d', help='Path to training data')
@click.option('--synthetic', '-s', default=10000, help='Number of synthetic samples')
@click.option('--output', '-o', default='models', help='Output directory')
@click.option('--wandb', is_flag=True, help='Use Weights & Biases logging')
def train(config, data, synthetic, output, wandb):
    """Train the anomaly detection model."""
    click.echo("Starting model training...")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    if wandb:
        trainer.config["use_wandb"] = True
    
    # Run training
    results = trainer.run_training_pipeline(
        data_path=data,
        num_synthetic=synthetic,
        output_dir=output
    )
    
    click.echo("Training completed successfully!")
    click.echo(f"Final F1-Score: {results['evaluation']['f1']:.4f}")
    click.echo(f"Model saved to: {results['training']['model_path']}")


@model.command()
@click.option('--model-path', '-m', required=True, help='Path to trained model')
@click.option('--data', '-d', required=True, help='Path to test data')
@click.option('--output', '-o', help='Output file for results')
def evaluate(model_path, data, output):
    """Evaluate a trained model."""
    click.echo(f"Loading model from {model_path}...")
    
    try:
        model = BERTAnomalyDetector.load_model(model_path)
        click.echo("Model loaded successfully!")
        
        # Load test data
        if data.endswith('.json'):
            with open(data, 'r') as f:
                test_data = json.load(f)
            messages = [item["message"] for item in test_data]
            labels = [item.get("is_anomaly", 0) for item in test_data]
        else:
            click.echo("Only JSON format supported for evaluation data")
            return
        
        # Evaluate
        predictions, scores = model.predict(messages)
        
        from sklearn.metrics import classification_report, accuracy_score
        
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        
        click.echo(f"Accuracy: {accuracy:.4f}")
        click.echo("Classification Report:")
        click.echo(report)
        
        if output:
            results = {
                "accuracy": accuracy,
                "classification_report": report,
                "predictions": predictions,
                "scores": scores.tolist(),
                "labels": labels
            }
            
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"Results saved to {output}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host, port, workers, reload):
    """Start the LogLens API server."""
    import uvicorn
    
    click.echo(f"Starting LogLens API server on {host}:{port}...")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.option('--input', '-i', required=True, help='Input log file or directory')
@click.option('--api-url', default='http://localhost:8000', help='LogLens API URL')
@click.option('--threshold', default=0.7, help='Anomaly detection threshold')
@click.option('--batch-size', default=100, help='Batch size for processing')
@click.option('--follow', '-f', is_flag=True, help='Follow log file for new entries')
def monitor(input, api_url, threshold, batch_size, follow):
    """Monitor log files for anomalies."""
    import requests
    import time
    from pathlib import Path
    
    click.echo(f"Monitoring {input} for anomalies...")
    click.echo(f"API URL: {api_url}")
    click.echo(f"Threshold: {threshold}")
    
    log_parser = LogParser()
    
    if follow:
        # Follow mode - watch for new log entries
        click.echo("Following log file for new entries (Ctrl+C to stop)...")
        
        try:
            import tailer
            for line in tailer.follow(open(input)):
                entry = log_parser.parse_line(line)
                if entry:
                    # Send to API
                    response = requests.post(
                        f"{api_url}/detect",
                        json={
                            "message": entry.message,
                            "timestamp": entry.timestamp.isoformat(),
                            "level": entry.level,
                            "source": entry.source
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result["is_anomaly"]:
                            click.echo(f"🚨 ANOMALY DETECTED: {entry.message[:100]}...")
                            click.echo(f"   Score: {result['anomaly_score']:.3f}, Severity: {result['severity']}")
                    else:
                        click.echo(f"API Error: {response.status_code}")
                        
        except KeyboardInterrupt:
            click.echo("Monitoring stopped.")
        except ImportError:
            click.echo("Install 'tailer' package for follow mode: pip install tailer")
            
    else:
        # Batch mode - process existing file
        entries = log_parser.parse_file(input)
        click.echo(f"Parsed {len(entries)} log entries")
        
        # Process in batches
        anomalies_found = 0
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            
            # Prepare batch request
            batch_data = {
                "logs": [
                    {
                        "message": entry.message,
                        "timestamp": entry.timestamp.isoformat(),
                        "level": entry.level,
                        "source": entry.source
                    }
                    for entry in batch
                ],
                "threshold": threshold
            }
            
            try:
                response = requests.post(
                    f"{api_url}/detect/batch",
                    json=batch_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    batch_anomalies = result["anomalies_detected"]
                    anomalies_found += batch_anomalies
                    
                    if batch_anomalies > 0:
                        click.echo(f"Batch {i//batch_size + 1}: {batch_anomalies} anomalies detected")
                        
                        # Show detailed results for anomalies
                        for j, res in enumerate(result["results"]):
                            if res["is_anomaly"]:
                                entry_idx = i + j
                                if entry_idx < len(entries):
                                    click.echo(f"  🚨 {entries[entry_idx].message[:80]}...")
                                    click.echo(f"     Score: {res['anomaly_score']:.3f}, Severity: {res['severity']}")
                else:
                    click.echo(f"API Error: {response.status_code}")
                    
            except Exception as e:
                click.echo(f"Error processing batch: {e}")
        
        click.echo(f"\nProcessing complete. Found {anomalies_found} anomalies out of {len(entries)} entries.")


@cli.command()
@click.option('--config', '-c', help='Path to alerting config file')
def test_alerts(config):
    """Test the alerting system."""
    click.echo("Testing alert system...")
    
    alert_manager = AlertManager(config)
    results = alert_manager.test_channels()
    
    for channel, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        click.echo(f"{channel}: {status}")


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API URL to check')
def status(url):
    """Check system status."""
    import requests
    
    try:
        response = requests.get(f"{url}/status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            click.echo("🔍 LogLens System Status")
            click.echo(f"Status: {data['status']}")
            click.echo(f"Model Loaded: {'✅' if data['model_loaded'] else '❌'}")
            click.echo(f"Uptime: {data['uptime']}")
            click.echo(f"Total Requests: {data['total_requests']}")
            click.echo(f"Anomalies Detected: {data['anomalies_detected']}")
            
            if data['last_alert']:
                click.echo(f"Last Alert: {data['last_alert']}")
            else:
                click.echo("Last Alert: None")
        else:
            click.echo(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        click.echo(f"❌ Connection Error: {e}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
