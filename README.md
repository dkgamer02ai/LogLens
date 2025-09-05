# LogLens: AI-Powered Log Anomaly Detection System

An enterprise-grade log anomaly detection system powered by transformer-based machine learning models.

## Features

- **AI-Powered Detection**: Fine-tuned BERT variant for log classification and anomaly detection
- **Real-time Processing**: Continuous monitoring with immediate anomaly flagging
- **Feature Extraction Pipeline**: Advanced log parsing and semantic similarity analysis
- **Scalable Deployment**: Docker containerization for production environments
- **Intelligent Alerting**: Automated notification system for anomalous events

## Architecture

```
LogLens/
├── src/
│   ├── models/           # BERT-based anomaly detection models
│   ├── preprocessing/    # Feature extraction and embeddings
│   ├── data/            # Synthetic log generation
│   ├── monitoring/      # Real-time detection system
│   └── alerting/        # Alert mechanisms
├── config/              # Configuration files
├── tests/               # Unit tests
└── docker/              # Container definitions
```

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Generate Synthetic Training Data**
```bash
python -m src.data.synthetic_logs --count 10000
```

3. **Train the Model**
```bash
python -m src.models.bert_anomaly_detector --train
```

4. **Start Real-time Monitoring**
```bash
python -m src.monitoring.realtime_detector --log-path /var/log/
```

## Docker Deployment

```bash
docker-compose up -d
```

## Configuration

Configure the system using `config/config.yaml`:

```yaml
model:
  name: "bert-base-uncased"
  max_length: 512
  
monitoring:
  log_paths: ["/var/log/"]
  check_interval: 5
  
alerting:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
```

## License

MIT License