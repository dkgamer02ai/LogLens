# Makefile for LogLens Project

.PHONY: help install test lint docker demo clean

help:  ## Show this help message
	@echo "LogLens: AI-Powered Log Anomaly Detection System"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run unit tests
	python -m pytest tests/ -v

lint:  ## Run code linting
	python -m flake8 src/ tests/ --max-line-length=120
	python -m black src/ tests/ --check

format:  ## Format code
	python -m black src/ tests/

data:  ## Generate synthetic training data
	python -m src.data.synthetic_logs --count 10000 --anomaly-rate 0.1

demo:  ## Run demonstration
	python demo_basic.py

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start Docker services
	docker-compose up -d

docker-down:  ## Stop Docker services
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f loglens

train:  ## Train BERT model (requires data)
	python -m src.models.bert_anomaly_detector --train --data data/synthetic_logs.csv

monitor:  ## Start real-time monitoring
	python -m src.monitoring.realtime_detector --log-paths ./logs/

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf data/*.csv
	rm -rf models/

setup-dev:  ## Setup development environment
	$(MAKE) install
	$(MAKE) data
	@echo "Development environment ready!"

all:  ## Run complete setup and demo
	$(MAKE) install
	$(MAKE) data
	$(MAKE) demo
	@echo "LogLens setup complete!"