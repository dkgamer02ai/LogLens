# Makefile for LogLens project

.PHONY: help install dev-install test lint format clean docker-build docker-run train generate-data

# Default target
help:
	@echo "LogLens - AI-Powered Log Anomaly Detection System"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install production dependencies"
	@echo "  dev-install   - Install development dependencies"
	@echo "  test          - Run tests"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black"
	@echo "  clean         - Clean build artifacts"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run with Docker Compose"
	@echo "  train         - Train the anomaly detection model"
	@echo "  generate-data - Generate synthetic training data"
	@echo "  serve         - Start the API server"
	@echo "  monitor       - Start log monitoring"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

dev-install: install
	pip install pytest pytest-asyncio black flake8 mypy

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Docker
docker-build:
	docker build -t loglens:latest .

docker-run:
	docker-compose up --build

docker-down:
	docker-compose down -v

# Data and training
generate-data:
	python -m src.data.generator --samples 10000 --output data/synthetic_logs.json

generate-samples:
	python data/samples.py

train:
	python -m src.models.trainer --synthetic 10000 --output models/

train-with-data:
	python -m src.models.trainer --data data/labeled_dataset.json --output models/

# Development server
serve:
	python -m src.api.main

serve-dev:
	python -m src.api.main --reload

# Monitoring
monitor-stdin:
	python -m src.monitor --input -

monitor-file:
	python -m src.monitor --input /var/log/syslog

# Utilities
status:
	python -m src.cli status

test-alerts:
	python -m src.cli test-alerts

# Setup development environment
setup-dev: dev-install generate-samples
	@echo "Development environment setup complete!"
	@echo "Run 'make train' to train a model"
	@echo "Run 'make serve' to start the API server"

# Production deployment
deploy: clean install generate-data train
	@echo "Production deployment ready!"
	@echo "Run 'make docker-run' to start services"
