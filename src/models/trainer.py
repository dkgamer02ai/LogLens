"""
Model Training Module

This module handles training the BERT anomaly detection model
on synthetic and real log datasets.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
import logging
import yaml
import wandb

from ..models.bert_classifier import BERTAnomalyDetector, BERTTrainer
from ..data.generator import SyntheticLogGenerator
from ..preprocessing.log_parser import LogParser
from ..preprocessing.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class LogDataset(Dataset):
    """PyTorch Dataset for log entries."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelTrainer:
    """
    Comprehensive trainer for the log anomaly detection model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model = None
        self.trainer = None
        self.feature_extractor = FeatureExtractor()
        self.log_parser = LogParser()
        
        # Initialize wandb for experiment tracking
        if self.config.get("use_wandb", False):
            wandb.init(
                project="loglens-anomaly-detection",
                config=self.config
            )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load training configuration."""
        if config_path is None:
            config_path = "config/model.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "model": {
                    "name": "bert-base-uncased",
                    "max_length": 512,
                    "num_labels": 2
                },
                "training": {
                    "batch_size": 16,
                    "learning_rate": 2e-5,
                    "num_epochs": 3
                }
            }
    
    def prepare_synthetic_data(self, num_samples: int = 10000) -> Tuple[List[str], List[int]]:
        """Generate and prepare synthetic training data."""
        logger.info(f"Generating {num_samples} synthetic log entries...")
        
        generator = SyntheticLogGenerator()
        synthetic_logs = generator.generate_dataset(
            num_samples=num_samples,
            anomaly_ratio=0.15,  # Higher ratio for training
            include_scenarios=True
        )
        
        # Extract texts and labels
        texts = [log["message"] for log in synthetic_logs]
        labels = [1 if log["is_anomaly"] else 0 for log in synthetic_logs]
        
        logger.info(f"Prepared {len(texts)} samples: {sum(labels)} anomalies, {len(labels) - sum(labels)} normal")
        return texts, labels
    
    def load_real_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load and prepare real log data."""
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            texts = [item["message"] for item in data]
            labels = [item.get("is_anomaly", 0) for item in data]
        
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            texts = df['message'].tolist()
            labels = df.get('is_anomaly', [0] * len(texts)).tolist()
        
        else:
            # Assume raw log file
            entries = self.log_parser.parse_file(data_path)
            texts = [entry.message for entry in entries]
            labels = [0] * len(texts)  # Default to normal, manual labeling needed
        
        return texts, labels
    
    def create_datasets(
        self, 
        texts: List[str], 
        labels: List[int],
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[LogDataset, LogDataset, LogDataset]:
        """Create train, validation, and test datasets."""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )
        
        logger.info(f"Dataset splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Initialize model to get tokenizer
        if self.model is None:
            self.model = BERTAnomalyDetector(
                model_name=self.config["model"]["name"],
                max_length=self.config["model"]["max_length"]
            )
        
        # Create datasets
        train_dataset = LogDataset(X_train, y_train, self.model.tokenizer, self.model.max_length)
        val_dataset = LogDataset(X_val, y_val, self.model.tokenizer, self.model.max_length)
        test_dataset = LogDataset(X_test, y_test, self.model.tokenizer, self.model.max_length)
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(
        self, 
        train_dataset: LogDataset,
        val_dataset: Optional[LogDataset] = None,
        output_dir: str = "models/checkpoints"
    ) -> Dict[str, Any]:
        """Train the BERT anomaly detection model."""
        
        if self.model is None:
            self.model = BERTAnomalyDetector(
                model_name=self.config["model"]["name"],
                max_length=self.config["model"]["max_length"]
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_steps=self.config["training"].get("warmup_steps", 500),
            weight_decay=self.config["training"].get("weight_decay", 0.01),
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=500 if val_dataset else None,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_f1" if val_dataset else None,
            greater_is_better=True,
            report_to=["wandb"] if self.config.get("use_wandb", False) else [],
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = BERTTrainer(self.model, training_args)
        
        # Add early stopping
        callbacks = []
        if val_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Train the model
        logger.info("Starting model training...")
        train_result = trainer.train(train_dataset, val_dataset)
        
        # Save the model
        model_path = os.path.join(output_dir, "final_model.pt")
        self.model.save_model(model_path)
        
        logger.info(f"Training completed. Model saved to {model_path}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "model_path": model_path
        }
    
    def evaluate_model(self, test_dataset: LogDataset) -> Dict[str, Any]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        # Get predictions
        all_predictions = []
        all_labels = []
        all_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                
                # Single prediction
                outputs = self.model.forward(
                    input_ids=item['input_ids'].unsqueeze(0),
                    attention_mask=item['attention_mask'].unsqueeze(0)
                )
                
                probabilities = outputs['probabilities'].cpu().numpy()[0]
                prediction = outputs['predictions'].cpu().numpy()[0]
                
                all_predictions.append(prediction)
                all_labels.append(item['labels'].numpy())
                all_scores.append(probabilities[1])  # Anomaly probability
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_scores)
        pr_auc = average_precision_score(all_labels, all_scores)
        
        # Classification report
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(all_labels),
            "num_anomalies": sum(all_labels)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        return results
    
    def run_training_pipeline(
        self,
        data_path: Optional[str] = None,
        num_synthetic: int = 10000,
        output_dir: str = "models"
    ) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        
        logger.info("Starting training pipeline...")
        
        # Prepare data
        if data_path:
            texts, labels = self.load_real_data(data_path)
        else:
            texts, labels = self.prepare_synthetic_data(num_synthetic)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(texts, labels)
        
        # Train model
        train_results = self.train_model(train_dataset, val_dataset, output_dir)
        
        # Evaluate model
        eval_results = self.evaluate_model(test_dataset)
        
        # Combine results
        results = {
            "training": train_results,
            "evaluation": eval_results,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training pipeline completed. Results saved to {results_path}")
        
        return results


def main():
    """Command-line interface for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LogLens anomaly detection model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--synthetic", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer
    trainer = ModelTrainer(args.config)
    
    if args.wandb:
        trainer.config["use_wandb"] = True
    
    # Run training
    results = trainer.run_training_pipeline(
        data_path=args.data,
        num_synthetic=args.synthetic,
        output_dir=args.output
    )
    
    print("Training completed successfully!")
    print(f"Final F1-Score: {results['evaluation']['f1']:.4f}")


if __name__ == "__main__":
    main()
