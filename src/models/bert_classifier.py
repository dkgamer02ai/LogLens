"""
BERT-based Anomaly Detection Model

This module implements a fine-tuned BERT classifier for log anomaly detection.
"""

import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    BertConfig,
    Trainer,
    TrainingArguments
)
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


class BERTAnomalyDetector(nn.Module):
    """
    BERT-based anomaly detection model for log analysis.
    
    This model fine-tunes a pre-trained BERT model for binary classification
    of log entries as normal or anomalous.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Initialize BERT configuration
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
        # Initialize BERT model
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Add custom layers for anomaly detection
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        logger.info(f"Initialized BERT Anomaly Detector with {model_name}")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for input sequences
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing loss, logits, and probabilities
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": torch.argmax(logits, dim=-1)
        }
        
        if labels is not None:
            result["loss"] = outputs.loss
            
        return result
    
    def predict(
        self, 
        texts: List[str], 
        threshold: float = 0.5
    ) -> Tuple[List[int], List[float]]:
        """
        Predict anomalies in log texts.
        
        Args:
            texts: List of log text entries
            threshold: Anomaly threshold for classification
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        self.eval()
        
        # Tokenize texts
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )
            
        probabilities = outputs["probabilities"].cpu().numpy()
        anomaly_scores = probabilities[:, 1]  # Probability of anomaly class
        
        predictions = (anomaly_scores > threshold).astype(int).tolist()
        confidence_scores = anomaly_scores.tolist()
        
        return predictions, confidence_scores
    
    def predict_single(self, text: str, threshold: float = 0.5) -> Dict[str, float]:
        """
        Predict anomaly for a single log entry.
        
        Args:
            text: Log text entry
            threshold: Anomaly threshold
            
        Returns:
            Dictionary with prediction details
        """
        predictions, scores = self.predict([text], threshold)
        
        return {
            "is_anomaly": bool(predictions[0]),
            "anomaly_score": scores[0],
            "confidence": max(scores[0], 1 - scores[0]),
            "threshold": threshold
        }
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_labels": self.num_labels
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> "BERTAnomalyDetector":
        """Load a trained model."""
        checkpoint = torch.load(path, map_location="cpu")
        
        model = cls(
            model_name=checkpoint["model_name"],
            max_length=checkpoint["max_length"],
            num_labels=checkpoint["num_labels"]
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model


class BERTTrainer:
    """
    Trainer class for fine-tuning BERT on log anomaly detection.
    """
    
    def __init__(
        self,
        model: BERTAnomalyDetector,
        training_args: TrainingArguments
    ):
        self.model = model
        self.training_args = training_args
        self.trainer = None
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def train(self, train_dataset, eval_dataset=None):
        """Train the model."""
        self.trainer = Trainer(
            model=self.model.bert,  # Use the BERT model for training
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )
        
        logger.info("Starting model training...")
        train_result = self.trainer.train()
        
        logger.info("Training completed!")
        return train_result
    
    def evaluate(self, eval_dataset):
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
            
        logger.info("Evaluating model...")
        eval_result = self.trainer.evaluate(eval_dataset)
        
        return eval_result
