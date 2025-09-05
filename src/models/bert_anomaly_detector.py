"""
BERT-based Log Anomaly Detection Model

Fine-tuned transformer model for classifying log entries as normal or anomalous.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import yaml
import os
import pickle
from typing import List, Dict, Tuple, Optional


class LogDataset(Dataset):
    """Dataset class for log anomaly detection."""
    
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
        
        # Tokenize the text
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


class BERTLogAnomalyDetector(nn.Module):
    """BERT-based model for log anomaly detection."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2, dropout: float = 0.3):
        super(BERTLogAnomalyDetector, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained BERT model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled_output
        }


class LogAnomalyTrainer:
    """Trainer class for the BERT log anomaly detector."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def load_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load training data from CSV file."""
        df = pd.read_csv(data_path)
        
        # Combine different log components for training
        texts = []
        for _, row in df.iterrows():
            # Combine timestamp, level, source, and message
            text = f"[{row['timestamp']}] [{row['level']}] [{row['source']}] {row['message']}"
            texts.append(text)
        
        labels = df['is_anomaly'].tolist()
        
        print(f"Loaded {len(texts)} log entries")
        print(f"Normal logs: {len([l for l in labels if l == 0])}")
        print(f"Anomalous logs: {len([l for l in labels if l == 1])}")
        
        return texts, labels
    
    def initialize_model(self):
        """Initialize the model, tokenizer, and optimizer."""
        model_config = self.config['model']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        
        # Initialize model
        self.model = BERTLogAnomalyDetector(
            model_name=model_config['name'],
            num_labels=model_config['num_labels']
        )
        self.model.to(self.device)
        
        print(f"Initialized model: {model_config['name']}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data_loaders(self, texts: List[str], labels: List[int], test_size: float = 0.2):
        """Prepare training and validation data loaders."""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = LogDataset(
            train_texts, train_labels, self.tokenizer, 
            max_length=self.config['model']['max_length']
        )
        val_dataset = LogDataset(
            val_texts, val_labels, self.tokenizer,
            max_length=self.config['model']['max_length']
        )
        
        # Create data loaders
        batch_size = self.config['model']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def setup_optimizer(self, train_loader):
        """Setup optimizer and learning rate scheduler."""
        model_config = self.config['model']
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=model_config['learning_rate'],
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * model_config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Get predictions
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, predictions, true_labels
    
    def train(self, data_path: str):
        """Full training pipeline."""
        print("Starting training pipeline...")
        
        # Load data
        texts, labels = self.load_data(data_path)
        
        # Initialize model
        self.initialize_model()
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(texts, labels)
        
        # Setup optimizer
        self.setup_optimizer(train_loader)
        
        # Training loop
        epochs = self.config['model']['epochs']
        best_accuracy = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Evaluate
            val_loss, val_accuracy, predictions, true_labels = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Print classification report
            if epoch == epochs - 1:  # Last epoch
                print("\nClassification Report:")
                print(classification_report(true_labels, predictions, 
                                          target_names=['Normal', 'Anomaly']))
                
                # ROC AUC if we have both classes
                if len(set(true_labels)) == 2:
                    auc = roc_auc_score(true_labels, predictions)
                    print(f"ROC AUC: {auc:.4f}")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model("models/best_model")
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
        
        # Save configuration
        with open(f"{path}/config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        # Load configuration
        with open(f"{path}/config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")
        
        # Initialize and load model
        model_config = self.config['model']
        self.model = BERTLogAnomalyDetector(
            model_name=model_config['name'],
            num_labels=model_config['num_labels']
        )
        self.model.load_state_dict(torch.load(f"{path}/model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {path}")
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Make predictions on new log entries."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded before making predictions")
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['model']['max_length'],
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Get probabilities
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                confidence = probs[0][prediction].item()
                
                results.append({
                    'text': text,
                    'prediction': prediction,
                    'confidence': confidence,
                    'is_anomaly': bool(prediction),
                    'normal_prob': probs[0][0].item(),
                    'anomaly_prob': probs[0][1].item()
                })
        
        return results


def main():
    """Main function for training or inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BERT Log Anomaly Detection")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data", type=str, default="data/synthetic_logs.csv", help="Training data path")
    parser.add_argument("--model", type=str, default="models/best_model", help="Model path")
    parser.add_argument("--predict", type=str, help="Text to predict")
    
    args = parser.parse_args()
    
    trainer = LogAnomalyTrainer()
    
    if args.train:
        # Training mode
        trainer.train(args.data)
    elif args.predict:
        # Prediction mode
        trainer.load_model(args.model)
        results = trainer.predict([args.predict])
        
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Prediction: {'Anomaly' if result['is_anomaly'] else 'Normal'}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Anomaly Probability: {result['anomaly_prob']:.4f}")
    else:
        print("Please specify --train or --predict")


if __name__ == "__main__":
    main()