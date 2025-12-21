"""
Embedding-based Trainer for Humor Classification
Uses sentence transformers to create embeddings, then trains simple classifiers
"""

import numpy as np
import pickle
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

from dataloader import LABEL2ID, ID2LABEL, load_labeled_json, split_data


class EmbeddingTrainer:
    """
    Trainer for embedding-based humor classification
    Computes embeddings using sentence transformers and trains simple classifiers
    """
    
    def __init__(self, 
                 embedding_model_name='all-MiniLM-L6-v2',
                 classifier_type='logistic',
                 cache_dir='./embeddings_cache',
                 device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.classifier_type = classifier_type
        self.cache_dir = Path(cache_dir)
        self.device = device
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize classifier
        self.classifier = self._get_classifier()
        self.is_trained = False
        
    def _get_classifier(self):
        """Initialize the appropriate classifier"""
        if self.classifier_type == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        elif self.classifier_type == 'svm':
            return LinearSVC(random_state=42, class_weight='balanced', max_iter=2000)
        elif self.classifier_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def _get_cache_path(self, data_split, model_name, dataset_name):
        """Generate cache file path"""
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        return self.cache_dir / f"{dataset_name}_{data_split}_{safe_model_name}_embeddings.pkl"
    
    def compute_and_cache_embeddings(self, data: List[Dict], data_split='train', dataset_name='humor'):
        """Compute embeddings and cache them"""
        model_name = self.embedding_model._modules['0'].auto_model.name_or_path
        cache_path = self._get_cache_path(data_split, model_name, dataset_name)
        
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['embeddings'], cached_data['labels']
        
        print(f"Computing embeddings for {len(data)} {data_split} samples...")
        
        # Extract texts and labels
        texts = [item['caption'] for item in data]
        labels = [LABEL2ID[item['humor_type']] for item in data]
        
        embeddings = []
        batch_size = 32
        
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing {data_split} embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        # Cache the embeddings
        cache_data = {
            'embeddings': embeddings,
            'labels': labels,
            'model_name': model_name,
            'embedding_dim': embeddings.shape[1],
            'num_samples': len(embeddings)
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cached embeddings to {cache_path} (shape: {embeddings.shape})")
        return embeddings, labels
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None, dataset_name: str = 'humor'):
        """Train the embedding-based classifier"""
        
        # Compute embeddings for training data
        train_embeddings, train_labels = self.compute_and_cache_embeddings(
            train_data, 'train', dataset_name
        )
        
        # Compute embeddings for validation data if provided
        val_embeddings, val_labels = None, None
        if val_data:
            val_embeddings, val_labels = self.compute_and_cache_embeddings(
                val_data, 'val', dataset_name
            )
        
        # Train classifier
        print(f"Training {self.classifier_type} classifier on {len(train_embeddings)} samples...")
        print(f"Embedding dimension: {train_embeddings.shape[1]}")
        
        # Print class distribution
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        print("Training class distribution:")
        for label_idx, count in zip(unique_labels, counts):
            print(f"  {ID2LABEL[label_idx]}: {count} samples")
        
        self.classifier.fit(train_embeddings, train_labels)
        self.is_trained = True
        
        # Evaluate on training set
        train_preds = self.classifier.predict(train_embeddings)
        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Training accuracy: {train_acc:.4f}")
        
        # Evaluate on validation set if provided
        val_acc = None
        if val_embeddings is not None and val_labels is not None:
            val_preds = self.classifier.predict(val_embeddings)
            val_acc = accuracy_score(val_labels, val_preds)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            print("\nValidation Classification Report:")
            report = classification_report(val_labels, val_preds, target_names=list(ID2LABEL.values()), digits=4)
            print(report)
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model_name': self.embedding_model._modules['0'].auto_model.name_or_path,
            'classifier_type': self.classifier_type
        }
    
    def evaluate(self, test_data: List[Dict], dataset_name: str = 'humor'):
        """Evaluate the trained classifier on test data"""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before evaluation")
        
        # Compute embeddings for test data
        test_embeddings, test_labels = self.compute_and_cache_embeddings(
            test_data, 'test', dataset_name
        )
        
        # Make predictions
        test_preds = self.classifier.predict(test_embeddings)
        test_acc = accuracy_score(test_labels, test_preds)
        
        print(f"\n{'='*50}")
        print(f"Test Results ({self.classifier_type})")
        print(f"{'='*50}")
        print(f"Test accuracy: {test_acc:.4f}")
        
        print("\nTest Classification Report:")
        report = classification_report(test_labels, test_preds, target_names=list(ID2LABEL.values()), digits=4)
        print(report)
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, test_preds)
        print(cm)
        
        return {
            'test_accuracy': test_acc,
            'predictions': test_preds,
            'true_labels': test_labels,
            'confusion_matrix': cm
        }
    
    def predict_single(self, text: str):
        """Predict humor type for a single text"""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        # Compute embedding
        embedding = self.embedding_model.encode([text], convert_to_tensor=False)
        
        # Make prediction
        pred_idx = self.classifier.predict(embedding)[0]
        
        # Get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(embedding)[0]
            prob_dict = {ID2LABEL[i]: float(prob) for i, prob in enumerate(probs)}
        else:
            prob_dict = None
        
        return {
            'predicted_humor_type': ID2LABEL[pred_idx],
            'predicted_label_id': int(pred_idx),
            'probabilities': prob_dict
        }
    
    def save_model(self, save_path: str):
        """Save the trained classifier"""
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        if not self.is_trained:
            raise ValueError("Classifier must be trained before saving")
        
        # Save classifier
        classifier_path = save_path.with_suffix('.pkl')
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Save metadata
        metadata = {
            'classifier_type': self.classifier_type,
            'embedding_model': self.embedding_model._modules['0'].auto_model.name_or_path,
            'embedding_dim': self.embedding_model.get_sentence_embedding_dimension(),
            'label_mapping': {
                'label2id': LABEL2ID,
                'id2label': ID2LABEL
            }
        }
        
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved model to {classifier_path}")
        print(f"Saved metadata to {metadata_path}")
    
    def load_model(self, load_path: str):
        """Load a trained classifier"""
        load_path = Path(load_path)
        
        # Load classifier
        classifier_path = load_path.with_suffix('.pkl')
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Load and verify metadata
        metadata_path = load_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify model compatibility
            if metadata['embedding_model'] != self.embedding_model._modules['0'].auto_model.name_or_path:
                print(f"Warning: Loaded model was trained with {metadata['embedding_model']}, "
                      f"but current embedding model is {self.embedding_model._modules['0'].auto_model.name_or_path}")
        
        self.is_trained = True
        print(f"Loaded model from {classifier_path}")


def compare_classifiers(data_path: str, embedding_models: List[str] = None, classifiers: List[str] = None):
    """Compare different embedding models and classifiers"""
    
    if embedding_models is None:
        embedding_models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
    
    if classifiers is None:
        classifiers = ['logistic', 'svm', 'rf']
    
    # Load and split data
    print(f"Loading data from {data_path}")
    data = load_labeled_json(data_path)
    train_data, val_data, test_data = split_data(data, test_size=0.15, val_size=0.15)
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    results = {}
    dataset_name = Path(data_path).stem
    
    for embedding_model in embedding_models:
        for classifier_type in classifiers:
            print(f"\n{'='*80}")
            print(f"Training: {embedding_model} + {classifier_type}")
            print(f"{'='*80}")
            
            trainer = EmbeddingTrainer(
                embedding_model_name=embedding_model,
                classifier_type=classifier_type,
                cache_dir='./embeddings_cache'
            )
            
            # Train
            train_results = trainer.train(train_data, val_data, dataset_name)
            
            # Test
            test_results = trainer.evaluate(test_data, dataset_name)
            
            # Store results
            key = f"{embedding_model}_{classifier_type}"
            results[key] = {
                **train_results,
                **test_results
            }
            
            # Save model
            model_path = Path(f"./models/{dataset_name}_{embedding_model.replace('/', '_')}_{classifier_type}")
            trainer.save_model(model_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    
    for key, result in results.items():
        print(f"{key:40} | Val: {result['val_accuracy']:.4f} | Test: {result['test_accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    # Quick test
    data_path = "../data/human_verified/labeled_nycc_100k_clean.json"
    if Path(data_path).exists():
        results = compare_classifiers(data_path)
    else:
        print(f"Data file not found: {data_path}")
        print("Please update the path to your labeled data file")