"""
Data Splitting and DataLoader for Humor Classification
- Loads labeled JSON
- Splits into train/val/test
- Creates PyTorch Dataset and DataLoader with BERT/RoBERTa tokenization
"""

import json
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


# Humor type to index mapping (6 categories, no 'none')
LABEL2ID = {
    'affiliative': 0,
    'sexual': 1,
    'offensive': 2,
    'irony_satire': 3,
    'absurdist': 4,
    'dark': 5
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def load_labeled_json(path: str, min_confidence: float = None) -> List[Dict]:
    """Load labeled JSON and optionally filter by confidence"""
    data = json.load(open(path))
    if min_confidence is not None:
        data = [d for d in data if d.get('confidence', 1.0) >= min_confidence]
    
    # Filter out invalid samples
    valid_data = []
    for d in data:
        # Check if humor_type is valid
        if d.get('humor_type') not in LABEL2ID:
            continue
        
        # Check if caption is a valid string
        caption = d.get('caption')
        if not caption or not isinstance(caption, str) or len(caption.strip()) == 0:
            continue
            
        # Check if required fields exist
        if not d.get('id'):
            continue
            
        valid_data.append(d)
    
    print(f"Filtered data: {len(data)} → {len(valid_data)} valid samples")
    return valid_data


def split_data(data: List[Dict], test_size: float = 0.1, val_size: float = 0.1, seed: int = 42) -> Tuple[List, List, List]:
    """Split data into train/val/test"""
    labels = [d['humor_type'] for d in data]
    
    # First split: train+val vs test
    train_val, test = train_test_split(data, test_size=test_size, random_state=seed, stratify=labels)
    
    # Second split: train vs val
    train_val_labels = [d['humor_type'] for d in train_val]
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed, stratify=train_val_labels)
    
    return train, val, test


class HumorDataset(Dataset):
    """PyTorch Dataset for humor classification"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        caption = item['caption']
        label = LABEL2ID[item['humor_type']]
        
        # Additional safety check for caption
        if not isinstance(caption, str):
            caption = str(caption)
        caption = caption.strip()
        
        # Construct input text with context if available
        input_text = caption
        
        # Add image description/context if available (like in Gemini processing)
        if 'image_description' in item and item['image_description']:
            context = item['image_description'].strip()
            input_text = f"Image: {context} Caption: {caption}"
        elif 'context' in item and item['context']:
            context = item['context'].strip()
            input_text = f"Context: {context} Caption: {caption}"
        elif 'description' in item and item['description']:
            context = item['description'].strip()
            input_text = f"Description: {context} Caption: {caption}"
        
        # Debug: print text length for first few items to verify longer sequences
        if idx < 3:
            tokens = self.tokenizer.tokenize(input_text)
            print(f"Sample {idx}: {len(tokens)} tokens - {input_text[:100]}...")
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'id': item.get('id', idx),
            'caption': caption
        }


def create_dataloaders(
    data_path: str,
    model_name: str = 'bert-base-uncased',
    batch_size: int = 64,
    max_length: int = 128,
    min_confidence: float = None,
    test_size: float = 0.05,
    val_size: float = 0.05,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """Create train/val/test DataLoaders"""
    
    # Load and split data
    data = load_labeled_json(data_path, min_confidence)
    train_data, val_data, test_data = split_data(data, test_size, val_size, seed)
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = HumorDataset(train_data, tokenizer, max_length)
    val_dataset = HumorDataset(val_data, tokenizer, max_length)
    test_dataset = HumorDataset(test_data, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, tokenizer

