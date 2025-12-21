"""
DataLoader for handling humor dataset loading and management
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """Prepare batch inputs for the chain"""
    return [
        {
            "caption": s['caption'],
            "image_description": s.get('image_description', '') or "Not provided",
            "uncanny_description": s.get('uncanny_description', '') or "Not provided"
        }
        for s in batch
    ]


class DataLoader:
    """Handle loading and managing humor dataset samples"""
    
    def __init__(self, file_path: str, limit: Optional[int] = None, batch_size: int = 10):
        """Initialize DataLoader with file path, optional limit, and batch size"""
        self.file_path = Path(file_path)
        self.limit = limit
        self.batch_size = batch_size
        self.samples = []
        self._load()
    
    def _load(self):
        """Load samples from file (only up to limit)"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Only keep up to limit samples
            self.samples = data[:self.limit] if self.limit else data
    
    def get_samples(self) -> List[Dict]:
        """Get all samples"""
        return self.samples
    
    def __len__(self):
        """Return number of batches"""
        import math
        return math.ceil(len(self.samples) / self.batch_size)
    
    def num_samples(self):
        """Return total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Support indexing and slicing"""
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[tuple[List[Dict], List[Dict]]]:
        """Iterate over batches, yielding (raw_batch, collated_inputs)"""
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i:i + self.batch_size]
            inputs = collate_fn(batch)
            yield batch, inputs


def save_results(results: List[Dict], output_path: Path):
    """Save labeled results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_checkpoint(checkpoint_path: Path) -> tuple[List[Dict], int]:
    """Load checkpoint if exists"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results, len(results)
    return [], 0
