"""
Main Script for LLM-based Humor Labeling using Gemini via LangChain
python label_data.py --input ../data/llm_labeled/unlabeled_nycc_10k.json --model gemini-2.5-flash-lite --limit 100
"""

import argparse
from pathlib import Path
from datetime import datetime
from gemini_client import GeminiClient
from dataloader import DataLoader, save_results, collate_fn
import json


class SampleIterator:
    """Iterator that yields batches of (samples, inputs) for processing"""
    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
    
    def __iter__(self):
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i:i + self.batch_size]
            inputs = collate_fn(batch)
            yield batch, inputs
    
    def __len__(self):
        import math
        return math.ceil(len(self.samples) / self.batch_size)


def main(args):
    # Paths
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.parent / f"labeled_{input_path.stem.replace('unlabeled_', '')}_{timestamp}.json"
    checkpoint_path = output_path.parent / f"checkpoint_{output_path.stem}.json"
    
    # Load data
    print(f"Loading: {input_path}")
    all_data = json.load(open(input_path))
    
    # Check for existing checkpoint and resume if found
    checkpoint_results = []
    processed_ids = set()
    
    if checkpoint_path.exists():
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_results = json.load(open(checkpoint_path))
        processed_ids = {item.get('id') for item in checkpoint_results if item.get('id')}
        print(f"   Already processed: {len(checkpoint_results)} samples")
        
        # Filter out already processed samples
        remaining_data = [item for item in all_data if item.get('id') not in processed_ids]
        print(f"   Remaining to process: {len(remaining_data)} samples")
    else:
        remaining_data = all_data
        print(f"   No checkpoint found - starting fresh")
        print(f"   Total samples: {len(remaining_data)}")
    
    if not remaining_data:
        print("All samples already processed! Using checkpoint as final result.")
        save_results(checkpoint_results, output_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"Output: {output_path}")
        return
    
    # Apply limit if specified
    if args.limit:
        remaining_data = remaining_data[:args.limit]
        print(f"   Limited to: {len(remaining_data)} samples")
    
    # Initialize client
    print(f"Initializing Gemini client: {args.model}")
    client = GeminiClient(model_name=args.model)
    
    # Create iterator for remaining data
    data_iter = SampleIterator(remaining_data, args.batch_size)
    
    # Checkpoint function
    def save_checkpoint(results):
        combined = checkpoint_results + results
        save_results(combined, checkpoint_path)
        print(f"   Checkpoint saved: {len(combined)} samples")
    
    # Process data
    results, processed, failed, none_samples = client.classify_batch(
        data_iter,
        checkpoint_fn=save_checkpoint,
        checkpoint_every=args.checkpoint_every
    )
    
    # Combine all results
    final_results = checkpoint_results + results
    
    # Save final results
    save_results(final_results, output_path)
    
    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {len(final_results)} samples")
    print(f"Failed: {failed} samples")
    print(f"Output: {output_path}")
    
    # Usage summary
    usage = client.get_usage()
    print(f"\nToken Usage:")
    print(f"   Model: {usage['model']}")
    print(f"   Total cost: ${usage['total_cost']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label humor data using Gemini")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Model name")
    parser.add_argument("--limit", type=int, help="Limit samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint every N batches")
    
    args = parser.parse_args()
    main(args)
