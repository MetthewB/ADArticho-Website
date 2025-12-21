#!/usr/bin/env python3
"""
Main Training Script for Humor Classification
Unified interface for both BERT fine-tuning and embedding-based approaches

Usage:
# BERT fine-tuning
python main.py --method bert --input ../data/human_verified/labeled_nycc_100k_clean.json --model roberta-large --epochs 5

# Embedding + simple classifier
python main.py --method embedding --input ../data/human_verified/labeled_nycc_100k_clean.json --embedding-model all-MiniLM-L6-v2 --classifier logistic

# Compare all embedding approaches
python main.py --method embedding --input ../data/human_verified/labeled_nycc_100k_clean.json --compare
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Humor classification training")

    # Common arguments
    parser.add_argument("--input", required=True, help="Path to labeled JSON file")
    parser.add_argument(
        "--method", required=True, choices=["bert", "embedding"], help="Training method"
    )
    parser.add_argument(
        "--model-dir", default="./models", help="Directory to save models"
    )
    parser.add_argument(
        "--min-confidence", type=float, help="Min confidence for data filtering"
    )

    # BERT arguments
    parser.add_argument(
        "--model", default="roberta-large", help="BERT/RoBERTa model name"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--max-length", type=int, default=256, help="Max sequence length"
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", help="Freeze backbone"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=200, help="Evaluation frequency"
    )
    parser.add_argument("--scheduler", default="cosine", help="LR scheduler")

    # Embedding arguments
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model"
    )
    parser.add_argument(
        "--classifier",
        default="logistic",
        choices=["logistic", "rf"],
        help="Classifier",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare all combinations"
    )
    parser.add_argument(
        "--cache-dir", default="./embeddings_cache", help="Embeddings cache dir"
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create directories
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)

    if args.method == "bert":
        from bert_train import train_bert

        print("=" * 60)
        print("BERT/RoBERTa Fine-tuning")
        print("=" * 60)

        results = train_bert(
            input_path=args.input,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            model_dir=args.model_dir,
            min_confidence=args.min_confidence,
            freeze_backbone=args.freeze_backbone,
            eval_steps=args.eval_steps,
            scheduler=args.scheduler,
        )

    elif args.method == "embedding":
        Path(args.cache_dir).mkdir(exist_ok=True, parents=True)

        print("=" * 60)
        print("Embedding + Simple Classifier")
        print("=" * 60)

        if args.compare:
            from embedding_train import compare_embedding_models

            results = compare_embedding_models(args.input)
        else:
            from embedding_train import train_embedding

            results = train_embedding(
                input_path=args.input,
                embedding_model=args.embedding_model,
                classifier=args.classifier,
                model_dir=args.model_dir,
                min_confidence=args.min_confidence,
                cache_dir=args.cache_dir,
            )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
