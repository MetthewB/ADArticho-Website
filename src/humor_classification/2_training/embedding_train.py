"""
Embedding Training Module
Simple interface for embedding-based humor classification
"""

from pathlib import Path
from dataloader import load_labeled_json, split_data
from embedding_trainer import EmbeddingTrainer, compare_classifiers


def train_embedding(
    input_path,
    embedding_model="all-MiniLM-L6-v2",
    classifier="logistic",
    model_dir="./models",
    min_confidence=None,
    cache_dir="./embeddings_cache",
):
    """
    Train embedding-based model for humor classification
    """
    print(f"Training embedding model: {embedding_model} + {classifier}")

    # Load and split data
    data = load_labeled_json(input_path, min_confidence=min_confidence)
    train_data, val_data, test_data = split_data(data, test_size=0.15, val_size=0.15)

    dataset_name = Path(input_path).stem

    # Train
    trainer = EmbeddingTrainer(
        embedding_model_name=embedding_model,
        classifier_type=classifier,
        cache_dir=cache_dir,
    )

    train_results = trainer.train(train_data, val_data, dataset_name)
    test_results = trainer.evaluate(test_data, dataset_name)

    # Save model
    model_path = (
        Path(model_dir)
        / f"{dataset_name}_{embedding_model.replace('/', '_')}_{classifier}"
    )
    model_path.parent.mkdir(exist_ok=True, parents=True)
    trainer.save_model(model_path)

    # Demo predictions
    demo_texts = [
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Life is short. Smile while you still have teeth.",
        "Politicians and diapers have one thing in common: they both need changing regularly.",
    ]

    print("\n" + "=" * 60)
    print("Demo Predictions:")
    for text in demo_texts:
        result = trainer.predict_single(text)
        print(f'\n "{text[:50]}..."')
        print(f"   → {result['predicted_humor_type']}")
        if result["probabilities"]:
            top_3 = sorted(
                result["probabilities"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            for humor_type, prob in top_3:
                print(f"     {humor_type}: {prob:.3f}")

    print(f"\nModel saved to: {model_path}")

    return {**train_results, **test_results, "model_path": str(model_path)}


def compare_embedding_models(input_path, embedding_models=None, classifiers=None):
    """
    Compare different embedding models and classifiers
    """
    if embedding_models is None:
        embedding_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

    if classifiers is None:
        classifiers = ["logistic", "rf"]

    print("Comparing embedding models and classifiers...")
    results = compare_classifiers(
        input_path, embedding_models=embedding_models, classifiers=classifiers
    )

    return results
