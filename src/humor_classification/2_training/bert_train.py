"""
BERT Training Module
Simple interface for BERT/RoBERTa fine-tuning
"""

from pathlib import Path
from dataloader import create_dataloaders
from bert_trainer import HumorClassifier, Trainer, predict_single


def train_bert(input_path, model_name='roberta-large', epochs=5, batch_size=16, 
               lr=1e-5, max_length=256, model_dir='./models', min_confidence=None,
               freeze_backbone=True, eval_steps=200, scheduler='cosine'):
    """
    Train BERT/RoBERTa model for humor classification
    """
    print(f"Training BERT model: {model_name}")
    
    # Paths
    input_path = Path(input_path)
    model_dir = Path(model_dir)
    
    # Extract dataset name from input file
    dataset_name = input_path.stem.split('_')[-1]
    model_path = model_dir / f"{model_name.replace('/', '_')}_{dataset_name}_humor_classifier.pt"
    tokenizer_path = model_dir / f"{model_name.replace('/', '_')}_{dataset_name}_tokenizer"
    
    # Create data loaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        str(input_path), model_name, batch_size=batch_size, max_length=max_length,
        min_confidence=min_confidence
    )
    
    print(f"Data loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # Initialize model and trainer
    model = HumorClassifier(model_name, freeze_backbone=freeze_backbone)
    trainer = Trainer(model, learning_rate=lr, scheduler_type=scheduler)
    
    # Train
    history, test_metrics = trainer.train(
        train_loader, val_loader, epochs=epochs, 
        save_path=str(model_path), eval_steps=eval_steps
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_path)
    
    # Test evaluation
    trainer.load_model(str(model_path))
    test_metrics = trainer.evaluate(test_loader, desc="Final Test")
    
    # Demo predictions
    demo_texts = [
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Life is short. Smile while you still have teeth.",
        "Politicians and diapers have one thing in common: they both need changing regularly."
    ]
    
    print("\n" + "=" * 60)
    print("Demo Predictions:")
    for text in demo_texts:
        result = predict_single(trainer, text, tokenizer)
        print(f"\n📝 \"{text[:50]}...\"")
        print(f"   → {result['humor_type']} ({result['confidence']:.2%})")
    
    print(f"\nModel saved to: {model_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'model_path': str(model_path),
        'tokenizer_path': str(tokenizer_path)
    }