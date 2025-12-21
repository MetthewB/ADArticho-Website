"""
Trainer for BERT/RoBERTa Humor Classification
- Freezes backbone, trains only classification head
- Supports training, validation, and testing
- Saves/loads model checkpoints
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dataloader import LABEL2ID, ID2LABEL, NUM_LABELS


def get_device(device: str = None) -> str:
    """Get the best available device: cuda > mps > cpu"""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HumorClassifier(nn.Module):
    """BERT/RoBERTa with frozen backbone and trainable classification head"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = NUM_LABELS,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen - training classification head only")

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - full fine-tuning enabled")

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class Trainer:
    """Trainer class for humor classification"""

    def __init__(
        self,
        model: HumorClassifier,
        device: str = None,
        lr: float = 2e-4,
        scheduler_type: str = "step",
    ):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Only optimize parameters that require gradients (classification head)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=lr)

        # Setup learning rate scheduler
        if scheduler_type == "step":
            self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.8)
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        else:
            self.scheduler = None

        print(f"Using device: {self.device}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"Scheduler: {scheduler_type if self.scheduler else 'None'}")

    def evaluate(self, data_loader, desc: str = "Evaluating") -> dict:
        """Evaluate model on a dataset"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            eval_pbar = tqdm(
                data_loader,
                desc=desc,
                leave=False,
                ncols=100,
                disable=len(data_loader) < 10,
            )
            for batch_idx, batch in enumerate(eval_pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar with current metrics
                current_avg_loss = total_loss / (batch_idx + 1)
                if len(all_preds) > 0:
                    current_accuracy = accuracy_score(
                        all_labels[: len(all_preds)], all_preds
                    )
                    eval_pbar.set_postfix(
                        {
                            "Loss": f"{current_avg_loss:.4f}",
                            "Acc": f"{current_accuracy:.3f}",
                            "Batch": f"{batch_idx + 1}/{len(data_loader)}",
                        }
                    )
                else:
                    eval_pbar.set_postfix({"Loss": f"{current_avg_loss:.4f}"})

        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        avg_loss = total_loss / len(data_loader)

        # Print evaluation results
        print(f"   {desc}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 5,
        save_path: str = None,
        eval_steps: int = 1000,
        save_history: bool = True,
    ) -> dict:
        """Full training loop with step-based validation"""
        best_val_acc = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "steps": [],
            "epochs": [],
            "learning_rates": [],
        }
        global_step = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train with step-based evaluation
            self.model.train()
            epoch_loss = 0
            step_loss = 0

            train_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True, ncols=100
            )
            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                step_loss += batch_loss
                global_step += 1

                # Update progress bar with current loss
                if batch_idx % 10 == 0:
                    current_loss = step_loss / min(batch_idx + 1, eval_steps)
                    train_pbar.set_postfix(
                        {"Loss": f"{current_loss:.4f}", "Step": global_step}
                    )

                # Evaluate every eval_steps
                if global_step % eval_steps == 0:
                    train_pbar.write(f"\n Evaluating at step {global_step}...")
                    avg_step_loss = step_loss / eval_steps
                    val_metrics = self.evaluate(val_loader, desc=f"Eval@{global_step}")
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    train_pbar.write(
                        f"   Step {global_step}: Train Loss: {avg_step_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | LR: {current_lr:.2e}"
                    )

                    # Save history
                    history["train_loss"].append(avg_step_loss)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_accuracy"].append(val_metrics["accuracy"])
                    history["steps"].append(global_step)
                    history["epochs"].append(epoch + 1)
                    history["learning_rates"].append(current_lr)

                    # Save only the best model (override previous best)
                    if val_metrics["accuracy"] > best_val_acc and save_path:
                        best_val_acc = val_metrics["accuracy"]
                        self.save_model(save_path)
                        train_pbar.write(
                            f"    NEW BEST MODEL! Accuracy: {best_val_acc:.4f}"
                        )

                    step_loss = 0
                    self.model.train()  # Back to training mode
                    train_pbar.write("")  # Empty line for spacing

            # Close progress bar and show epoch summary
            train_pbar.close()
            avg_epoch_loss = epoch_loss / len(train_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f" Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.2e}"
            )

            # Step scheduler at end of epoch
            if self.scheduler:
                self.scheduler.step()

        # Save training history to CSV
        if save_history and save_path:
            self.save_training_history(history, save_path)

        return history

    def test(self, test_loader) -> dict:
        """Final test evaluation with detailed report"""
        print("\n Testing...")
        metrics = self.evaluate(test_loader, desc="Testing")

        print(f"\nTest Results:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nClassification Report:")
        print(
            classification_report(
                metrics["labels"],
                metrics["predictions"],
                target_names=list(LABEL2ID.keys()),
            )
        )
        print(f"\nConfusion Matrix:")
        print(metrics["confusion_matrix"])

        return metrics

    def save_model(self, path: str):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "model_name": self.model.config._name_or_path,
                    "num_labels": NUM_LABELS,
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                },
            },
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, device: str = None) -> "Trainer":
        """Load model from checkpoint"""
        device = get_device(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint["config"]
        model = HumorClassifier(
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            freeze_backbone=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(model, device=device)
        print(f"Model loaded from {path}")
        return trainer

    def save_training_history(self, history: dict, model_path: str):
        """Save training history to CSV file"""
        model_path = Path(model_path)
        history_path = model_path.parent / f"{model_path.stem}_training_history.csv"

        # Convert to DataFrame
        df = pd.DataFrame(history)
        df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")
