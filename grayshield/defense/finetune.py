"""
Fine-Tuning Defense Baseline

Fine-tunes the model on a small clean subset as a defense mechanism.
This is a standard baseline that naturally disrupts steganographic payloads
through gradient updates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import copy


@dataclass
class FineTuneReport:
    """Report for fine-tuning defense application."""
    n_steps: int
    n_samples: int
    learning_rate: float
    initial_loss: float
    final_loss: float
    loss_reduction: float


class FineTuneDefense:
    """
    Defense that fine-tunes the model on a small clean dataset.

    This standard baseline disrupts steganographic payloads through
    natural weight updates from gradient descent, while potentially
    improving or maintaining model utility.
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
    ):
        """
        Initialize fine-tuning defense.

        Args:
            learning_rate: Learning rate for fine-tuning
            weight_decay: Weight decay (L2 regularization)
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def apply(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        n_steps: int = 100,
        target_names: Optional[List[str]] = None,
        device: str = "cuda",
        seed: int = 0,
    ) -> FineTuneReport:
        """
        Apply fine-tuning defense.

        Args:
            model: Model to fine-tune (modified in-place)
            train_loader: DataLoader with clean training samples
            n_steps: Number of fine-tuning steps
            target_names: Optional list of parameter names to update
                         If None, updates all parameters
            device: Device to use
            seed: Random seed for reproducibility

        Returns:
            FineTuneReport with training statistics
        """
        torch.manual_seed(seed)

        model.train()
        model.to(device)

        # Setup optimizer
        if target_names is not None:
            # Only fine-tune specified parameters
            params = []
            named = dict(model.named_parameters())
            for name in target_names:
                if name in named:
                    params.append(named[name])
        else:
            params = model.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Determine loss function based on model output
        criterion = nn.CrossEntropyLoss()

        # Training loop
        initial_loss = None
        final_loss = None
        n_samples = 0
        step = 0

        data_iter = iter(train_loader)

        while step < n_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Handle different batch formats
            if hasattr(batch, 'inputs') and hasattr(batch, 'labels'):
                # EvalBatch format from TaskRunner
                inputs = {k: v.to(device) for k, v in batch.inputs.items()}
                labels = batch.labels.to(device)
                outputs = model(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
            elif isinstance(batch, dict):
                # HuggingFace style batch
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch.get('labels', batch.get('label')).to(device)
                outputs = model(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # Standard (input, label) format
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            loss = criterion(logits, labels)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()
            n_samples += labels.shape[0]
            step += 1

        model.eval()

        return FineTuneReport(
            n_steps=n_steps,
            n_samples=n_samples,
            learning_rate=self.learning_rate,
            initial_loss=initial_loss or 0.0,
            final_loss=final_loss or 0.0,
            loss_reduction=(initial_loss - final_loss) if initial_loss and final_loss else 0.0,
        )


def create_finetune_loader(
    task_runner: Any,
    n_samples: int = 256,
    batch_size: int = 16,
    seed: int = 42,
) -> DataLoader:
    """
    Create a small clean dataset loader for fine-tuning.

    Args:
        task_runner: TaskRunner instance with dataset access
        n_samples: Number of samples to use
        batch_size: Batch size
        seed: Random seed for reproducibility

    Returns:
        DataLoader with clean training samples
    """
    # This function will be called from the runner with task_runner
    # that has access to the training dataset
    return task_runner.make_train_loader(n_samples, batch_size, seed=seed)
