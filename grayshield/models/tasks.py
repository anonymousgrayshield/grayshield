from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

@dataclass
class EvalBatch:
    inputs: Dict[str, torch.Tensor]
    labels: torch.Tensor

class TaskRunner:
    '''
    Unified evaluation:
    - text: GLUE SST-2, IMDB
    - vision: CIFAR-10
    '''
    def __init__(self, task: str, preset_task_type: str, processor, device: str = "auto", min_free_gb: float = 2.0):
        self.task = task
        self.task_type = preset_task_type
        self.processor = processor
        # self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.device = "cpu" # 預設值

        if torch.cuda.is_available():
            if device == "auto":
                num_gpus = torch.cuda.device_count()
                best_gpu = 0
                best_free = -1

                print(f"🔍 Detecting {num_gpus} GPUs for auto-selection...")
                
                for i in range(num_gpus):
                    free, total = torch.cuda.mem_get_info(i)
                    free_gb = free / (1024**3)
                    total_gb = total / (1024**3)
                    print(f"    GPU {i}: free={free_gb:.2f}GB / total={total_gb:.2f}GB")

                    if free > best_free:
                        best_free = free
                        best_gpu = i

                
                current_free_gb = best_free / (1024**3)
                if current_free_gb < min_free_gb:
                    raise RuntimeError(
                        f"No GPU meets min_free_gb={min_free_gb}. Best free={current_free_gb:.2f}GB"
                    )

                self.device = f"cuda:{best_gpu}"
                print(f" Auto-selected device: {self.device} (Most free memory)")
            
            elif "cuda" in device:
                
                self.device = device
                print(f" Using specified device: {self.device}")
            else:
                self.device = "cpu"
                print(" CUDA available but not used (device set to cpu)")
        else:
            self.device = "cpu"
            print(" CUDA not available, using CPU.")

    def make_eval_loader(
        self,
        n_eval: int,
        batch_size: int,
        full_eval: bool = False,
        eval_seed: int = 42,
    ) -> DataLoader:
        """Create evaluation DataLoader.

        Args:
            n_eval: Number of samples to evaluate (ignored if full_eval=True)
            batch_size: Batch size for DataLoader
            full_eval: If True, use the entire validation/test split
            eval_seed: Seed for deterministic sampling when not using full_eval

        Returns:
            DataLoader with evaluation batches
        """
        if self.task_type == "text":
            if self.task == "sst2":
                ds = load_dataset("glue", "sst2", split="validation")
                if not full_eval and n_eval < len(ds):
                    # Use shuffled selection with fixed seed for reproducibility
                    ds = ds.shuffle(seed=eval_seed).select(range(n_eval))

                def collate(batch):
                    texts = [x["sentence"] for x in batch]
                    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
                    enc = self.processor(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                    return EvalBatch(inputs=dict(enc), labels=labels)

                return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

            elif self.task == "imdb":
                ds = load_dataset("imdb", split="test")
                if not full_eval and n_eval < len(ds):
                    ds = ds.shuffle(seed=eval_seed).select(range(n_eval))

                def collate(batch):
                    texts = [x["text"] for x in batch]
                    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
                    enc = self.processor(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    return EvalBatch(inputs=dict(enc), labels=labels)

                return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

            else:
                raise ValueError(f"Supported text tasks: sst2, imdb. Got: {self.task}")

        if self.task_type == "vision":
            if self.task != "cifar10":
                raise ValueError(f"Supported vision task: cifar10. Got: {self.task}")
            ds = load_dataset("cifar10", split="test")
            if not full_eval and n_eval < len(ds):
                ds = ds.shuffle(seed=eval_seed).select(range(n_eval))

            def collate(batch):
                images = [x["img"] for x in batch]
                labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
                enc = self.processor(images=images, return_tensors="pt")
                return EvalBatch(inputs=dict(enc), labels=labels)

            return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

        raise ValueError(f"Unknown task_type: {self.task_type}")

    def make_train_loader(
        self,
        n_train: int,
        batch_size: int,
        seed: int = 42,
    ) -> DataLoader:
        """Create training DataLoader for fine-tuning defense.

        Args:
            n_train: Number of training samples
            batch_size: Batch size for DataLoader
            seed: Seed for deterministic sampling

        Returns:
            DataLoader with training batches
        """
        if self.task_type == "text":
            if self.task == "sst2":
                ds = load_dataset("glue", "sst2", split="train")
                if n_train < len(ds):
                    ds = ds.shuffle(seed=seed).select(range(n_train))

                def collate(batch):
                    texts = [x["sentence"] for x in batch]
                    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
                    enc = self.processor(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                    return EvalBatch(inputs=dict(enc), labels=labels)

                return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

            elif self.task == "imdb":
                ds = load_dataset("imdb", split="train")
                if n_train < len(ds):
                    ds = ds.shuffle(seed=seed).select(range(n_train))

                def collate(batch):
                    texts = [x["text"] for x in batch]
                    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
                    enc = self.processor(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    return EvalBatch(inputs=dict(enc), labels=labels)

                return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

            else:
                raise ValueError(f"Supported text tasks: sst2, imdb. Got: {self.task}")

        if self.task_type == "vision":
            if self.task != "cifar10":
                raise ValueError(f"Supported vision task: cifar10. Got: {self.task}")
            ds = load_dataset("cifar10", split="train")
            if n_train < len(ds):
                ds = ds.shuffle(seed=seed).select(range(n_train))

            def collate(batch):
                images = [x["img"] for x in batch]
                labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
                enc = self.processor(images=images, return_tensors="pt")
                return EvalBatch(inputs=dict(enc), labels=labels)

            return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

        raise ValueError(f"Unknown task_type: {self.task_type}")

    @torch.no_grad()
    def evaluate_accuracy(self, model: torch.nn.Module, loader: DataLoader) -> float:
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        for batch in tqdm(loader, desc="eval", leave=False):
            inputs = {k: v.to(self.device) for k, v in batch.inputs.items()}
            labels = batch.labels.to(self.device)
            out = model(**inputs)
            pred = out.logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
        return float(accuracy_score(y_true, y_pred))

    def loss_and_backward(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        n_batches: int = 1,
        use_eval_mode: bool = True,
    ) -> float:
        """
        Compute loss and gradients on a small number of batches for sensitivity estimation.

        Args:
            model: Model to compute gradients for
            loader: DataLoader with evaluation batches
            n_batches: Number of batches to use
            use_eval_mode: If True (default), use eval mode to disable dropout for
                          reproducible gradient computation. Gradients are still computed.

        Returns:
            Average loss over the batches
        """
        # Save original training state
        was_training = model.training

        if use_eval_mode:
            # Use eval mode to disable dropout for reproducibility
            # Gradients can still be computed in eval mode
            model.eval()
        else:
            model.train()

        loss_fn = torch.nn.CrossEntropyLoss()
        total = 0.0
        count = 0

        # Clear existing gradients before accumulation
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        for batch in loader:
            inputs = {k: v.to(self.device) for k, v in batch.inputs.items()}
            labels = batch.labels.to(self.device)

            # Forward pass (gradients enabled even in eval mode)
            out = model(**inputs)
            loss = loss_fn(out.logits, labels)

            # Backward pass to accumulate gradients
            loss.backward()

            total += float(loss.detach().cpu())
            count += 1
            if count >= n_batches:
                break

        # Restore original training state
        if was_training:
            model.train()
        else:
            model.eval()

        return total / max(1, count)
