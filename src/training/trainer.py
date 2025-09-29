"""
Training utilities for transformer models.

This module provides training loops, evaluation functions,
and utilities for training transformer-based language models.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


class TextDataset(Dataset):
    """
    Simple dataset class for text data.
    """
    
    def __init__(self, input_ids: List[List[int]], attention_masks: List[List[int]]):
        """
        Initialize dataset.
        
        Args:
            input_ids: List of tokenized sequences
            attention_masks: List of attention masks
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long)
        }


class LanguageModelTrainer:
    """
    Trainer class for language models.
    
    Handles training loop, evaluation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        output_dir: str = "./checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: The language model to train
            tokenizer: Tokenizer used for the model
            device: Device to train on
            output_dir: Directory to save checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Logging
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
    
    def create_optimizer(
        self, 
        learning_rate: float = 1e-4, 
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95)
    ) -> optim.Optimizer:
        """
        Create AdamW optimizer with weight decay.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            betas: Adam betas
            
        Returns:
            Configured optimizer
        """
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to bias and layer norm parameters
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters, 
            lr=learning_rate, 
            betas=betas
        )
        return optimizer
    
    def create_scheduler(
        self, 
        optimizer: optim.Optimizer, 
        num_training_steps: int,
        warmup_steps: int = 1000
    ) -> optim.lr_scheduler.LambdaLR:
        """
        Create learning rate scheduler with warmup.
        
        Args:
            optimizer: Optimizer to schedule
            num_training_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            
        Returns:
            Learning rate scheduler
        """
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, 
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        logits, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # For language modeling, labels are the same as inputs
        )
        
        return loss
    
    def train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LambdaLR] = None,
        max_grad_norm: float = 1.0
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Compute loss
            loss = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            
            # Update global step
            self.global_step += 1
            
            # Logging
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            
            progress_bar.set_postfix({'eval_loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        self.eval_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, filename: Optional[str] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.eval_losses = checkpoint.get('eval_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        print(f"Checkpoint loaded from {filepath}")
    
    @torch.no_grad()
    def generate_sample(
        self, 
        prompt: str, 
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        """
        Generate a sample text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), 
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate
        generated = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True
        )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        return generated_text
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        eval_steps: int = 500,
        save_steps: int = 1000,
        max_grad_norm: float = 1.0,
        sample_prompts: Optional[List[str]] = None
    ) -> None:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Optional evaluation dataloader
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            eval_steps: Steps between evaluations
            save_steps: Steps between checkpoints
            max_grad_norm: Maximum gradient norm
            sample_prompts: Prompts for sample generation during training
        """
        # Setup
        num_training_steps = len(train_dataloader) * num_epochs
        optimizer = self.create_optimizer(learning_rate)
        scheduler = self.create_scheduler(optimizer, num_training_steps, warmup_steps)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Total training steps: {num_training_steps}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Train
            train_loss = self.train_epoch(
                train_dataloader, 
                optimizer, 
                scheduler, 
                max_grad_norm
            )
            
            print(f"Train loss: {train_loss:.4f}")
            
            # Evaluate
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"Eval loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint("best_model.pt")
                    print(f"New best model saved (eval loss: {eval_loss:.4f})")
            
            # Generate samples
            if sample_prompts is not None:
                print("\n--- Sample generations ---")
                for prompt in sample_prompts[:3]:  # Limit to first 3 prompts
                    try:
                        generated = self.generate_sample(prompt, max_new_tokens=30)
                        print(f"Prompt: {prompt}")
                        print(f"Generated: {generated}")
                        print()
                    except Exception as e:
                        print(f"Error generating sample: {e}")
            
            # Save checkpoint
            self.save_checkpoint()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training loss
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Evaluation loss
        if self.eval_losses:
            axes[0, 1].plot(self.eval_losses)
            axes[0, 1].set_title('Evaluation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
        
        # Learning rate
        if self.learning_rates:
            axes[1, 0].plot(self.learning_rates)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].set_yscale('log')
        
        # Combined losses
        if self.eval_losses:
            axes[1, 1].plot(self.train_losses, label='Train')
            axes[1, 1].plot(self.eval_losses, label='Eval')
            axes[1, 1].set_title('Training vs Evaluation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return math.exp(loss)


def create_data_loaders(
    train_data: Dict[str, List],
    eval_data: Optional[Dict[str, List]] = None,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders from tokenized data.
    
    Args:
        train_data: Training data dictionary
        eval_data: Optional evaluation data dictionary
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    # Create datasets
    train_dataset = TextDataset(
        train_data['input_ids'], 
        train_data['attention_mask']
    )
    
    eval_dataset = None
    if eval_data is not None:
        eval_dataset = TextDataset(
            eval_data['input_ids'], 
            eval_data['attention_mask']
        )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    # Example usage
    print("Training utilities loaded successfully!")
    print("Use these utilities with your transformer model for training.")