#!/usr/bin/env python3
"""
Example training script for the transformer language model.

This script demonstrates how to train a small language model
on sample text data using the implemented transformer architecture.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.language_model import create_small_language_model
from utils.tokenizer import SimpleTokenizer, create_simple_dataset
from training.trainer import LanguageModelTrainer, create_data_loaders


def load_sample_data():
    """
    Load sample training data.
    
    In a real scenario, you would load your dataset here.
    For demonstration, we use a collection of sample texts.
    """
    sample_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models have shown remarkable capabilities in text generation.",
        "PyTorch provides powerful tools for implementing deep learning models.",
        "Machine learning continues to advance at an unprecedented pace.",
        "Natural language understanding requires sophisticated neural architectures.",
        "The attention is all you need paper introduced a breakthrough approach.",
        "Deep learning models can learn complex patterns from data.",
        "Text generation has improved significantly with transformer models.",
        "Neural networks have become increasingly powerful and versatile.",
        "Artificial intelligence is transforming many aspects of technology.",
        "Language models can generate coherent and contextually relevant text.",
        "The field of NLP has seen tremendous progress in recent years.",
        "Transformer models have achieved state-of-the-art results on many tasks.",
        "Understanding attention mechanisms is crucial for modern AI research.",
        "Large-scale training requires careful optimization and engineering.",
        "The scalability of transformer architectures is remarkable.",
        "Self-attention allows for parallel processing of sequences.",
        "Positional encoding helps models understand sequence order.",
        "Multi-head attention captures different types of relationships.",
    ]
    
    # Extend the dataset by repeating and slightly modifying texts
    extended_texts = []
    for text in sample_texts:
        extended_texts.append(text)
        # Add some variations
        extended_texts.append(text + " This demonstrates the power of neural networks.")
        extended_texts.append("In recent developments, " + text.lower())
        extended_texts.append(text + " Such advances continue to shape the future.")
    
    return extended_texts


def main():
    """Main training function."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load sample data
    print("Loading sample data...")
    texts = load_sample_data()
    print(f"Loaded {len(texts)} training samples")
    
    # Create tokenizer and build vocabulary
    print("Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=2000, min_freq=1)
    tokenizer.build_vocabulary(texts)
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_simple_dataset(texts, tokenizer, max_length=128)
    
    # Split into train/validation
    split_idx = int(0.8 * len(dataset['input_ids']))
    train_data = {
        'input_ids': dataset['input_ids'][:split_idx],
        'attention_mask': dataset['attention_mask'][:split_idx]
    }
    eval_data = {
        'input_ids': dataset['input_ids'][split_idx:],
        'attention_mask': dataset['attention_mask'][split_idx:]
    }
    
    print(f"Train samples: {len(train_data['input_ids'])}")
    print(f"Eval samples: {len(eval_data['input_ids'])}")
    
    # Create data loaders
    train_dataloader, eval_dataloader = create_data_loaders(
        train_data, eval_data, batch_size=8, shuffle=True
    )
    
    # Create model
    print("Creating model...")
    model = create_small_language_model(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_len=128,
        dropout=0.1
    )
    
    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create trainer
    trainer = LanguageModelTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir="./checkpoints"
    )
    
    # Sample prompts for generation during training
    sample_prompts = [
        "The transformer architecture",
        "Machine learning",
        "Natural language processing",
        "Deep learning models"
    ]
    
    # Train the model
    print("Starting training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=10,
        learning_rate=1e-4,
        warmup_steps=100,
        eval_steps=50,
        save_steps=100,
        sample_prompts=sample_prompts
    )
    
    # Generate some sample text
    print("\n" + "="*50)
    print("SAMPLE GENERATIONS")
    print("="*50)
    
    for prompt in sample_prompts[:3]:
        print(f"\nPrompt: '{prompt}'")
        try:
            generated = trainer.generate_sample(
                prompt, 
                max_new_tokens=50, 
                temperature=0.8,
                top_k=50
            )
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"Error generating sample: {e}")
    
    # Plot training curves
    print("\nPlotting training curves...")
    try:
        trainer.plot_training_curves("./training_curves.png")
    except Exception as e:
        print(f"Error plotting curves: {e}")
    
    # Save tokenizer
    tokenizer.save("./tokenizer.pkl")
    print("Tokenizer saved to ./tokenizer.pkl")
    
    print("Training completed!")


if __name__ == "__main__":
    main()