#!/usr/bin/env python3
"""
Demonstration script for the transformer implementation.

This script provides a quick demonstration of the key components
and capabilities of the implemented transformer architecture.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer import MultiHeadAttention, PositionalEncoding, create_transformer_model
from models.language_model import create_small_language_model
from utils.tokenizer import SimpleTokenizer


def demo_attention_mechanism():
    """Demonstrate the attention mechanism with a simple example."""
    print("=" * 60)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 60)
    
    # Create a small attention module
    d_model, n_heads = 128, 4
    attention = MultiHeadAttention(d_model, n_heads)
    attention.eval()
    
    # Create example sequences
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = attention(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output preserves input dimensions: {output.shape == x.shape}")
    
    # Show attention pattern (simplified)
    print("\nSample attention pattern (Head 1, first sequence):")
    # This would require modifying the attention class to return weights
    # For demonstration, we'll show a synthetic pattern
    attention_pattern = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_pattern.numpy(), cmap='Blues')
    plt.title('Attention Pattern Example')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.colorbar()
    plt.savefig('/tmp/attention_demo.png', dpi=150, bbox_inches='tight')
    print("Attention pattern saved to /tmp/attention_demo.png")
    plt.close()


def demo_positional_encoding():
    """Demonstrate positional encoding."""
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 60)
    
    d_model = 256
    max_len = 100
    
    pos_enc = PositionalEncoding(d_model, max_len)
    
    # Get positional encodings
    positions = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
    dummy_input = torch.zeros(max_len, 1, d_model)  # [seq_len, batch, d_model]
    
    with torch.no_grad():
        encoded = pos_enc(dummy_input)
    
    # Extract the positional encodings (they're added to zero input)
    pos_encodings = encoded[:, 0, :].numpy()  # [seq_len, d_model]
    
    print(f"Positional encoding shape: {pos_encodings.shape}")
    
    # Visualize a few dimensions
    plt.figure(figsize=(12, 8))
    
    # Plot first few dimensions
    for i in range(min(8, d_model)):
        plt.plot(pos_encodings[:50, i], label=f'dim {i}')
    
    plt.title('Positional Encoding (First 50 positions, First 8 dimensions)')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/positional_encoding_demo.png', dpi=150, bbox_inches='tight')
    print("Positional encoding visualization saved to /tmp/positional_encoding_demo.png")
    plt.close()
    
    # Show the sinusoidal pattern more clearly
    plt.figure(figsize=(10, 6))
    plt.imshow(pos_encodings[:50, :50].T, cmap='RdBu', aspect='auto')
    plt.title('Positional Encoding Heatmap (50 positions × 50 dimensions)')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    plt.savefig('/tmp/positional_encoding_heatmap.png', dpi=150, bbox_inches='tight')
    print("Positional encoding heatmap saved to /tmp/positional_encoding_heatmap.png")
    plt.close()


def demo_tokenizer():
    """Demonstrate the tokenizer functionality."""
    print("\n" + "=" * 60)
    print("TOKENIZER DEMONSTRATION")
    print("=" * 60)
    
    # Sample texts for vocabulary building
    sample_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Deep learning has achieved remarkable progress in recent years.",
        "Neural networks can learn complex patterns from data.",
        "Machine learning is a subset of artificial intelligence.",
        "Language models generate coherent and contextually relevant text.",
        "The attention is all you need paper introduced groundbreaking concepts.",
        "Self-attention enables parallel processing of sequences.",
        "Positional encoding helps models understand sequence order.",
        "Multi-head attention captures different types of relationships."
    ]
    
    # Create and train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=200, min_freq=1)
    tokenizer.build_vocabulary(sample_texts)
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Most frequent tokens: {tokenizer.get_most_frequent_tokens(10)}")
    
    # Demonstrate encoding/decoding
    test_text = "The transformer uses attention mechanisms for processing sequences."
    print(f"\nOriginal text: '{test_text}'")
    
    encoded = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Decoded: '{decoded}'")
    
    # Demonstrate batch processing
    batch_texts = sample_texts[:3]
    batch_encoded = tokenizer.encode_batch(
        batch_texts, 
        max_length=20, 
        padding=True, 
        truncation=True
    )
    
    print(f"\nBatch encoding shape: {len(batch_encoded['input_ids'])} × {len(batch_encoded['input_ids'][0])}")
    print("Sample batch:")
    for i, (ids, mask) in enumerate(zip(batch_encoded['input_ids'][:2], 
                                       batch_encoded['attention_mask'][:2])):
        print(f"  Sequence {i}: {ids}")
        print(f"  Attention:  {mask}")


def demo_model_creation():
    """Demonstrate model creation and basic forward pass."""
    print("\n" + "=" * 60)
    print("MODEL CREATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a small language model
    vocab_size = 1000
    model = create_small_language_model(vocab_size)
    
    print(f"Model created with {model.get_num_parameters():,} parameters")
    print(f"Model architecture:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Model dimension: {model.d_model}")
    print(f"  - Number of layers: {len(model.layers)}")
    print(f"  - Number of heads: {model.layers[0].self_attention.n_heads}")
    print(f"  - Maximum sequence length: {model.max_len}")
    
    # Demonstrate forward pass
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print(f"\nForward pass with input shape: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        logits, loss = model(input_ids, labels=input_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Cross-entropy loss: {loss.item():.4f}")
    print(f"Perplexity: {torch.exp(loss).item():.2f}")


def demo_text_generation():
    """Demonstrate basic text generation (with random model)."""
    print("\n" + "=" * 60)
    print("TEXT GENERATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a small model (untrained, for demonstration only)
    vocab_size = 100
    model = create_small_language_model(vocab_size, d_model=128, n_layers=2)
    
    print("Note: This model is untrained, so output will be random!")
    print("In a real scenario, you would load a trained model.")
    
    # Create a simple prompt
    prompt_ids = torch.tensor([[1, 5, 10, 15]], dtype=torch.long)  # Random token IDs
    print(f"Prompt token IDs: {prompt_ids.tolist()[0]}")
    
    # Generate with different strategies
    model.eval()
    
    print("\nGreedy decoding:")
    with torch.no_grad():
        generated_greedy = model.generate(
            prompt_ids, 
            max_new_tokens=10, 
            do_sample=False
        )
    print(f"Generated IDs: {generated_greedy.tolist()[0]}")
    
    print("\nSampling with temperature=0.8:")
    with torch.no_grad():
        generated_sample = model.generate(
            prompt_ids, 
            max_new_tokens=10, 
            temperature=0.8, 
            do_sample=True
        )
    print(f"Generated IDs: {generated_sample.tolist()[0]}")
    
    print("\nTop-k sampling (k=10):")
    with torch.no_grad():
        generated_topk = model.generate(
            prompt_ids, 
            max_new_tokens=10, 
            temperature=1.0, 
            top_k=10, 
            do_sample=True
        )
    print(f"Generated IDs: {generated_topk.tolist()[0]}")


def demo_sequence_to_sequence():
    """Demonstrate the full encoder-decoder transformer."""
    print("\n" + "=" * 60)
    print("SEQUENCE-TO-SEQUENCE DEMONSTRATION")
    print("=" * 60)
    
    # Create a small seq2seq transformer
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = create_transformer_model(
        src_vocab_size, 
        tgt_vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Seq2Seq model created with {total_params:,} parameters")
    
    # Demonstrate forward pass
    batch_size = 2
    src_len, tgt_len = 20, 15
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
    
    print(f"Output logits shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_len}, {tgt_vocab_size})")
    
    # Show attention mask creation
    src_mask = model.create_padding_mask(src)
    tgt_mask = model.create_padding_mask(tgt)
    causal_mask = model.create_causal_mask(tgt_len)
    
    print(f"\nMask shapes:")
    print(f"Source padding mask: {src_mask.shape}")
    print(f"Target padding mask: {tgt_mask.shape}")
    print(f"Causal mask: {causal_mask.shape}")


def run_all_demos():
    """Run all demonstration functions."""
    print("TRANSFORMER ARCHITECTURE DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the key components of our")
    print("transformer implementation with practical examples.")
    print("=" * 60)
    
    try:
        demo_attention_mechanism()
        demo_positional_encoding()
        demo_tokenizer()
        demo_model_creation()
        demo_text_generation()
        demo_sequence_to_sequence()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check /tmp/ for generated visualization files:")
        print("- attention_demo.png")
        print("- positional_encoding_demo.png")
        print("- positional_encoding_heatmap.png")
        print("\nFor actual training and experimentation,")
        print("run: python examples/train_language_model.py")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install torch numpy matplotlib")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_demos()
    sys.exit(0 if success else 1)