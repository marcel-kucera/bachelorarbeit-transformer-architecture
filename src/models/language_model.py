"""
GPT-style Language Model Implementation

This module implements a decoder-only transformer language model
similar to GPT, designed for text generation tasks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .transformer import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding


class DecoderOnlyLayer(nn.Module):
    """
    Decoder-only transformer layer for language modeling.
    
    Similar to GPT architecture, this uses only masked self-attention
    without cross-attention.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize decoder-only layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of decoder-only layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Causal attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Masked self-attention with residual connection and layer norm
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class GPTLanguageModel(nn.Module):
    """
    GPT-style language model for text generation.
    
    This is a decoder-only transformer that can be used for
    autoregressive language modeling and text generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        Initialize GPT language model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: Padding token ID
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie input and output embeddings (common practice)
        self.output_projection.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        """Initialize parameters using scaled normal initialization."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if 'weight' in name and 'norm' not in name:
                    # Use scaled initialization for transformer weights
                    std = 0.02
                    if 'output_projection' in name:
                        std = 0.02 / math.sqrt(2 * len(self.layers))
                    nn.init.normal_(param, mean=0.0, std=std)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of language model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for loss computation [batch_size, seq_len]
            
        Returns:
            Tuple of (logits, loss)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match causal mask shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            combined_mask = causal_mask.unsqueeze(0) & attention_mask
        else:
            combined_mask = causal_mask.unsqueeze(0)  # [1, seq_len, seq_len]
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, combined_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = self.forward(generated)
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # Truncate if sequence gets too long
            if generated.size(1) >= self.max_len:
                break
        
        return generated
    
    def get_num_parameters(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params


def create_small_language_model(
    vocab_size: int,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 1024,
    max_len: int = 512,
    dropout: float = 0.1
) -> GPTLanguageModel:
    """
    Create a small language model for experimentation.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
        
    Returns:
        Small GPT language model
    """
    return GPTLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )


if __name__ == "__main__":
    # Example usage
    model = create_small_language_model(vocab_size=5000)
    
    # Example forward pass
    input_ids = torch.randint(1, 5000, (4, 64))  # batch_size=4, seq_len=64
    
    logits, loss = model(input_ids, labels=input_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item() if loss is not None else 'None'}")
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Example generation
    prompt = torch.randint(1, 100, (1, 10))  # Single prompt
    generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
    print(f"Generated shape: {generated.shape}")