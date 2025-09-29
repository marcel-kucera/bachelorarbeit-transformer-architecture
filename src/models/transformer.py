"""
Transformer Model Implementation

This module contains a complete implementation of the Transformer architecture
as described in "Attention Is All You Need" (Vaswani et al., 2017).

The implementation includes:
- Multi-Head Attention mechanism
- Position-wise Feed-Forward Networks
- Positional Encoding
- Layer Normalization and Residual Connections
- Complete Encoder and Decoder stacks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    This implements the scaled dot-product attention with multiple attention heads
    as described in the original Transformer paper.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_k]
            key: Key tensor [batch_size, seq_len, d_k]
            value: Value tensor [batch_size, seq_len, d_k]
            mask: Optional mask tensor
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This implements the feed-forward sublayer used in each transformer layer.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize Position-wise Feed-Forward Network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Position-wise Feed-Forward Network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sinusoidal functions.
    
    This adds positional information to token embeddings to give the model
    information about the position of tokens in the sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize Positional Encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Embeddings with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer of the Transformer.
    
    Consists of multi-head self-attention and position-wise feed-forward network,
    each with residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize Encoder Layer.
        
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
        Forward pass of Encoder Layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer of the Transformer.
    
    Consists of masked multi-head self-attention, cross-attention with encoder,
    and position-wise feed-forward network, each with residual connections
    and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize Decoder Layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Decoder Layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal mask)
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Masked self-attention with residual connection and layer norm
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))
        
        # Cross-attention with encoder with residual connection and layer norm
        cross_attention_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple encoder layers.
    """
    
    def __init__(self, layer: EncoderLayer, n_layers: int):
        """
        Initialize Transformer Encoder.
        
        Args:
            layer: Encoder layer prototype
            n_layers: Number of encoder layers
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(layer.self_attention.d_model, 
                        layer.self_attention.n_heads,
                        layer.feed_forward.linear1.out_features,
                        layer.dropout.p)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(layer.self_attention.d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Encoded output [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder consisting of multiple decoder layers.
    """
    
    def __init__(self, layer: DecoderLayer, n_layers: int):
        """
        Initialize Transformer Decoder.
        
        Args:
            layer: Decoder layer prototype
            n_layers: Number of decoder layers
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(layer.self_attention.d_model,
                        layer.self_attention.n_heads,
                        layer.feed_forward.linear1.out_features,
                        layer.dropout.p)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(layer.self_attention.d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            Decoded output [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    This is the full implementation of the Transformer architecture
    including embeddings, positional encoding, encoder, and decoder.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize complete Transformer model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder/decoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder
        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        
        self.encoder = TransformerEncoder(encoder_layer, n_layers)
        self.decoder = TransformerDecoder(decoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        """Initialize parameters using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            seq: Sequence tensor [batch_size, seq_len]
            pad_idx: Padding token index
            
        Returns:
            Padding mask [batch_size, 1, 1, seq_len]
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create causal (look-ahead) mask for decoder self-attention.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0
    ) -> torch.Tensor:
        """
        Forward pass of complete Transformer.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_pad_idx: Source padding token index
            tgt_pad_idx: Target padding token index
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Create masks
        src_mask = self.create_padding_mask(src, src_pad_idx)
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
        tgt_mask = tgt_mask & causal_mask
        
        # Embed and add positional encoding
        src_embedded = self.dropout(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_embedded = self.dropout(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Encode and decode
        encoder_output = self.encoder(src_embedded, src_mask)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output


def create_transformer_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    max_len: int = 5000,
    dropout: float = 0.1
) -> Transformer:
    """
    Factory function to create a Transformer model.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder/decoder layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        
    Returns:
        Initialized Transformer model
    """
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )


if __name__ == "__main__":
    # Example usage
    model = create_transformer_model(
        src_vocab_size=10000, 
        tgt_vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=6
    )
    
    # Example forward pass
    src = torch.randint(1, 1000, (32, 50))  # batch_size=32, src_len=50
    tgt = torch.randint(1, 1000, (32, 40))  # batch_size=32, tgt_len=40
    
    output = model(src, tgt)
    print(f"Model output shape: {output.shape}")  # [32, 40, 10000]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")