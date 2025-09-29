"""
Simple tokenizer implementation for the language model.

This module provides a basic tokenizer that can be used for text preprocessing
and vocabulary management.
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from collections import Counter
import pickle


class SimpleTokenizer:
    """
    A simple tokenizer for text processing.
    
    This tokenizer performs basic text preprocessing and maintains
    a vocabulary of tokens with special tokens for unknown words,
    padding, beginning/end of sequence, etc.
    """
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a token to be included in vocabulary
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        
        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_freq: Dict[str, int] = {}
        
        # Initialize with special tokens
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.special_tokens['<pad>']
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.special_tokens['<unk>']
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.special_tokens['<bos>']
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.special_tokens['<eos>']
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Add spaces around punctuation
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        
        # Remove special characters (keep letters, numbers, basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9.!?,:;\s]', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into list of tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = self.preprocess_text(text)
        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of input texts
        """
        print("Building vocabulary...")
        
        # Count token frequencies
        token_counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counter.update(tokens)
        
        # Store token frequencies
        self.token_freq = dict(token_counter)
        
        # Filter tokens by minimum frequency and sort by frequency
        filtered_tokens = [
            token for token, freq in token_counter.items() 
            if freq >= self.min_freq
        ]
        filtered_tokens.sort(key=lambda x: token_counter[x], reverse=True)
        
        # Limit vocabulary size (reserve space for special tokens)
        max_regular_tokens = self.vocab_size - len(self.special_tokens)
        filtered_tokens = filtered_tokens[:max_regular_tokens]
        
        # Add tokens to vocabulary
        for token in filtered_tokens:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
        
        print(f"Vocabulary built with {len(self.token_to_id)} tokens")
        print(f"Most frequent tokens: {filtered_tokens[:10]}")
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def encode_batch(
        self, 
        texts: List[str], 
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = False
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded_texts = []
        attention_masks = []
        
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=add_special_tokens)
            
            # Truncate if necessary
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(token_ids)
            
            # Pad if necessary
            if padding and max_length:
                padding_length = max_length - len(token_ids)
                if padding_length > 0:
                    token_ids.extend([self.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)
            
            encoded_texts.append(token_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': encoded_texts,
            'attention_mask': attention_masks
        }
    
    def save(self, filepath: str) -> None:
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save the tokenizer
        """
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'token_freq': self.token_freq,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SimpleTokenizer':
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to load the tokenizer from
            
        Returns:
            Loaded tokenizer instance
        """
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=tokenizer_data['vocab_size'],
            min_freq=tokenizer_data['min_freq']
        )
        
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.token_freq = tokenizer_data['token_freq']
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        
        print(f"Tokenizer loaded from {filepath}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)
    
    def get_token_frequency(self, token: str) -> int:
        """Get frequency of a specific token."""
        return self.token_freq.get(token, 0)
    
    def get_most_frequent_tokens(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequent tokens.
        
        Args:
            n: Number of tokens to return
            
        Returns:
            List of (token, frequency) tuples
        """
        sorted_tokens = sorted(
            self.token_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_tokens[:n]


def create_simple_dataset(texts: List[str], tokenizer: SimpleTokenizer, 
                         max_length: int = 512) -> Dict[str, List]:
    """
    Create a simple dataset from texts.
    
    Args:
        texts: List of input texts
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized data
    """
    print(f"Creating dataset from {len(texts)} texts...")
    
    # Encode all texts
    encoded = tokenizer.encode_batch(
        texts, 
        max_length=max_length,
        padding=True,
        truncation=True,
        add_special_tokens=True
    )
    
    print(f"Dataset created with {len(encoded['input_ids'])} sequences")
    return encoded


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Hello, this is a sample text for testing the tokenizer.",
        "The tokenizer should handle various punctuation marks and normalize text.",
        "We can build vocabulary from multiple texts and encode them efficiently.",
        "This is another example sentence with different words and structure."
    ]
    
    # Create and train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000, min_freq=1)
    tokenizer.build_vocabulary(sample_texts)
    
    # Test encoding/decoding
    text = "Hello, this is a test sentence."
    encoded = tokenizer.encode(text, add_special_tokens=True)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test batch encoding
    batch_encoded = tokenizer.encode_batch(sample_texts, max_length=20, padding=True)
    print(f"Batch encoded shape: {len(batch_encoded['input_ids'])} x {len(batch_encoded['input_ids'][0])}")
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Most frequent tokens: {tokenizer.get_most_frequent_tokens(5)}")