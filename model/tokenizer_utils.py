"""
Tokenizer utilities for LED model.

Handles tokenization with global attention mask for LED.
"""

import logging
import torch
from typing import Dict, List
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class TokenizerUtils:
    """Utilities for tokenizing documents with global attention."""
    
    @staticmethod
    def tokenize_with_global_attention(
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
        apply_global_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts and create global attention mask.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            apply_global_attention: Whether to apply global attention
            
        Returns:
            Dictionary with input_ids, attention_mask, global_attention_mask
        """
        # Tokenize
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create global attention mask if requested
        if apply_global_attention:
            # Apply global attention to first token (<s>)
            global_attention_mask = torch.zeros_like(encoded["input_ids"])
            global_attention_mask[:, 0] = 1
            encoded["global_attention_mask"] = global_attention_mask
        
        return encoded
    
    @staticmethod
    def decode_summary(
        token_ids: torch.Tensor,
        tokenizer: AutoTokenizer,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode generated summary token IDs.
        
        Args:
            token_ids: Generated token IDs
            tokenizer: Tokenizer instance
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded summaries
        """
        return tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
