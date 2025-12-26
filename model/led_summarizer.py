"""
LED Summarizer Wrapper.

High-level interface for LED-based summarization.
"""

import logging
import torch
from typing import List, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from configs.model_config import ModelConfig
from model.tokenizer_utils import TokenizerUtils


logger = logging.getLogger(__name__)


class LEDSummarizer:
    """
    High-level wrapper for LED summarization model.
    
    Provides easy-to-use interface for:
    - Document summarization
    - Batch processing
    - Global attention handling
    """
    
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize the LED Summarizer.
        
        Args:
            model: Loaded LED model
            tokenizer: Loaded tokenizer
            model_config: Model configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config or ModelConfig.from_env()
        self.device = next(model.parameters()).device
    
    def summarize(
        self,
        texts: List[str],
        max_new_tokens: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 2.0
    ) -> List[str]:
        """
        Generate summaries for input texts.
        
        Args:
            texts: List of documents to summarize
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            
        Returns:
            List of generated summaries
        """
        max_new_tokens = max_new_tokens or self.model_config.max_target_length
        
        # Tokenize with global attention
        inputs = TokenizerUtils.tokenize_with_global_attention(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.model_config.max_input_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )
        
        # Decode
        summaries = TokenizerUtils.decode_summary(generated_ids, self.tokenizer)
        
        return summaries
