"""
Configuration for the LED Summarization Model.

This module defines the configuration for the Longformer Encoder-Decoder (LED)
model used for document summarization with global attention mechanism.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """
    Configuration class for LED model and tokenization.
    
    Attributes:
        model_name: HuggingFace model identifier
        max_input_length: Maximum input sequence length (LED supports up to 16384)
        max_target_length: Maximum target/summary sequence length
        load_in_4bit: Enable 4-bit quantization for memory efficiency
        use_gradient_checkpointing: Enable gradient checkpointing to save VRAM
        global_attention_on_first_token: Apply global attention to <s> token
        attention_window: Local attention window size for LED
        cache_dir: Directory for model cache
    """
    
    # LARGE MODEL (for production - requires more VRAM and download time)
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "pszemraj/led-large-book-summary")
    )
    
    # BASE MODEL (for testing - faster download, less VRAM)
    # model_name: str = field(
    #     default_factory=lambda: os.getenv("MODEL_NAME", "pszemraj/led-base-book-summary")
    # )
    
    max_input_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_INPUT_LENGTH", "8192"))
    )
    
    max_target_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_TARGET_LENGTH", "512"))
    )
    
    load_in_4bit: bool = field(
        default_factory=lambda: os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    )
    
    use_gradient_checkpointing: bool = field(
        default_factory=lambda: os.getenv("USE_GRADIENT_CHECKPOINTING", "true").lower() == "true"
    )
    
    global_attention_on_first_token: bool = True
    
    attention_window: int = 512
    
    cache_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "/root/.cache")
    )
    
    # Quantization configuration
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_input_length > 16384:
            raise ValueError("LED model supports maximum 16384 tokens")
        
        if self.max_target_length > 1024:
            raise ValueError("Target length should not exceed 1024 tokens")
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model_name": self.model_name,
            "max_input_length": self.max_input_length,
            "max_target_length": self.max_target_length,
            "load_in_4bit": self.load_in_4bit,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "global_attention_on_first_token": self.global_attention_on_first_token,
            "attention_window": self.attention_window,
        }
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            ModelConfig instance populated from environment
        """
        return cls()
