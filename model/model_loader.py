"""
Model Loader with Quantization Support.

This module handles loading the LED model with 4-bit quantization
using bitsandbytes for memory efficiency.
"""

import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from typing import Tuple, Optional

from configs.model_config import ModelConfig


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads LED model with optional 4-bit quantization.
    
    Handles:
    - Model loading from HuggingFace
    - 4-bit quantization configuration
    - Gradient checkpointing setup
    - Device placement
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        Initialize the Model Loader.
        
        Args:
            model_config: Model configuration (uses env if None)
        """
        self.model_config = model_config or ModelConfig.from_env()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(
        self
    ) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        Load the LED model and tokenizer with quantization if enabled.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading model: {self.model_config.model_name}")
        
        try:
            # Load tokenizer
            tokenizer = self._load_tokenizer()
            
            # Load model with or without quantization
            if self.model_config.load_in_4bit and self.device == "cuda":
                model = self._load_quantized_model()
            else:
                model = self._load_full_precision_model()
            
            # Enable gradient checkpointing if requested
            if self.model_config.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()
                # Configure use_reentrant=False for better compatibility with LoRA
                if hasattr(model.config, 'use_cache'):
                    model.config.use_cache = False
                logger.info("Gradient checkpointing enabled with use_cache=False")
            
            # Log model info
            self._log_model_info(model)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer.
        
        Returns:
            Loaded tokenizer
        """
        logger.info("Loading tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            trust_remote_code=True
        )
        
        logger.info(f"Tokenizer loaded: {len(tokenizer)} tokens")
        return tokenizer
    
    def _load_quantized_model(self) -> AutoModelForSeq2SeqLM:
        """
        Load model with 4-bit quantization.
        
        Returns:
            Quantized model
        """
        logger.info("Loading model with 4-bit quantization...")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.model_config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.model_config.bnb_4bit_use_double_quant,
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=bnb_config,
            cache_dir=self.model_config.cache_dir,
            trust_remote_code=True,
        )
        
        logger.info("Model loaded with 4-bit quantization")
        return model
    
    def _load_full_precision_model(self) -> AutoModelForSeq2SeqLM:
        """
        Load model in full precision.
        
        Returns:
            Full precision model
        """
        logger.info("Loading model in full precision...")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            trust_remote_code=True,
        )
        
        # Move to device
        model = model.to(self.device)
        
        logger.info("Model loaded in full precision")
        return model
    
    def _log_model_info(self, model: AutoModelForSeq2SeqLM) -> None:
        """
        Log information about the loaded model.
        
        Args:
            model: The loaded model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # Memory usage
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    @staticmethod
    def prepare_model_for_training(model: AutoModelForSeq2SeqLM) -> AutoModelForSeq2SeqLM:
        """
        Prepare model for training by freezing/unfreezing layers.
        
        Args:
            model: Model to prepare
            
        Returns:
            Prepared model
        """
        # For quantized models, we need to use PEFT
        # The base model will be frozen and adapters will be trainable
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info("Base model frozen for adapter training")
        return model
