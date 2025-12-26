"""
LoRA Manager for PEFT.

Handles LoRA adapter configuration and management.
"""

import logging
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModelForSeq2SeqLM
from typing import Optional

from configs.federated_config import FederatedConfig


logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Manages LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    
    Handles:
    - LoRA configuration
    - Adapter injection
    - Parameter counting
    """
    
    def __init__(self, federated_config: Optional[FederatedConfig] = None):
        """
        Initialize the LoRA Manager.
        
        Args:
            federated_config: Federated configuration (uses env if None)
        """
        self.fed_config = federated_config or FederatedConfig.from_env()
        logger.info("LoRA Manager initialized")
    
    def create_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Returns:
            LoraConfig instance
        """
        lora_config = LoraConfig(
            r=self.fed_config.lora_r,
            lora_alpha=self.fed_config.lora_alpha,
            target_modules=self.fed_config.lora_target_modules,
            lora_dropout=self.fed_config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        
        logger.info(f"LoRA Config - r={self.fed_config.lora_r}, alpha={self.fed_config.lora_alpha}")
        return lora_config
    
    def inject_lora_adapters(
        self,
        model: AutoModelForSeq2SeqLM
    ) -> PeftModel:
        """
        Inject LoRA adapters into the model.
        
        Args:
            model: Base model
            
        Returns:
            PEFT model with LoRA adapters
        """
        lora_config = self.create_lora_config()
        
        logger.info("Injecting LoRA adapters...")
        peft_model = get_peft_model(model, lora_config)
        
        # Enable training mode and ensure gradients for LoRA parameters
        peft_model.train()
        
        # Explicitly enable requires_grad for LoRA parameters
        for name, param in peft_model.named_parameters():
            if 'lora_' in name or param.requires_grad:
                param.requires_grad = True
        
        # Log trainable parameters
        self._log_trainable_params(peft_model)
        
        return peft_model
    
    def _log_trainable_params(self, model: PeftModel) -> None:
        """
        Log trainable parameter information.
        
        Args:
            model: PEFT model
        """
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_params
        
        logger.info(f"Total parameters: {all_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {trainable_percent:.4f}%")
