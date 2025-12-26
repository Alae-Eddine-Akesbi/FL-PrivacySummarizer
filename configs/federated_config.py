"""
Configuration for Federated Learning and FedProx.

This module defines the configuration for federated learning parameters,
including FedProx proximal term, LoRA adapters, and training hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FederatedConfig:
    """
    Configuration class for Federated Learning with FedProx and LoRA.
    
    Attributes:
        num_rounds: Number of federated learning rounds
        steps_per_round: Training steps per round (for balanced computation)
        min_fit_clients: Minimum clients required for training round
        min_evaluate_clients: Minimum clients required for evaluation
        min_available_clients: Minimum clients that must be available
        fedprox_mu: Proximal term coefficient for FedProx (µ)
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        lora_r: LoRA rank (lower = fewer parameters)
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Which modules to apply LoRA to
        batch_size: Training batch size per client
        gradient_accumulation_steps: Steps to accumulate gradients
    """
    
    # Federated Learning Parameters
    num_rounds: int = field(
        default_factory=lambda: int(os.getenv("NUM_ROUNDS", "10"))
    )
    
    steps_per_round: int = field(
        default_factory=lambda: int(os.getenv("STEPS_PER_ROUND", "50"))
    )
    
    min_fit_clients: int = field(
        default_factory=lambda: int(os.getenv("MIN_FIT_CLIENTS", "3"))
    )
    
    min_evaluate_clients: int = field(
        default_factory=lambda: int(os.getenv("MIN_EVALUATE_CLIENTS", "3"))
    )
    
    min_available_clients: int = field(
        default_factory=lambda: int(os.getenv("MIN_AVAILABLE_CLIENTS", "3"))
    )
    
    # FedProx Parameters
    fedprox_mu: float = field(
        default_factory=lambda: float(os.getenv("FEDPROX_MU", "0.01"))
    )
    
    # Optimization Parameters
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("LEARNING_RATE", "2e-5"))
    )
    
    weight_decay: float = field(
        default_factory=lambda: float(os.getenv("WEIGHT_DECAY", "0.01"))
    )
    
    # LoRA Parameters
    lora_r: int = field(
        default_factory=lambda: int(os.getenv("LORA_R", "16"))
    )
    
    lora_alpha: int = field(
        default_factory=lambda: int(os.getenv("LORA_ALPHA", "32"))
    )
    
    lora_dropout: float = field(
        default_factory=lambda: float(os.getenv("LORA_DROPOUT", "0.05"))
    )
    
    lora_target_modules: List[str] = field(
        default_factory=lambda: os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")
    )
    
    # Training Parameters
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "1"))
    )
    
    gradient_accumulation_steps: int = field(
        default_factory=lambda: int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
    )
    
    # Checkpoint Parameters
    checkpoint_dir: str = field(
        default_factory=lambda: os.getenv("CHECKPOINT_DIR", "/app/checkpoints")
    )
    
    save_every_n_rounds: int = field(
        default_factory=lambda: int(os.getenv("SAVE_EVERY_N_ROUNDS", "1"))
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.fedprox_mu < 0:
            raise ValueError("FedProx mu must be non-negative")
        
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        
        if self.min_fit_clients > self.min_available_clients:
            raise ValueError("min_fit_clients cannot exceed min_available_clients")
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "num_rounds": self.num_rounds,
            "steps_per_round": self.steps_per_round,
            "min_fit_clients": self.min_fit_clients,
            "fedprox_mu": self.fedprox_mu,
            "learning_rate": self.learning_rate,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "batch_size": self.batch_size,
        }
    
    @classmethod
    def from_env(cls) -> "FederatedConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            FederatedConfig instance populated from environment
        """
        return cls()
    
    def get_effective_batch_size(self) -> int:
        """
        Calculate effective batch size with gradient accumulation.
        
        Returns:
            Effective batch size
        """
        return self.batch_size * self.gradient_accumulation_steps
