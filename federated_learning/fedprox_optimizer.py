"""
FedProx Optimizer.

Implements the FedProx algorithm with proximal term for federated optimization.
"""

import logging
import torch
from torch.optim import AdamW
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class FedProxOptimizer:
    """
    FedProx optimizer with proximal term.
    
    FedProx adds a proximal term to the loss function:
    L_FedProx = L_local + (mu/2) * ||w - w_global||^2
    
    Where:
    - L_local is the local loss
    - mu is the proximal term coefficient
    - w is current weights
    - w_global is global model weights
    """
    
    def __init__(
        self,
        parameters,
        lr: float = 2e-5,
        mu: float = 0.01,
        weight_decay: float = 0.01
    ):
        """
        Initialize FedProx optimizer.
        
        Args:
            parameters: Model parameters to optimize
            lr: Learning rate
            mu: Proximal term coefficient
            weight_decay: Weight decay for regularization
        """
        self.mu = mu
        self.optimizer = AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Store global model parameters (will be set before training)
        self.global_params: Optional[Dict[str, torch.Tensor]] = None
        
        logger.info(f"FedProx Optimizer initialized - mu={mu}, lr={lr}")
    
    def set_global_params(self, global_params: Dict[str, torch.Tensor]) -> None:
        """
        Set global model parameters for proximal term.
        
        Args:
            global_params: Dictionary of global parameter tensors
        """
        self.global_params = {
            name: param.clone().detach()
            for name, param in global_params.items()
        }
        logger.debug("Global parameters set for FedProx")
    
    def compute_proximal_loss(
        self,
        model_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the proximal term: (mu/2) * ||w - w_global||^2
        
        Args:
            model_params: Current model parameters
            
        Returns:
            Proximal loss term
        """
        if self.global_params is None:
            return torch.tensor(0.0)
        
        proximal_term = 0.0
        
        for name, param in model_params.items():
            if name in self.global_params and param.requires_grad:
                proximal_term += torch.sum((param - self.global_params[name]) ** 2)
        
        return (self.mu / 2) * proximal_term
    
    def step(self, model, loss: torch.Tensor) -> torch.Tensor:
        """
        Perform optimization step with proximal term.
        
        Args:
            model: Model being trained
            loss: Local loss
            
        Returns:
            Total loss (local + proximal)
        """
        # Compute proximal term
        model_params = dict(model.named_parameters())
        proximal_loss = self.compute_proximal_loss(model_params)
        
        # Total loss
        total_loss = loss + proximal_loss
        
        # Backward and step
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self) -> Dict:
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)
