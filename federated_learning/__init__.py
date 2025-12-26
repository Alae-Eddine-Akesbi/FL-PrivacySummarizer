"""Federated Learning modules with Flower."""

from .flower_client import FederatedSummarizationClient
from .flower_server import create_strategy, start_server
from .lora_manager import LoRAManager
from .fedprox_optimizer import FedProxOptimizer

__all__ = [
    "FederatedSummarizationClient",
    "create_strategy",
    "start_server",
    "LoRAManager",
    "FedProxOptimizer"
]
