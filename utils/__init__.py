"""Utility modules for the Federated Summarization Platform."""

# Import only logger by default (torch-free)
from .logger import setup_logger

# Lazy imports for modules requiring torch (only for clients)
def __getattr__(name):
    if name == "CheckpointManager":
        from .checkpoint_manager import CheckpointManager
        return CheckpointManager
    elif name == "KafkaOffsetManager":
        from .kafka_offset_manager import KafkaOffsetManager
        return KafkaOffsetManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["CheckpointManager", "KafkaOffsetManager", "setup_logger"]
