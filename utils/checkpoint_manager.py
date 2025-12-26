"""
Checkpoint Manager for LoRA Adapters.

Handles saving and loading of LoRA adapter checkpoints
with support for recovery from failures.
"""

import logging
import os
import json
import torch
from typing import Optional, Dict
from pathlib import Path
from peft import PeftModel


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for LoRA adapters.
    
    Handles:
    - Saving LoRA adapter weights
    - Loading checkpoints for recovery
    - Metadata management
    - Cleanup of old checkpoints
    """
    
    def __init__(self, checkpoint_dir: str = "/app/checkpoints"):
        """
        Initialize the Checkpoint Manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: PeftModel,
        round_num: int,
        client_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a checkpoint for LoRA adapters.
        
        Args:
            model: PEFT model with LoRA adapters
            round_num: Current training round
            client_id: Client identifier
            metadata: Optional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"{client_id}_round_{round_num}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        logger.info(f"Saving checkpoint: {checkpoint_name}")
        
        try:
            # Save LoRA adapters
            model.save_pretrained(checkpoint_path)
            
            # Save metadata
            if metadata:
                metadata_path = checkpoint_path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Checkpoint saved successfully: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        model: PeftModel,
        round_num: int,
        client_id: str
    ) -> Optional[Dict]:
        """
        Load a checkpoint for recovery.
        
        Args:
            model: PEFT model to load adapters into
            round_num: Round number to load
            client_id: Client identifier
            
        Returns:
            Metadata if available, None otherwise
        """
        checkpoint_name = f"{client_id}_round_{round_num}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        logger.info(f"Loading checkpoint: {checkpoint_name}")
        
        try:
            # Load LoRA adapters
            model.load_adapter(checkpoint_path, adapter_name="default")
            
            # Load metadata
            metadata_path = checkpoint_path / "metadata.json"
            metadata = None
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            
            logger.info(f"Checkpoint loaded successfully: {checkpoint_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_latest_checkpoint(self, client_id: str) -> Optional[int]:
        """
        Get the latest checkpoint round number for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Latest round number or None
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{client_id}_round_*"))
        
        if not checkpoints:
            return None
        
        # Extract round numbers
        round_numbers = []
        for cp in checkpoints:
            try:
                round_num = int(cp.name.split("_")[-1])
                round_numbers.append(round_num)
            except ValueError:
                continue
        
        return max(round_numbers) if round_numbers else None
    
    def cleanup_old_checkpoints(
        self,
        client_id: str,
        keep_last_n: int = 3
    ) -> None:
        """
        Remove old checkpoints, keeping only the last N.
        
        Args:
            client_id: Client identifier
            keep_last_n: Number of checkpoints to keep
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{client_id}_round_*"))
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Sort by round number
        checkpoints.sort(key=lambda x: int(x.name.split("_")[-1]))
        
        # Remove old checkpoints
        for cp in checkpoints[:-keep_last_n]:
            logger.info(f"Removing old checkpoint: {cp}")
            import shutil
            shutil.rmtree(cp)
