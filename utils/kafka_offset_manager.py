"""
Kafka Offset Manager.

Manages Kafka consumer offsets for reliable message processing
and recovery from failures.
"""

import logging
import json
from typing import Dict, Optional
from pathlib import Path
from kafka import TopicPartition


logger = logging.getLogger(__name__)


class KafkaOffsetManager:
    """
    Manages Kafka consumer offsets for recovery.
    
    Handles:
    - Saving current offsets
    - Loading offsets for recovery
    - Offset tracking per partition
    """
    
    def __init__(self, offset_dir: str = "/app/checkpoints/offsets"):
        """
        Initialize the Offset Manager.
        
        Args:
            offset_dir: Directory for storing offset information
        """
        self.offset_dir = Path(offset_dir)
        self.offset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Offset directory: {self.offset_dir}")
    
    def save_offsets(
        self,
        client_id: str,
        topic: str,
        offsets: Dict[int, int]
    ) -> None:
        """
        Save current offsets for a topic.
        
        Args:
            client_id: Client identifier
            topic: Kafka topic
            offsets: Dictionary mapping partition to offset
        """
        offset_file = self.offset_dir / f"{client_id}_{topic}.json"
        
        try:
            with open(offset_file, "w") as f:
                json.dump(offsets, f, indent=2)
            
            logger.debug(f"Offsets saved for {client_id}/{topic}: {offsets}")
            
        except Exception as e:
            logger.error(f"Failed to save offsets: {e}")
    
    def load_offsets(
        self,
        client_id: str,
        topic: str
    ) -> Optional[Dict[int, int]]:
        """
        Load saved offsets for recovery.
        
        Args:
            client_id: Client identifier
            topic: Kafka topic
            
        Returns:
            Dictionary mapping partition to offset, or None
        """
        offset_file = self.offset_dir / f"{client_id}_{topic}.json"
        
        if not offset_file.exists():
            logger.info(f"No saved offsets found for {client_id}/{topic}")
            return None
        
        try:
            with open(offset_file, "r") as f:
                offsets = json.load(f)
            
            # Convert string keys to int
            offsets = {int(k): v for k, v in offsets.items()}
            
            logger.info(f"Loaded offsets for {client_id}/{topic}: {offsets}")
            return offsets
            
        except Exception as e:
            logger.error(f"Failed to load offsets: {e}")
            return None
    
    def commit_offset(
        self,
        client_id: str,
        topic: str,
        partition: int,
        offset: int
    ) -> None:
        """
        Commit a single offset.
        
        Args:
            client_id: Client identifier
            topic: Kafka topic
            partition: Partition number
            offset: Offset to commit
        """
        # Load existing offsets
        offsets = self.load_offsets(client_id, topic) or {}
        
        # Update offset
        offsets[partition] = offset
        
        # Save updated offsets
        self.save_offsets(client_id, topic, offsets)
