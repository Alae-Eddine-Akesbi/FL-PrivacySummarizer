"""
Configuration for Kafka streaming infrastructure.

This module defines the configuration for Kafka topics, consumer groups,
and streaming parameters for both training and inference phases.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class KafkaConfig:
    """
    Configuration class for Kafka infrastructure.
    
    Attributes:
        bootstrap_servers: Kafka broker addresses
        health_topic: Topic for health department documents
        finance_topic: Topic for finance department documents
        legal_topic: Topic for legal department documents
        inference_topic: Topic for inference requests
        consumer_group_id: Consumer group identifier
        auto_offset_reset: Where to start consuming ('earliest' or 'latest')
        enable_auto_commit: Enable automatic offset commit
        max_poll_records: Maximum records per poll
        session_timeout_ms: Session timeout in milliseconds
        heartbeat_interval_ms: Heartbeat interval in milliseconds
        max_poll_interval_ms: Maximum time between polls
    """
    
    # Kafka Connection
    bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    )
    
    # Topic Configuration
    health_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_HEALTH_TOPIC", "health-documents")
    )
    
    finance_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_FINANCE_TOPIC", "finance-documents")
    )
    
    legal_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_LEGAL_TOPIC", "legal-documents")
    )
    
    inference_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_INFERENCE_TOPIC", "inference-requests")
    )
    
    # Consumer Configuration
    consumer_group_id: str = field(
        default_factory=lambda: os.getenv("KAFKA_CONSUMER_GROUP", "federated-clients")
    )
    
    auto_offset_reset: str = field(
        default_factory=lambda: os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
    )
    
    enable_auto_commit: bool = field(
        default_factory=lambda: os.getenv("KAFKA_ENABLE_AUTO_COMMIT", "false").lower() == "true"
    )
    
    # Performance Configuration
    max_poll_records: int = field(
        default_factory=lambda: int(os.getenv("KAFKA_MAX_POLL_RECORDS", "100"))
    )
    
    session_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000"))
    )
    
    heartbeat_interval_ms: int = field(
        default_factory=lambda: int(os.getenv("KAFKA_HEARTBEAT_INTERVAL_MS", "10000"))
    )
    
    max_poll_interval_ms: int = field(
        default_factory=lambda: int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "300000"))
    )
    
    # Producer Configuration
    acks: str = field(
        default_factory=lambda: os.getenv("KAFKA_ACKS", "all")
    )
    
    retries: int = field(
        default_factory=lambda: int(os.getenv("KAFKA_RETRIES", "3"))
    )
    
    compression_type: str = field(
        default_factory=lambda: os.getenv("KAFKA_COMPRESSION_TYPE", "gzip")
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.auto_offset_reset not in ["earliest", "latest", "none"]:
            raise ValueError("auto_offset_reset must be 'earliest', 'latest', or 'none'")
        
        if self.acks not in ["0", "1", "all"]:
            raise ValueError("acks must be '0', '1', or 'all'")
    
    def get_topic_by_client(self, client_id: str) -> str:
        """
        Get the appropriate Kafka topic for a given client.
        
        Args:
            client_id: Identifier for the client (health, finance, legal)
            
        Returns:
            Topic name for the client
            
        Raises:
            ValueError: If client_id is not recognized
        """
        topic_map = {
            "health": self.health_topic,
            "finance": self.finance_topic,
            "legal": self.legal_topic,
        }
        
        if client_id not in topic_map:
            raise ValueError(f"Unknown client_id: {client_id}. Must be one of {list(topic_map.keys())}")
        
        return topic_map[client_id]
    
    def get_all_topics(self) -> List[str]:
        """
        Get list of all configured topics.
        
        Returns:
            List of topic names
        """
        return [
            self.health_topic,
            self.finance_topic,
            self.legal_topic,
            self.inference_topic,
        ]
    
    def get_producer_config(self) -> Dict[str, any]:
        """
        Get Kafka producer configuration.
        
        Returns:
            Dictionary of producer configuration
        """
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "acks": self.acks,
            "retries": self.retries,
            "compression_type": self.compression_type,
        }
    
    def get_consumer_config(self, group_id: Optional[str] = None) -> Dict[str, any]:
        """
        Get Kafka consumer configuration.
        
        Args:
            group_id: Optional custom consumer group ID
            
        Returns:
            Dictionary of consumer configuration
        """
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": group_id or self.consumer_group_id,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "max_poll_records": self.max_poll_records,
            "session_timeout_ms": self.session_timeout_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "max_poll_interval_ms": self.max_poll_interval_ms,
        }
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "topics": {
                "health": self.health_topic,
                "finance": self.finance_topic,
                "legal": self.legal_topic,
                "inference": self.inference_topic,
            },
            "consumer_group": self.consumer_group_id,
        }
    
    @classmethod
    def from_env(cls) -> "KafkaConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            KafkaConfig instance populated from environment
        """
        return cls()
