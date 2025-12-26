"""Data ingestion modules for Kafka-based streaming."""

from .producer import KafkaDocumentProducer
from .data_loader import DatasetLoader
from .topic_manager import TopicManager

__all__ = ["KafkaDocumentProducer", "DatasetLoader", "TopicManager"]
