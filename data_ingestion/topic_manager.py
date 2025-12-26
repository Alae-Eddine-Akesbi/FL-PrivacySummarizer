"""
Kafka Topic Manager.

This module handles the creation and management of Kafka topics
for the federated learning platform.
"""

import logging
import time
from typing import List
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaError

from configs.kafka_config import KafkaConfig


logger = logging.getLogger(__name__)


class TopicManager:
    """
    Manages Kafka topics for the federated learning system.
    
    Handles:
    - Topic creation with appropriate configuration
    - Topic validation
    - Topic cleanup (if needed)
    """
    
    def __init__(self, kafka_config: KafkaConfig):
        """
        Initialize the Topic Manager.
        
        Args:
            kafka_config: Kafka configuration instance
        """
        self.kafka_config = kafka_config
        self.admin_client = None
    
    def _get_admin_client(self) -> KafkaAdminClient:
        """
        Get or create Kafka Admin Client.
        
        Returns:
            KafkaAdminClient instance
        """
        if self.admin_client is None:
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                client_id="topic_manager"
            )
        return self.admin_client
    
    def create_topics(
        self,
        num_partitions: int = 3,
        replication_factor: int = 1
    ) -> None:
        """
        Create all required Kafka topics.
        
        Args:
            num_partitions: Number of partitions per topic
            replication_factor: Replication factor for topics
        """
        topics = [
            self.kafka_config.health_topic,
            self.kafka_config.finance_topic,
            self.kafka_config.legal_topic,
            self.kafka_config.inference_topic,
        ]
        
        logger.info(f"Creating {len(topics)} Kafka topics...")
        
        new_topics = [
            NewTopic(
                name=topic,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )
            for topic in topics
        ]
        
        try:
            admin_client = self._get_admin_client()
            admin_client.create_topics(new_topics=new_topics, validate_only=False)
            logger.info(f"Successfully created topics: {topics}")
            
        except TopicAlreadyExistsError:
            logger.info("Topics already exist, skipping creation")
            
        except KafkaError as e:
            logger.error(f"Failed to create topics: {e}")
            raise
    
    def list_topics(self) -> List[str]:
        """
        List all available Kafka topics.
        
        Returns:
            List of topic names
        """
        try:
            admin_client = self._get_admin_client()
            topics = admin_client.list_topics()
            logger.info(f"Available topics: {topics}")
            return topics
            
        except KafkaError as e:
            logger.error(f"Failed to list topics: {e}")
            return []
    
    def topic_exists(self, topic_name: str) -> bool:
        """
        Check if a topic exists.
        
        Args:
            topic_name: Name of the topic to check
            
        Returns:
            True if topic exists, False otherwise
        """
        topics = self.list_topics()
        return topic_name in topics
    
    def ensure_topics_exist(self) -> None:
        """
        Ensure all required topics exist, create if needed.
        """
        required_topics = self.kafka_config.get_all_topics()
        
        for topic in required_topics:
            if not self.topic_exists(topic):
                logger.warning(f"Topic '{topic}' does not exist, creating...")
                self.create_topics()
                break
        else:
            logger.info("All required topics exist")
    
    def close(self) -> None:
        """Close the admin client."""
        if self.admin_client:
            self.admin_client.close()
            logger.info("Topic Manager closed")
