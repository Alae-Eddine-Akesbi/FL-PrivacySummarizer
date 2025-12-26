"""
Kafka Producer for Document Ingestion.

This module implements a robust Single Producer that loads datasets from
HuggingFace and routes documents to appropriate Kafka topics with intelligent
error handling and retry mechanisms.
"""

import json
import logging
import os
import sys
import time
from typing import Dict, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.kafka_config import KafkaConfig
from data_ingestion.data_loader import DatasetLoader
from data_ingestion.topic_manager import TopicManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KafkaDocumentProducer:
    """
    Single Producer for distributing documents to Kafka topics.
    
    This producer handles:
    - Loading datasets from HuggingFace
    - Intelligent routing to topic based on document type
    - Robust error handling with retry mechanism
    - Progress tracking and logging
    
    Attributes:
        kafka_config: Kafka configuration instance
        producer: KafkaProducer instance
        dataset_loader: DatasetLoader instance
        topic_manager: TopicManager instance
    """
    
    def __init__(
        self,
        kafka_config: Optional[KafkaConfig] = None,
        max_retries: int = 5,
        retry_backoff_ms: int = 1000
    ):
        """
        Initialize the Kafka Document Producer.
        
        Args:
            kafka_config: Kafka configuration (uses env if None)
            max_retries: Maximum number of retry attempts
            retry_backoff_ms: Backoff time between retries in milliseconds
        """
        self.kafka_config = kafka_config or KafkaConfig.from_env()
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        
        logger.info("Initializing Kafka Document Producer...")
        
        # Initialize components
        self.dataset_loader = DatasetLoader()
        self.topic_manager = TopicManager(self.kafka_config)
        
        # Initialize Kafka Producer
        self.producer = self._create_producer()
        
        # Statistics
        self.stats = {
            "health": {"sent": 0, "failed": 0},
            "finance": {"sent": 0, "failed": 0},
            "legal": {"sent": 0, "failed": 0},
        }
    
    def _create_producer(self) -> KafkaProducer:
        """
        Create and configure Kafka Producer with retry logic.
        
        Returns:
            Configured KafkaProducer instance
            
        Raises:
            RuntimeError: If producer creation fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self.kafka_config.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks=self.kafka_config.acks,
                    retries=self.kafka_config.retries,
                    compression_type=self.kafka_config.compression_type,
                    max_in_flight_requests_per_connection=5,
                    request_timeout_ms=30000,
                )
                logger.info("Kafka Producer created successfully")
                return producer
                
            except KafkaError as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_ms / 1000)
                else:
                    raise RuntimeError("Failed to create Kafka Producer after retries")
    
    def _send_with_retry(
        self,
        topic: str,
        document: Dict,
        key: Optional[str] = None
    ) -> bool:
        """
        Send a document to Kafka with retry mechanism.
        
        Args:
            topic: Kafka topic name
            document: Document data to send
            key: Optional partition key
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                future = self.producer.send(
                    topic,
                    value=document,
                    key=key.encode('utf-8') if key else None
                )
                
                # Wait for acknowledgment with timeout
                record_metadata = future.get(timeout=10)
                
                logger.debug(
                    f"Sent to {topic} - Partition: {record_metadata.partition}, "
                    f"Offset: {record_metadata.offset}"
                )
                return True
                
            except KafkaTimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_ms / 1000)
                    
            except KafkaError as e:
                logger.error(f"Kafka error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_ms / 1000)
        
        return False
    
    def produce_documents(
        self,
        client_type: str,
        dataset_name: str,
        num_documents: int = 20000
    ) -> None:
        """
        Load and produce documents for a specific client type.
        
        Args:
            client_type: Type of client (health, finance, legal)
            dataset_name: HuggingFace dataset identifier
            num_documents: Number of documents to produce
            
        Raises:
            ValueError: If client_type is not recognized
        """
        # Get appropriate topic
        try:
            topic = self.kafka_config.get_topic_by_client(client_type)
        except ValueError as e:
            logger.error(str(e))
            raise
        
        logger.info(f"Loading {num_documents} documents from {dataset_name}...")
        
        # Load dataset
        documents = self.dataset_loader.load_dataset(
            dataset_name=dataset_name,
            client_type=client_type,
            num_samples=num_documents
        )
        
        logger.info(f"Producing {len(documents)} documents to topic '{topic}'...")
        
        # Send documents with progress bar
        with tqdm(total=len(documents), desc=f"Producing {client_type}") as pbar:
            for idx, doc in enumerate(documents):
                # Prepare document with metadata
                document = {
                    "id": f"{client_type}_{idx}",
                    "text": doc["text"],
                    "summary": doc["summary"],
                    "client_type": client_type,
                    "timestamp": time.time()
                }
                
                # Send to Kafka
                success = self._send_with_retry(
                    topic=topic,
                    document=document,
                    key=f"{client_type}_{idx}"
                )
                
                # Update statistics
                if success:
                    self.stats[client_type]["sent"] += 1
                else:
                    self.stats[client_type]["failed"] += 1
                    logger.error(f"Failed to send document {idx} after retries")
                
                pbar.update(1)
        
        # Flush producer to ensure all messages are sent
        self.producer.flush()
        
        logger.info(
            f"Completed {client_type}: {self.stats[client_type]['sent']} sent, "
            f"{self.stats[client_type]['failed']} failed"
        )
    
    def produce_all_departments(self) -> None:
        """
        Produce documents for all three departments with their specific datasets.
        
        Each department uses its dedicated dataset:
        - Health: ccdv/pubmed-summarization
        - Finance: mrSoul7766/ECTSum  
        - Legal: FiscalNote/billsum
        """
        num_documents = int(os.getenv("DATASET_SIZE_PER_CLIENT", "20000"))
        
        logger.info("=" * 60)
        logger.info("Starting Full Dataset Ingestion Pipeline")
        logger.info("Using dedicated datasets per client")
        logger.info("=" * 60)
        
        # Define datasets for each client
        datasets_config = {
            "health": {
                "dataset_name": "ccdv/pubmed-summarization",
                "topic": self.kafka_config.health_topic
            },
            "finance": {
                "dataset_name": "mrSoul7766/ECTSum",
                "topic": self.kafka_config.finance_topic
            },
            "legal": {
                "dataset_name": "FiscalNote/billsum",
                "topic": self.kafka_config.legal_topic
            }
        }
        
        # Process each department with its specific dataset
        for client_type, config in datasets_config.items():
            logger.info(f"\\nProcessing {client_type.upper()} department...")
            logger.info(f"Dataset: {config['dataset_name']}")
            logger.info(f"Loading {num_documents} documents...")
            
            try:
                # Load dataset specific to this client
                documents = self.dataset_loader.load_dataset(
                    dataset_name=config["dataset_name"],
                    client_type=client_type,
                    num_samples=num_documents
                )
                
                # Send documents to appropriate topic
                topic = config["topic"]
                logger.info(f"Producing {len(documents)} documents to topic '{topic}'...")
                
                with tqdm(total=len(documents), desc=f"Producing {client_type}") as pbar:
                    for idx, doc in enumerate(documents):
                        document = {
                            "id": f"{client_type}_{idx}",
                            "text": doc["text"],
                            "summary": doc["summary"],
                            "client_type": client_type,
                            "timestamp": time.time()
                        }
                        
                        success = self._send_with_retry(
                            topic=topic,
                            document=document,
                            key=document["id"]
                        )
                        
                        if success:
                            self.stats[client_type]["sent"] += 1
                        else:
                            self.stats[client_type]["failed"] += 1
                        
                        pbar.update(1)
                
                self.producer.flush()
                logger.info(
                    f"Completed {client_type}: {self.stats[client_type]['sent']} sent, "
                    f"{self.stats[client_type]['failed']} failed"
                )
                
            except Exception as e:
                logger.error(f"Error processing {client_type} dataset: {e}")
                logger.exception("Detailed traceback:")
                continue
        
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print final statistics summary."""
        logger.info("=" * 60)
        logger.info("Ingestion Summary")
        logger.info("=" * 60)
        
        total_sent = sum(stats["sent"] for stats in self.stats.values())
        total_failed = sum(stats["failed"] for stats in self.stats.values())
        
        for client_type, stats in self.stats.items():
            logger.info(
                f"{client_type.upper()}: {stats['sent']} sent, {stats['failed']} failed"
            )
        
        logger.info("-" * 60)
        logger.info(f"TOTAL: {total_sent} sent, {total_failed} failed")
        if (total_sent + total_failed) > 0:
            logger.info(f"Success Rate: {total_sent / (total_sent + total_failed) * 100:.2f}%")
        else:
            logger.warning("No documents were processed")
        logger.info("=" * 60)
    
    def close(self) -> None:
        """Close the Kafka producer and cleanup resources."""
        if self.producer:
            logger.info("Closing Kafka Producer...")
            self.producer.flush()
            self.producer.close()
            logger.info("Producer closed successfully")


def main():
    """Main entry point for the producer script."""
    try:
        # Wait for Kafka to be ready
        logger.info("Waiting for Kafka to be ready...")
        time.sleep(10)
        
        # Create producer
        producer = KafkaDocumentProducer()
        
        # Produce documents for all departments
        producer.produce_all_departments()
        
        # Cleanup
        producer.close()
        
        logger.info("Producer finished successfully")
        
    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
    except Exception as e:
        logger.error(f"Producer failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
