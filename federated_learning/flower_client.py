"""
Flower Client for Federated Summarization.

Implements the FL client with:
- FedProx optimization
- LoRA adapters
- Kafka consumer for data streaming
- LED model for summarization
"""

import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
import flwr as fl

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import ModelConfig
from configs.federated_config import FederatedConfig
from configs.kafka_config import KafkaConfig
from model.model_loader import ModelLoader
from model.tokenizer_utils import TokenizerUtils
from federated_learning.lora_manager import LoRAManager
from federated_learning.fedprox_optimizer import FedProxOptimizer
from utils.checkpoint_manager import CheckpointManager
from utils.kafka_offset_manager import KafkaOffsetManager
from utils.logger import setup_logger


logger = setup_logger(__name__, log_level=os.getenv("LOG_LEVEL", "INFO"))


class FederatedSummarizationClient(fl.client.NumPyClient):
    """
    Flower client for federated document summarization.
    
    This client implements:
    1. FedProx algorithm for robust federated optimization
    2. LoRA adapters for parameter-efficient fine-tuning
    3. Kafka consumer for streaming data ingestion
    4. LED model for long-document summarization
    
    Attributes:
        client_id: Unique identifier for this client
        model: PEFT model with LoRA adapters
        tokenizer: Tokenizer for LED model
        optimizer: FedProx optimizer
        kafka_consumer: Kafka consumer for data
        checkpoint_manager: Manages LoRA checkpoints
        current_round: Current federated learning round
    """
    
    def __init__(
        self,
        client_id: str,
        model_config: Optional[ModelConfig] = None,
        fed_config: Optional[FederatedConfig] = None,
        kafka_config: Optional[KafkaConfig] = None
    ):
        """
        Initialize the Federated Summarization Client.
        
        Args:
            client_id: Client identifier (health, finance, legal)
            model_config: Model configuration
            fed_config: Federated learning configuration
            kafka_config: Kafka configuration
        """
        super().__init__()
        
        self.client_id = client_id
        self.model_config = model_config or ModelConfig.from_env()
        self.fed_config = fed_config or FederatedConfig.from_env()
        self.kafka_config = kafka_config or KafkaConfig.from_env()
        
        logger.info(f"Initializing Federated Client: {client_id}")
        
        # Initialize components
        self._init_model()
        self._init_kafka_consumer()
        self._init_managers()
        
        self.current_round = 0
        self.training_data = []
        
        logger.info(f"Client {client_id} initialized successfully")
    
    def _init_model(self) -> None:
        """Initialize the model with LoRA adapters."""
        logger.info("Loading model and tokenizer...")
        
        # Load model and tokenizer
        import time as _time
        start_time = _time.time()
        model_loader = ModelLoader(self.model_config)
        self.base_model, self.tokenizer = model_loader.load_model_and_tokenizer()
        logger.info(f"Model loaded in {(_time.time() - start_time):.2f} seconds.")
        
        start_token_time = _time.time()
        # Prepare for training
        self.base_model = ModelLoader.prepare_model_for_training(self.base_model)
        logger.info(f"Model prepared for training in {(_time.time() - start_token_time):.2f} seconds.")
        
        # Inject LoRA adapters
        lora_manager = LoRAManager(self.fed_config)
        self.model = lora_manager.inject_lora_adapters(self.base_model)
        
        # Initialize optimizer
        self.optimizer = FedProxOptimizer(
            self.model.parameters(),
            lr=self.fed_config.learning_rate,
            mu=self.fed_config.fedprox_mu,
            weight_decay=self.fed_config.weight_decay
        )
        
        logger.info("Model initialized with LoRA adapters")
    
    def _init_kafka_consumer(self) -> None:
        """Initialize Kafka consumer for data streaming."""
        topic = self.kafka_config.get_topic_by_client(self.client_id)
        
        logger.info(f"Initializing Kafka consumer for topic: {topic}")
        
        consumer_config = self.kafka_config.get_consumer_config(
            group_id=f"{self.client_id}-client"
        )
        
        self.kafka_consumer = KafkaConsumer(
            topic,
            **consumer_config,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        logger.info("Kafka consumer initialized")
    
    def _init_managers(self) -> None:
        """Initialize checkpoint and offset managers."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.fed_config.checkpoint_dir
        )
        self.offset_manager = KafkaOffsetManager()
        
        logger.info("Managers initialized")
    
    def _load_training_data(self, num_samples: int) -> List[Dict]:
        """
        Load training data from Kafka.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            List of training documents
        """
        logger.info(f"Loading {num_samples} training samples from Kafka...")
        
        data = []
        timeout_ms = 5000
        
        try:
            for message in self.kafka_consumer:
                document = message.value
                data.append(document)
                
                # Commit offset
                topic = message.topic
                partition = message.partition
                offset = message.offset
                self.offset_manager.commit_offset(
                    self.client_id, topic, partition, offset
                )
                
                if len(data) >= num_samples:
                    break
            
            logger.info(f"Loaded {len(data)} samples")
            return data
            
        except KafkaError as e:
            logger.error(f"Kafka error while loading data: {e}")
            return data
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters (LoRA adapters only).
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of parameter arrays
        """
        # Only send trainable parameters (LoRA adapters)
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.detach().cpu().numpy())
        
        logger.debug(f"Sending {len(params)} parameter tensors")
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.
        
        Args:
            parameters: List of parameter arrays from server
        """
        logger.info("Receiving parameters from server...")
        
        # Update only trainable parameters
        params_dict = {}
        param_idx = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_param = torch.from_numpy(parameters[param_idx]).to(param.device)
                param.data = new_param
                params_dict[name] = new_param
                param_idx += 1
        
        # Set global parameters for FedProx
        self.optimizer.set_global_params(params_dict)
        
        logger.info("Parameters updated")
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model for one round.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        self.current_round = config.get("round", 0)
        logger.info(f"Starting training round {self.current_round}")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Load training data if needed
        if not self.training_data:
            self.training_data = self._load_training_data(
                num_samples=self.fed_config.steps_per_round * self.fed_config.batch_size
            )
        
        # Train for fixed number of steps
        metrics = self._train_steps(self.fed_config.steps_per_round)
        
        # Save checkpoint
        if self.current_round % self.fed_config.save_every_n_rounds == 0:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                round_num=self.current_round,
                client_id=self.client_id,
                metadata=metrics
            )
        
        # Get updated parameters
        updated_params = self.get_parameters({})
        
        logger.info(f"Round {self.current_round} completed - Loss: {metrics['train_loss']:.4f}")
        
        return updated_params, len(self.training_data), metrics
    
    def _train_steps(self, num_steps: int) -> Dict:
        """
        Train for a fixed number of steps.
        
        Args:
            num_steps: Number of training steps
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        device = next(self.model.parameters()).device
        
        import time as _time
        for step in range(num_steps):
            step_start = _time.time()
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Preparing batch...")
            
            batch_texts = []
            batch_summaries = []
            for i in range(self.fed_config.batch_size):
                batch_idx = (step * self.fed_config.batch_size + i) % len(self.training_data)
                sample = self.training_data[batch_idx]
                batch_texts.append(sample["text"])
                batch_summaries.append(sample["summary"])

            tok_start = _time.time()
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Tokenizing inputs...")
            inputs = TokenizerUtils.tokenize_with_global_attention(
                texts=batch_texts,
                tokenizer=self.tokenizer,
                max_length=self.model_config.max_input_length
            )
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Tokenizing inputs done in {(_time.time()-tok_start):.2f}s.")

            tok_label_start = _time.time()
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Tokenizing labels...")
            labels = self.tokenizer(
                batch_summaries,
                max_length=self.model_config.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Tokenizing labels done in {(_time.time()-tok_label_start):.2f}s.")

            move_start = _time.time()
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Moving tensors to device {device}...")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Move to device done in {(_time.time()-move_start):.2f}s.")

            fwd_start = _time.time()
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Forward pass...")
            outputs = self.model(**inputs, labels=labels, use_cache=False)
            loss = outputs.loss
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Forward pass done in {(_time.time()-fwd_start):.2f}s.")

            opt_start = _time.time()
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - FedProx optimizer step...")
            total_loss_with_prox = self.optimizer.step(self.model, loss)
            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Optimizer step done in {(_time.time()-opt_start):.2f}s.")

            total_loss += loss.item()

            logger.info(f"[TRAIN] Step {step + 1}/{num_steps} - Loss: {loss.item():.4f} | Total step time: {(_time.time()-step_start):.2f}s.")
        
        avg_loss = total_loss / num_steps
        
        return {
            "train_loss": avg_loss,
            "num_steps": num_steps,
            "client_id": self.client_id
        }
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model (placeholder - full evaluation in separate module).
        
        Args:
            parameters: Model parameters
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        # For now, return minimal evaluation (no separate eval dataset yet)
        # Full ROUGE/BERTScore evaluation will be done separately
        # Return num_samples=1 to avoid division by zero
        return 0.0, 1, {"client_id": self.client_id}


def main():
    """Main entry point for the Flower client."""
    # Get client configuration from environment
    client_id = os.getenv("CLIENT_ID", "health")
    server_address = os.getenv("FLOWER_SERVER_ADDRESS", "localhost:8080")
    
    logger.info(f"Starting Flower client: {client_id}")
    logger.info(f"Connecting to server: {server_address}")
    
    # Wait for server to be ready
    time.sleep(15)
    
    try:
        # Create client
        client = FederatedSummarizationClient(client_id=client_id)
        
        # Start Flower client
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        
        logger.info(f"Client {client_id} finished successfully")
        
    except Exception as e:
        logger.error(f"Client {client_id} failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
