"""
Inference Pipeline for Real-Time Summarization.

Phase 2: Post-training inference on new documents.
"""

import logging
import os
import sys
from typing import List, Optional
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import ModelConfig
from configs.federated_config import FederatedConfig
from model.model_loader import ModelLoader
from model.led_summarizer import LEDSummarizer
from federated_learning.lora_manager import LoRAManager


logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Inference pipeline for trained model.
    
    Loads trained LoRA adapters and performs summarization.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        fed_config: Optional[FederatedConfig] = None
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            checkpoint_path: Path to trained LoRA checkpoint
            model_config: Model configuration
            fed_config: Federated configuration
        """
        self.model_config = model_config or ModelConfig.from_env()
        self.fed_config = fed_config or FederatedConfig.from_env()
        self.checkpoint_path = checkpoint_path or self._find_latest_checkpoint()
        
        logger.info("Initializing Inference Pipeline...")
        self._load_model()
    
    def _find_latest_checkpoint(self) -> str:
        """Find the latest checkpoint."""
        checkpoint_dir = Path(self.fed_config.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Find all checkpoints
        checkpoints = list(checkpoint_dir.glob("*_round_*"))
        
        if not checkpoints:
            raise ValueError("No checkpoints found")
        
        # Get latest by round number
        latest = max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))
        
        logger.info(f"Using checkpoint: {latest}")
        return str(latest)
    
    def _load_model(self) -> None:
        """Load model with trained adapters."""
        logger.info("Loading model...")
        
        # Load base model
        model_loader = ModelLoader(self.model_config)
        base_model, tokenizer = model_loader.load_model_and_tokenizer()
        
        # Inject LoRA
        lora_manager = LoRAManager(self.fed_config)
        model = lora_manager.inject_lora_adapters(base_model)
        
        # Load trained weights
        if self.checkpoint_path:
            model.load_adapter(self.checkpoint_path, adapter_name="default")
            logger.info(f"Loaded adapters from: {self.checkpoint_path}")
        
        # Set to eval mode
        model.eval()
        
        # Create summarizer
        self.summarizer = LEDSummarizer(model, tokenizer, self.model_config)
        
        logger.info("Inference pipeline ready")
    
    def summarize(
        self,
        documents: List[str],
        max_length: int = 256
    ) -> List[str]:
        """
        Generate summaries for documents.
        
        Args:
            documents: List of documents to summarize
            max_length: Maximum summary length
            
        Returns:
            List of summaries
        """
        logger.info(f"Generating summaries for {len(documents)} documents...")
        
        summaries = self.summarizer.summarize(
            texts=documents,
            max_new_tokens=max_length
        )
        
        logger.info("Summaries generated")
        return summaries


def main():
    """Demo inference."""
    # Example document
    document = """
    Machine learning is a subset of artificial intelligence that focuses on 
    developing systems that can learn from and make decisions based on data. 
    It has applications in various fields including natural language processing, 
    computer vision, and recommendation systems. Deep learning, a subset of 
    machine learning, uses neural networks with multiple layers to learn 
    complex patterns in data.
    """
    
    try:
        pipeline = InferencePipeline()
        summaries = pipeline.summarize([document])
        
        print("\n" + "="*60)
        print("INFERENCE DEMO")
        print("="*60)
        print("\nOriginal Document:")
        print(document.strip())
        print("\nGenerated Summary:")
        print(summaries[0])
        print("="*60)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
