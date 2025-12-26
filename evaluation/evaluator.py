"""
Model Evaluator.

End-to-end evaluation pipeline for the federated summarization model.
"""

import logging
from typing import Dict, List
from model.led_summarizer import LEDSummarizer
from evaluation.metrics import MetricsCalculator


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates the summarization model on test data.
    
    Provides end-to-end evaluation pipeline from documents to metrics.
    """
    
    def __init__(self, summarizer: LEDSummarizer):
        """
        Initialize the evaluator.
        
        Args:
            summarizer: LED summarizer instance
        """
        self.summarizer = summarizer
        self.metrics_calculator = MetricsCalculator()
        logger.info("Model Evaluator initialized")
    
    def evaluate(
        self,
        test_documents: List[str],
        reference_summaries: List[str],
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_documents: List of documents to summarize
            reference_summaries: List of reference summaries
            batch_size: Batch size for generation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_documents)} documents...")
        
        # Generate summaries in batches
        predictions = []
        for i in range(0, len(test_documents), batch_size):
            batch = test_documents[i:i + batch_size]
            batch_preds = self.summarizer.summarize(batch)
            predictions.extend(batch_preds)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions=predictions,
            references=reference_summaries
        )
        
        return metrics
