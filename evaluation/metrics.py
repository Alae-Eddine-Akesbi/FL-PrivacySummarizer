"""
Metrics Calculator for ROUGE and BERTScore.

Implements calculation of ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore.
"""

import logging
from typing import Dict, List
from rouge_score import rouge_scorer
from bert_score import score as bert_score


logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates evaluation metrics for summarization.
    
    Metrics:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    - BERTScore: Contextual embedding similarity
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        logger.info("Metrics Calculator initialized")
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": sum(rouge1_scores) / len(rouge1_scores),
            "rouge2": sum(rouge2_scores) / len(rouge2_scores),
            "rougeL": sum(rougeL_scores) / len(rougeL_scores),
        }
    
    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = "en"
    ) -> Dict[str, float]:
        """
        Calculate BERTScore.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            lang: Language code
            
        Returns:
            Dictionary with BERTScore
        """
        P, R, F1 = bert_score(
            predictions,
            references,
            lang=lang,
            verbose=False
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    
    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Calculating metrics for {len(predictions)} samples...")
        
        rouge_scores = self.calculate_rouge(predictions, references)
        bert_scores = self.calculate_bertscore(predictions, references)
        
        metrics = {**rouge_scores, **bert_scores}
        
        logger.info(f"ROUGE-1: {metrics['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {metrics['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {metrics['rougeL']:.4f}")
        logger.info(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
        
        return metrics
