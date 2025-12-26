"""
Metrics Aggregator for Federated Evaluation.

Aggregates metrics from multiple clients for global evaluation.
"""

import logging
from typing import Dict, List
import numpy as np


logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Aggregates evaluation metrics across federated clients.
    
    Performs weighted averaging of metrics based on client sample counts.
    """
    
    @staticmethod
    def aggregate_metrics(
        client_metrics: List[Dict],
        client_samples: List[int]
    ) -> Dict[str, float]:
        """
        Aggregate metrics from multiple clients.
        
        Args:
            client_metrics: List of metric dictionaries from clients
            client_samples: List of sample counts per client
            
        Returns:
            Aggregated metrics dictionary
        """
        if not client_metrics:
            return {}
        
        total_samples = sum(client_samples)
        aggregated = {}
        
        # Get all metric keys
        metric_keys = client_metrics[0].keys()
        
        for key in metric_keys:
            # Weighted average
            weighted_sum = sum(
                metrics[key] * samples
                for metrics, samples in zip(client_metrics, client_samples)
            )
            aggregated[key] = weighted_sum / total_samples
        
        logger.info("Aggregated global metrics:")
        for key, value in aggregated.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return aggregated
    
    @staticmethod
    def compute_statistics(
        client_metrics: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics (mean, std, min, max) across clients.
        
        Args:
            client_metrics: List of metric dictionaries
            
        Returns:
            Dictionary with statistics for each metric
        """
        if not client_metrics:
            return {}
        
        stats = {}
        metric_keys = client_metrics[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in client_metrics]
            stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        
        return stats
