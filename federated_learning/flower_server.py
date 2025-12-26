"""
Flower Server with Custom Aggregation Strategy.

Implements the federated learning server with FedAvg aggregation
and coordination of the three clients.
"""

import logging
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.federated_config import FederatedConfig
from utils.logger import setup_logger


logger = setup_logger(__name__, log_level=os.getenv("LOG_LEVEL", "INFO"))


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy with enhanced logging and metrics aggregation.
    
    Extends Flower's FedAvg to add:
    - Detailed round-by-round logging
    - Metrics aggregation across clients
    - Progress tracking
    """
    
    def __init__(
        self,
        *args,
        fed_config: Optional[FederatedConfig] = None,
        **kwargs
    ):
        """
        Initialize the custom strategy.
        
        Args:
            fed_config: Federated learning configuration
            *args, **kwargs: Arguments for parent FedAvg class
        """
        super().__init__(*args, **kwargs)
        self.fed_config = fed_config or FederatedConfig.from_env()
        self.round_metrics = []
        
        # Create directory for saving global model
        self.global_model_dir = Path("/app/checkpoints/global_model")
        self.global_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("FedAvg strategy initialized with custom logging")
        logger.info(f"Global model will be saved to: {self.global_model_dir}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from clients.
        
        Args:
            server_round: Current round number
            results: Training results from clients
            failures: Failed client updates
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        logger.info("=" * 60)
        logger.info(f"Round {server_round}: Aggregating results from {len(results)} clients")
        
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} clients failed")
        
        # Log client metrics
        for client_proxy, fit_res in results:
            client_metrics = fit_res.metrics
            logger.info(
                f"  Client {client_metrics.get('client_id', 'unknown')}: "
                f"Loss={client_metrics.get('train_loss', 0):.4f}, "
                f"Samples={fit_res.num_examples}"
            )
        
        # Aggregate using parent method (FedAvg)
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Calculate aggregated metrics
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        weighted_loss = sum([
            fit_res.metrics.get("train_loss", 0) * fit_res.num_examples
            for _, fit_res in results
        ]) / total_samples if total_samples > 0 else 0.0
        
        metrics["aggregated_loss"] = weighted_loss
        metrics["num_clients"] = len(results)
        metrics["total_samples"] = total_samples
        
        # Store metrics
        round_data = {
            "round": server_round,
            "loss": weighted_loss,
            "num_clients": len(results),
            "total_samples": total_samples
        }
        self.round_metrics.append(round_data)
        
        logger.info(f"Round {server_round}: Aggregated Loss = {weighted_loss:.4f}")
        
        # Save the global aggregated model
        if aggregated_parameters is not None:
            self._save_global_model(aggregated_parameters, server_round, round_data)
        
        logger.info("=" * 60)
        
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: Evaluation results from clients
            failures: Failed evaluations
            
        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        if not results:
            return None, {}
        
        logger.info(f"Round {server_round}: Aggregating evaluation from {len(results)} clients")
        
        # Aggregate using parent method
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        logger.info(f"Round {server_round}: Evaluation Loss = {loss_str}")
        
        return loss, metrics
    
    def _save_global_model(
        self, 
        parameters: fl.common.Parameters, 
        server_round: int, 
        metrics: Dict[str, Any]
    ) -> None:
        """
        Save the global aggregated model after each round.
        
        Args:
            parameters: Aggregated model parameters from FedAvg
            server_round: Current round number
            metrics: Round metrics (loss, num_clients, total_samples)
        """
        try:
            # Create round-specific directory
            round_dir = self.global_model_dir / f"round_{server_round}"
            round_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert Parameters to numpy arrays
            from flwr.common import parameters_to_ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            
            # Save parameters as numpy arrays
            params_file = round_dir / "global_parameters.npz"
            import numpy as np
            np.savez(params_file, *ndarrays)
            
            # Save metadata
            metadata = {
                "round": server_round,
                "aggregated_loss": metrics.get("loss", 0.0),
                "num_clients": metrics.get("num_clients", 0),
                "total_samples": metrics.get("total_samples", 0),
                "num_parameters": len(ndarrays),
                "parameter_shapes": [arr.shape for arr in ndarrays],
                "timestamp": time.time()
            }
            
            metadata_file = round_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Global model saved: {round_dir}")
            logger.info(f"   - Parameters: {len(ndarrays)} tensors")
            logger.info(f"   - Loss: {metadata['aggregated_loss']:.4f}")
            logger.info(f"   - Clients: {metadata['num_clients']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save global model for round {server_round}: {e}")



def create_strategy(fed_config: Optional[FederatedConfig] = None) -> FedAvgWithLogging:
    """
    Create the federated learning strategy.
    
    Args:
        fed_config: Federated learning configuration
        
    Returns:
        Configured strategy instance
    """
    fed_config = fed_config or FederatedConfig.from_env()
    
    def fit_config(server_round: int):
        """Return training configuration with round number."""
        return {"round": server_round}
    
    strategy = FedAvgWithLogging(
        fed_config=fed_config,
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=fed_config.min_fit_clients,
        min_evaluate_clients=fed_config.min_evaluate_clients,
        min_available_clients=fed_config.min_available_clients,
        on_fit_config_fn=fit_config,
    )
    
    logger.info("Strategy created successfully")
    return strategy


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    fed_config: Optional[FederatedConfig] = None
) -> None:
    """
    Start the Flower federated learning server.
    
    Args:
        server_address: Address to bind the server
        num_rounds: Number of federated learning rounds
        fed_config: Federated learning configuration
    """
    fed_config = fed_config or FederatedConfig.from_env()
    num_rounds = fed_config.num_rounds
    
    logger.info("=" * 80)
    logger.info("Starting Flower Federated Learning Server")
    logger.info("=" * 80)
    logger.info(f"Server Address: {server_address}")
    logger.info(f"Number of Rounds: {num_rounds}")
    logger.info(f"Min Clients (Fit): {fed_config.min_fit_clients}")
    logger.info(f"Min Clients (Eval): {fed_config.min_evaluate_clients}")
    logger.info(f"Steps per Round: {fed_config.steps_per_round}")
    logger.info(f"FedProx µ: {fed_config.fedprox_mu}")
    logger.info("=" * 80)
    
    # Create strategy
    strategy = create_strategy(fed_config)
    
    # Configure server
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    # Start server
    try:
        fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=strategy,
        )
        
        logger.info("=" * 80)
        logger.info("Federated Learning Completed Successfully!")
        logger.info("=" * 80)
        
        # Print final summary
        _print_training_summary(strategy)
        
    except Exception as e:
        logger.error(f"Server failed with error: {e}", exc_info=True)
        raise


def _print_training_summary(strategy: FedAvgWithLogging) -> None:
    """
    Print training summary statistics.
    
    Args:
        strategy: Strategy instance with metrics
    """
    if not strategy.round_metrics:
        return
    
    logger.info("Training Summary:")
    logger.info("-" * 80)
    
    for metrics in strategy.round_metrics:
        logger.info(
            f"Round {metrics['round']}: "
            f"Loss={metrics['loss']:.4f}, "
            f"Clients={metrics['num_clients']}, "
            f"Samples={metrics['total_samples']}"
        )
    
    # Final statistics
    final_loss = strategy.round_metrics[-1]["loss"]
    initial_loss = strategy.round_metrics[0]["loss"]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100
    
    logger.info("-" * 80)
    logger.info(f"Initial Loss: {initial_loss:.4f}")
    logger.info(f"Final Loss: {final_loss:.4f}")
    logger.info(f"Improvement: {improvement:.2f}%")
    logger.info("=" * 80)


def main():
    """Main entry point for the Flower server."""
    server_address = os.getenv("FLOWER_SERVER_ADDRESS", "0.0.0.0:8080")
    
    try:
        start_server(server_address=server_address)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
