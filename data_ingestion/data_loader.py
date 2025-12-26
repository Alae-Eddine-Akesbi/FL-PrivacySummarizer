"""
Dataset Loader for HuggingFace Datasets.

This module handles loading and preprocessing of datasets from HuggingFace
for the three departments: Health, Finance, and Legal.
"""

import logging
from typing import Dict, List
from datasets import load_dataset, Dataset


logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and preprocesses datasets from HuggingFace.
    
    Handles dataset-specific preprocessing for:
    - Health: PubMed Summarization
    - Finance: ECTSum
    - Legal: BillSum
    """
    
    # Dataset-specific column mappings
    COLUMN_MAPPING = {
        "health": {
            "text_column": "article",
            "summary_column": "abstract",
        },
        "finance": {
            "text_column": "text",
            "summary_column": "summary",
        },
        "legal": {
            "text_column": "text",
            "summary_column": "summary",
        },
    }
    
    def load_dataset(
        self,
        dataset_name: str,
        client_type: str,
        num_samples: int = 20000,
        split: str = "train"
    ) -> List[Dict[str, str]]:
        """
        Load a dataset from HuggingFace and preprocess it.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            client_type: Type of client (health, finance, legal)
            num_samples: Number of samples to load
            split: Dataset split to use
            
        Returns:
            List of dictionaries with 'text' and 'summary' keys
            
        Raises:
            ValueError: If client_type is not recognized
            RuntimeError: If dataset loading fails
        """
        if client_type not in self.COLUMN_MAPPING:
            raise ValueError(
                f"Unknown client_type: {client_type}. "
                f"Must be one of {list(self.COLUMN_MAPPING.keys())}"
            )
        
        logger.info(f"Loading dataset '{dataset_name}' for {client_type}...")
        
        try:
            # Load dataset from HuggingFace with fallback strategies
            dataset = None
            errors = []
            
            # Strategy 1: Try with trust_remote_code and force_redownload
            try:
                dataset = load_dataset(dataset_name, split=split, trust_remote_code=True, download_mode="force_redownload")
            except (TypeError, ValueError) as e:
                errors.append(f"Strategy 1 failed: {str(e)[:100]}")
                
                # Strategy 2: Try without trust_remote_code
                try:
                    dataset = load_dataset(dataset_name, split=split, download_mode="force_redownload")
                except (TypeError, ValueError) as e2:
                    errors.append(f"Strategy 2 failed: {str(e2)[:100]}")
                    
                    # Strategy 3: Try with minimal parameters
                    try:
                        dataset = load_dataset(dataset_name, split=split)
                    except Exception as e3:
                        errors.append(f"Strategy 3 failed: {str(e3)[:100]}")
                        raise RuntimeError(f"All loading strategies failed. Errors: {'; '.join(errors)}")
            
            if dataset is None:
                raise RuntimeError("Dataset loading returned None")
            
            # Handle dataset size
            if isinstance(dataset, Dataset):
                total_size = len(dataset)
                num_samples = min(num_samples, total_size)
                logger.info(f"Dataset size: {total_size}, loading {num_samples} samples")
                
                # Take subset if needed
                if num_samples < total_size:
                    dataset = dataset.select(range(num_samples))
            
            # Get column names for this client type
            column_map = self.COLUMN_MAPPING[client_type]
            text_col = column_map["text_column"]
            summary_col = column_map["summary_column"]
            
            # Preprocess and extract data
            documents = []
            for item in dataset:
                # Extract text and summary
                text = str(item.get(text_col, ""))
                summary = str(item.get(summary_col, ""))
                
                # Skip empty documents
                if not text.strip() or not summary.strip():
                    continue
                
                documents.append({
                    "text": text,
                    "summary": summary
                })
            
            logger.info(f"Successfully loaded {len(documents)} valid documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise RuntimeError(f"Dataset loading failed: {e}")
    
    def validate_document(self, document: Dict[str, str]) -> bool:
        """
        Validate a document has required fields.
        
        Args:
            document: Document dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["text", "summary"]
        
        if not all(key in document for key in required_keys):
            return False
        
        if not document["text"].strip() or not document["summary"].strip():
            return False
        
        return True
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            
        Returns:
            Dictionary with dataset information
        """
        try:
            dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
            
            return {
                "name": dataset_name,
                "num_rows": len(dataset) if isinstance(dataset, Dataset) else None,
                "columns": dataset.column_names if isinstance(dataset, Dataset) else None,
            }
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {"name": dataset_name, "error": str(e)}
