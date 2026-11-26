#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Model Loader for IBM Granite TTM (TinyTimeMixer)
Handles loading and initialization of the TTM model
"""

import torch
import logging
import psutil
import os

try:
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    TSFM_AVAILABLE = True
except ImportError:
    TSFM_AVAILABLE = False
    TinyTimeMixerForPrediction = None

logger = logging.getLogger(__name__)


class TTMModelLoader:
    """Loads and manages the TTM model"""
    
    def __init__(self, model_path: str = "ibm/TTM"):
        """
        Initialize model loader
        
        Args:
            model_path: Path to model or HuggingFace model ID (default: ibm/TTM)
        """
        if not TSFM_AVAILABLE:
            raise ImportError(
                "tsfm_public library is required to load TTM models. "
                "Install it with: pip install tsfm_public"
            )
        
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        
        logger.info(f"Model loader initialized for: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def load(self):
        """
        Load the TTM model using tsfm_public library
        
        Returns:
            Loaded model
        """
        try:
            logger.info(f"Loading TTM model from {self.model_path}")
            
            # Load model using TinyTimeMixerForPrediction
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                self.model_path,
                revision="main"
            )
            
            # Move to appropriate device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            logger.info(f"Model parameters: {self._count_parameters():,}")
            logger.info(f"Memory usage: {self.get_memory_usage():.2f} MB")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def get_model_version(self) -> str:
        """Get model version string"""
        if self.model and hasattr(self.model, 'config'):
            return getattr(self.model.config, 'model_version', 'unknown')
        return 'unknown'
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB
        
        Returns:
            Memory usage in megabytes
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _get_device(self) -> torch.device:
        """
        Determine the best device to use
        
        Returns:
            torch.device (cuda, mps, or cpu)
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple MPS available")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        return device
    
    def _count_parameters(self) -> int:
        """Count total model parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

# Made with Bob
