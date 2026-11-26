#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Time Series Preprocessor
Handles normalization, interpolation, and formatting of time series data
"""

import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Preprocesses time series data for TTM-R2 model"""
    
    def __init__(self):
        """Initialize preprocessor"""
        logger.info("Time series preprocessor initialized")
    
    def preprocess_batch(
        self,
        time_series_list: List[Dict],
        context_length: int = 512
    ) -> List[Dict]:
        """
        Preprocess a batch of time series
        
        Args:
            time_series_list: List of raw time series
            context_length: Target length for time series
            
        Returns:
            List of preprocessed time series
        """
        processed = []
        
        for ts in time_series_list:
            try:
                processed_ts = self._preprocess_single(ts, context_length)
                processed.append(processed_ts)
            except Exception as e:
                logger.error(f"Failed to preprocess series {ts.get('series_id', 'unknown')}: {str(e)}")
                # Skip failed series
                continue
        
        logger.info(f"Preprocessed {len(processed)}/{len(time_series_list)} series")
        return processed
    
    def _preprocess_single(
        self,
        ts: Dict,
        context_length: int
    ) -> Dict:
        """Preprocess a single time series"""
        
        values = np.array(ts['values'], dtype=np.float64)
        timestamps = np.array(ts['timestamps'], dtype=np.int64)
        
        # Handle missing values
        values, timestamps = self._handle_missing_values(values, timestamps)
        
        # Resample to regular intervals if needed
        values, timestamps = self._resample_to_regular_intervals(values, timestamps)
        
        # Truncate or pad to context length
        values, timestamps = self._adjust_length(values, timestamps, context_length)
        
        # Normalize
        normalized_values, mean, std = self._normalize(values)
        
        # Create processed time series
        processed = {
            'series_id': ts['series_id'],
            'metric_name': ts['metric_name'],
            'attributes': ts['attributes'],
            'values': values,
            'timestamps': timestamps,
            'normalized_values': normalized_values,
            'mean': mean,
            'std': std,
            'original_length': len(ts['values'])
        }
        
        return processed
    
    def _handle_missing_values(
        self,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values (NaN, inf) in time series
        
        Args:
            values: Time series values
            timestamps: Corresponding timestamps
            
        Returns:
            Cleaned values and timestamps
        """
        # Find valid indices
        valid_mask = np.isfinite(values)
        
        if not np.any(valid_mask):
            # All values are invalid - return zeros
            logger.warning("All values are invalid, returning zeros")
            return np.zeros_like(values), timestamps
        
        # Remove invalid values
        values = values[valid_mask]
        timestamps = timestamps[valid_mask]
        
        return values, timestamps
    
    def _resample_to_regular_intervals(
        self,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample time series to regular intervals
        
        Args:
            values: Time series values
            timestamps: Corresponding timestamps (nanoseconds)
            
        Returns:
            Resampled values and timestamps
        """
        if len(timestamps) < 2:
            return values, timestamps
        
        # Calculate median interval
        intervals = np.diff(timestamps)
        median_interval = np.median(intervals)
        
        # Check if already regular (within 10% tolerance)
        if np.all(np.abs(intervals - median_interval) / median_interval < 0.1):
            return values, timestamps
        
        # Create regular timestamp grid
        start_time = timestamps[0]
        end_time = timestamps[-1]
        num_points = int((end_time - start_time) / median_interval) + 1
        regular_timestamps = np.linspace(start_time, end_time, num_points, dtype=np.int64)
        
        # Interpolate values to regular grid
        regular_values = np.interp(regular_timestamps, timestamps, values)
        
        logger.debug(f"Resampled from {len(values)} to {len(regular_values)} points")
        
        return regular_values, regular_timestamps
    
    def _adjust_length(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        target_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust time series length to target length
        
        Args:
            values: Time series values
            timestamps: Corresponding timestamps
            target_length: Desired length
            
        Returns:
            Adjusted values and timestamps
        """
        current_length = len(values)
        
        if current_length == target_length:
            return values, timestamps
        
        elif current_length > target_length:
            # Truncate - keep most recent data
            values = values[-target_length:]
            timestamps = timestamps[-target_length:]
            logger.debug(f"Truncated from {current_length} to {target_length}")
        
        else:
            # Pad with zeros at the beginning
            pad_length = target_length - current_length
            values = np.pad(values, (pad_length, 0), mode='constant', constant_values=0)
            
            # Extend timestamps backwards
            if len(timestamps) > 1:
                interval = timestamps[1] - timestamps[0]
            else:
                interval = 60_000_000_000  # 1 minute default in nanoseconds
            
            pad_timestamps = np.array([
                timestamps[0] - (pad_length - i) * interval
                for i in range(pad_length)
            ], dtype=np.int64)
            
            timestamps = np.concatenate([pad_timestamps, timestamps])
            logger.debug(f"Padded from {current_length} to {target_length}")
        
        return values, timestamps
    
    def _normalize(
        self,
        values: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Normalize time series using z-score normalization
        
        Args:
            values: Time series values
            
        Returns:
            Tuple of (normalized_values, mean, std)
        """
        mean = np.mean(values)
        std = np.std(values)
        
        # Avoid division by zero
        if std < 1e-8:
            std = 1.0
        
        normalized = (values - mean) / std
        
        return normalized, float(mean), float(std)
    
    def denormalize(
        self,
        normalized_values: np.ndarray,
        mean: float,
        std: float
    ) -> np.ndarray:
        """
        Denormalize time series
        
        Args:
            normalized_values: Normalized values
            mean: Original mean
            std: Original std
            
        Returns:
            Denormalized values
        """
        return normalized_values * std + mean

# Made with Bob
