#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Anomaly Detector using IBM Granite TTM-R2
Implements anomaly detection logic using reconstruction and forecast errors
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in time series using TTM-R2 model"""
    
    def __init__(self, model):
        """
        Initialize anomaly detector
        
        Args:
            model: Loaded TTM-R2 model
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Sensitivity thresholds
        self.sensitivity_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }
        
        logger.info("Anomaly detector initialized")
    
    def detect_batch(
        self,
        time_series_list: List[Dict],
        threshold: float = 0.7,
        detection_mode: str = "reconstruction",
        sensitivity: str = "medium"
    ) -> List[Dict]:
        """
        Detect anomalies in a batch of time series
        
        Args:
            time_series_list: List of preprocessed time series
            threshold: Base anomaly threshold (0-1)
            detection_mode: "reconstruction", "forecast", or "both"
            sensitivity: "low", "medium", or "high"
            
        Returns:
            List of anomaly detection results
        """
        results = []
        
        # Adjust threshold based on sensitivity
        sensitivity_mult = self.sensitivity_multipliers.get(sensitivity, 1.0)
        adjusted_threshold = threshold * sensitivity_mult
        
        logger.info(f"Detecting anomalies for {len(time_series_list)} series")
        logger.info(f"Mode: {detection_mode}, Threshold: {adjusted_threshold:.3f}")
        
        for ts in time_series_list:
            try:
                result = self._detect_single(
                    ts,
                    adjusted_threshold,
                    detection_mode
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to detect anomalies for series {ts['series_id']}: {str(e)}",
                    exc_info=True
                )
                logger.error(
                    "Detection context: series_id=%s normalized_shape=%s values_shape=%s metric_name=%s",
                    ts.get('series_id'),
                    np.shape(ts.get('normalized_values')),
                    np.shape(ts.get('values')),
                    ts.get('metric_name'),
                )
                # Return default result on error
                results.append(self._create_error_result(ts['series_id']))
        
        return results
    
    def _detect_single(
        self,
        ts: Dict,
        threshold: float,
        detection_mode: str
    ) -> Dict:
        """Detect anomalies in a single time series"""
        
        # Prepare input tensor
        input_tensor = self._prepare_input(ts['normalized_values'])
        
        with torch.no_grad():
            if detection_mode == "reconstruction":
                anomaly_scores, reconstruction = self._reconstruction_based_detection(
                    input_tensor
                )
            elif detection_mode == "forecast":
                anomaly_scores = self._forecast_based_detection(
                    input_tensor
                )
                reconstruction = None
            else:  # both
                recon_scores, reconstruction = self._reconstruction_based_detection(
                    input_tensor
                )
                forecast_scores = self._forecast_based_detection(input_tensor)
                # Combine scores (average)
                anomaly_scores = (recon_scores + forecast_scores) / 2
        
        # Convert to numpy
        anomaly_scores_np = anomaly_scores.detach().cpu().numpy()
        
        # Binary classification
        is_anomaly = anomaly_scores_np > threshold
        
        # Overall score (max or mean)
        overall_score = float(np.max(anomaly_scores_np)) if anomaly_scores_np.size > 0 else 0.0
        
        # Generate explanation
        explanation = self._generate_explanation(
            anomaly_scores_np,
            is_anomaly,
            detection_mode
        )
        
        result = {
            'series_id': ts['series_id'],
            'anomaly_scores': anomaly_scores_np.flatten().tolist(),
            'is_anomaly': is_anomaly.flatten().tolist(),
            'overall_score': overall_score,
            'explanation': explanation,
            'predicted_value': 0.0,
            'actual_value': 0.0,
            'error_value': 0.0,
        }
        
        # Add forecast/reconstruction data if available
        if reconstruction is not None:
            forecast_values = reconstruction.detach().cpu().numpy()
            
            # For forecast-based detection, we only have errors for the forecast horizon
            # Get the actual values that correspond to the forecast
            forecast_len = forecast_values.shape[1]
            input_len = input_tensor.shape[1]
            
            if forecast_len < input_len:
                # Compare forecast with last forecast_len steps of input
                actual_values = input_tensor[:, -forecast_len:, :].detach().cpu().numpy()
                compare_values = forecast_values
                errors = np.abs(actual_values - compare_values)
            else:
                # Compare directly if same length
                min_len = min(forecast_len, input_len)
                actual_values = input_tensor[:, :min_len, :].detach().cpu().numpy()
                compare_values = forecast_values[:, :min_len, :]
                errors = np.abs(actual_values - compare_values)
            
            result['reconstruction'] = {
                'values': forecast_values.flatten().tolist(),
                'errors': errors.flatten().tolist()
            }
            result['predicted_value'] = float(compare_values.reshape(-1)[-1]) if compare_values.size > 0 else 0.0
            result['actual_value'] = float(actual_values.reshape(-1)[-1]) if actual_values.size > 0 else 0.0
            result['error_value'] = float(errors.reshape(-1)[-1]) if errors.size > 0 else 0.0
        
        return result
    
    def _reconstruction_based_detection(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies using the IBM Granite TTM model approach
        
        Based on the official notebook example:
        1. Pass input through model to get predictions
        2. Extract 'prediction_outputs' from model output
        3. Compare actual vs predicted values
        4. Use statistical threshold (mean + 2*std) for anomaly detection
        
        Args:
            input_tensor: Input time series tensor [batch, seq_len, channels]
            
        Returns:
            Tuple of (anomaly_scores, predictions)
        """
        # Forward pass through model - TTM expects named HuggingFace-style inputs.
        # TinyTimeMixerForPrediction.forward signature uses `past_values`.
        logger.info(f"Running model forward pass with input shape: {tuple(input_tensor.shape)}")
        outputs = self.model(
            past_values=input_tensor,
            return_loss=False,
            return_dict=True,
        )
        logger.info(f"Model output type: {type(outputs)}")
        
        # Extract prediction_outputs as shown in the notebook
        # The model may return an object, dict-like structure, tuple, or tensor.
        if hasattr(outputs, 'prediction_outputs'):
            predictions = outputs.prediction_outputs
        elif isinstance(outputs, dict):
            if 'prediction_outputs' in outputs:
                predictions = outputs['prediction_outputs']
            elif 'prediction' in outputs:
                predictions = outputs['prediction']
            elif 'predictions' in outputs:
                predictions = outputs['predictions']
            else:
                logger.warning("Dict model output missing prediction keys, using input fallback")
                predictions = input_tensor.clone()
        elif isinstance(outputs, tuple):
            predictions = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            predictions = outputs
        else:
            predictions = input_tensor.clone()
            logger.warning(f"Unexpected output type: {type(outputs)}, using fallback")
        
        # Ensure predictions is a tensor
        if not isinstance(predictions, torch.Tensor):
            logger.error(f"Unexpected prediction type: {type(predictions)}")
            predictions = input_tensor.clone()
        
        logger.info(
            f"Input shape: {tuple(input_tensor.shape)}, Prediction type: {type(predictions)}"
        )
        
        # Compare actual vs predicted (as in notebook: actual vs predicted)
        # Handle different prediction lengths
        pred_len = predictions.shape[1]
        input_len = input_tensor.shape[1]
        
        if pred_len == input_len:
            # Direct comparison
            actual = input_tensor
            error = torch.abs(actual - predictions)
        elif pred_len < input_len:
            # Predictions are shorter - compare last pred_len steps
            actual = input_tensor[:, -pred_len:, :]
            error = torch.abs(actual - predictions)
            
            # Pad error to match input length
            padding = torch.zeros(
                (error.shape[0], input_len - pred_len, error.shape[2]),
                device=error.device
            )
            error = torch.cat([padding, error], dim=1)
        else:
            # Predictions are longer - use first input_len steps
            actual = input_tensor
            predictions_trimmed = predictions[:, :input_len, :]
            error = torch.abs(actual - predictions_trimmed)
        
        # Calculate anomaly scores using statistical threshold (as in notebook)
        # threshold = mean(error) + 2 * std(error)
        anomaly_scores = self._calculate_anomaly_scores(error)
        
        return anomaly_scores, predictions
    
    def _forecast_based_detection(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect anomalies using forecast error
        
        Args:
            input_tensor: Input time series tensor
            
        Returns:
            Anomaly scores tensor
        """
        # Use first part for context, last part for comparison
        context_length = input_tensor.shape[1] // 2
        context = input_tensor[:, :context_length]
        actual = input_tensor[:, context_length:]
        
        # Generate forecast
        try:
            forecast = self.model.generate(context, max_length=actual.shape[1])
        except:
            # Fallback: use model output directly
            forecast = self.model(context)
        
        # Calculate forecast error
        error = torch.abs(actual - forecast[:, :actual.shape[1]])
        
        # Normalize to anomaly score
        anomaly_scores = self._normalize_scores(error)
        
        # Pad to match input length
        padding = torch.zeros(
            anomaly_scores.shape[0],
            context_length,
            device=anomaly_scores.device
        )
        anomaly_scores = torch.cat([padding, anomaly_scores], dim=1)
        
        return anomaly_scores
    
    def _calculate_anomaly_scores(self, error: torch.Tensor) -> torch.Tensor:
        """
        Calculate anomaly scores using statistical threshold approach from the notebook
        
        Approach from granitetimeseries.ipynb:
        - threshold = mean(error) + 2 * std(error)
        - anomaly_score = error / threshold (normalized)
        
        Args:
            error: Error tensor (absolute difference between actual and predicted)
            
        Returns:
            Anomaly scores tensor [0, 1]
        """
        # Calculate mean and std across all dimensions except batch
        error_flat = error.view(error.shape[0], -1)  # (batch, seq_len * channels)
        
        mean_error = torch.mean(error_flat, dim=1, keepdim=True)  # (batch, 1)
        std_error = torch.std(error_flat, dim=1, keepdim=True)    # (batch, 1)
        
        # Statistical threshold: mean + 2*std (as in notebook)
        threshold = mean_error + 2 * std_error
        threshold = torch.clamp(threshold, min=1e-6)  # Avoid division by zero
        
        # Normalize error by threshold
        # Reshape threshold to match error shape
        threshold = threshold.view(error.shape[0], 1, 1)
        
        # Calculate normalized scores
        scores = error / threshold
        
        # Clip to [0, 1] range using sigmoid for smooth transition
        scores = torch.sigmoid(scores - 1.0)  # Center at threshold=1
        
        return scores
    
    def _normalize_scores(self, error: torch.Tensor) -> torch.Tensor:
        """
        Normalize error to anomaly score [0, 1] (legacy method)
        
        Args:
            error: Error tensor
            
        Returns:
            Normalized anomaly scores
        """
        # Use sigmoid with scaling for normalization
        # Higher error -> higher anomaly score
        scale_factor = 10.0  # Tunable parameter
        scores = torch.sigmoid(error * scale_factor)
        
        return scores
    
    def _prepare_input(self, values: np.ndarray) -> torch.Tensor:
        """
        Prepare input tensor from normalized values
        
        Args:
            values: Normalized time series values
            
        Returns:
            Input tensor ready for model (batch_size, sequence_length, num_channels)
        """
        # Convert to tensor
        tensor = torch.tensor(values, dtype=torch.float32)
        
        # Ensure 3D shape: (batch_size, sequence_length, num_channels)
        if tensor.dim() == 1:
            # Shape: (seq_len,) -> (1, seq_len, 1)
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        elif tensor.dim() == 2:
            # Shape: (batch, seq_len) -> (batch, seq_len, 1)
            tensor = tensor.unsqueeze(-1)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _generate_explanation(
        self,
        scores: np.ndarray,
        is_anomaly: np.ndarray,
        mode: str
    ) -> str:
        """Generate human-readable explanation"""
        
        flat_scores = np.asarray(scores).reshape(-1)
        flat_is_anomaly = np.asarray(is_anomaly).reshape(-1)
        
        num_anomalies = int(np.sum(flat_is_anomaly))
        max_score = float(np.max(flat_scores)) if flat_scores.size > 0 else 0.0
        
        if num_anomalies == 0:
            return "No anomalies detected"
        
        total_points = max(len(flat_scores), 1)
        anomaly_pct = (num_anomalies / total_points) * 100
        
        point_label = "point" if num_anomalies == 1 else "points"
        explanation = (
            f"Detected {num_anomalies}/{total_points} anomalous {point_label} "
            f"({anomaly_pct:.1f}%) using {mode} method. "
            f"Maximum anomaly score: {max_score:.3f}"
        )
        
        return explanation
    
    def _create_error_result(self, series_id: str) -> Dict:
        """Create default result for error cases"""
        return {
            'series_id': series_id,
            'anomaly_scores': [],
            'is_anomaly': [],
            'overall_score': 0.0,
            'explanation': 'Error during anomaly detection',
            'predicted_value': 0.0,
            'actual_value': 0.0,
            'error_value': 0.0,
        }
    
    def forecast_batch(
        self,
        time_series_list: List[Dict],
        forecast_horizon: int = 96,
        include_confidence: bool = True
    ) -> List[Dict]:
        """
        Generate forecasts for a batch of time series
        
        Args:
            time_series_list: List of preprocessed time series
            forecast_horizon: Number of steps to forecast
            include_confidence: Whether to include confidence intervals
            
        Returns:
            List of forecast results
        """
        results = []
        
        logger.info(f"Forecasting {len(time_series_list)} series for {forecast_horizon} steps")
        
        for ts in time_series_list:
            try:
                result = self._forecast_single(
                    ts,
                    forecast_horizon,
                    include_confidence
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to forecast series {ts['series_id']}: {str(e)}")
                results.append(self._create_error_forecast_result(ts['series_id']))
        
        return results
    
    def _forecast_single(
        self,
        ts: Dict,
        forecast_horizon: int,
        include_confidence: bool
    ) -> Dict:
        """Generate forecast for a single time series"""
        
        # Prepare input
        input_tensor = self._prepare_input(ts['normalized_values'])
        
        with torch.no_grad():
            # Generate forecast
            try:
                forecast = self.model.generate(
                    input_tensor,
                    max_length=forecast_horizon
                )
            except:
                # Fallback: use model output
                forecast = self.model(input_tensor)
                forecast = forecast[:, :forecast_horizon]
        
        # Convert to numpy and denormalize
        forecast_np = forecast.cpu().numpy().flatten()
        forecast_denorm = self._denormalize(forecast_np, ts)
        
        # Generate timestamps
        last_timestamp = ts['timestamps'][-1]
        interval = ts['timestamps'][-1] - ts['timestamps'][-2] if len(ts['timestamps']) > 1 else 60e9  # 1 minute default
        forecast_timestamps = [
            last_timestamp + (i + 1) * interval
            for i in range(len(forecast_denorm))
        ]
        
        result = {
            'series_id': ts['series_id'],
            'forecast_values': forecast_denorm.tolist(),
            'forecast_timestamps': forecast_timestamps
        }
        
        # Add confidence intervals if requested
        if include_confidence:
            # Simple confidence intervals (can be improved with quantile regression)
            std = np.std(ts['normalized_values'])
            lower = forecast_denorm - 1.96 * std
            upper = forecast_denorm + 1.96 * std
            
            result['confidence'] = {
                'lower': lower.tolist(),
                'upper': upper.tolist()
            }
        
        return result
    
    def _denormalize(self, values: np.ndarray, ts: Dict) -> np.ndarray:
        """Denormalize values using original statistics"""
        mean = ts.get('mean', 0.0)
        std = ts.get('std', 1.0)
        return values * std + mean
    
    def _create_error_forecast_result(self, series_id: str) -> Dict:
        """Create default forecast result for error cases"""
        return {
            'series_id': series_id,
            'forecast_values': [],
            'forecast_timestamps': []
        }

# Made with Bob
