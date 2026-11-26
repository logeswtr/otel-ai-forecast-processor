#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
TTM-R2 Inference Service
Provides gRPC API for time series anomaly detection using IBM Granite TTM-R2 model
"""

import grpc
from grpc_reflection.v1alpha import reflection
from concurrent import futures
import logging
import time
import sys
from typing import List, Dict

# Import generated protobuf code
import ttm_service_pb2
import ttm_service_pb2_grpc

# Import our modules
from model_loader import TTMModelLoader
from anomaly_detector import AnomalyDetector
from preprocessor import TimeSeriesPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/ttm_service_debug.log"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


class TTMServiceServicer(ttm_service_pb2_grpc.TTMServiceServicer):
    """Implementation of TTM gRPC service"""
    
    def __init__(self, model_path: str = "ibm/TTM"):
        """
        Initialize the service with TTM-R2 model
        
        Args:
            model_path: Path or HuggingFace model ID for TTM-R2
        """
        logger.info(f"Initializing TTM service with model: {model_path}")
        
        # Load model
        self.model_loader = TTMModelLoader(model_path)
        self.model = self.model_loader.load()
        
        # Initialize components
        self.detector = AnomalyDetector(self.model)
        self.preprocessor = TimeSeriesPreprocessor()
        
        # Service statistics
        self.start_time = time.time()
        self.total_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        
        logger.info("TTM service initialized successfully")
    
    def DetectAnomalies(self, request, context):
        """
        Detect anomalies in time series data
        
        Args:
            request: AnomalyRequest containing time series data
            context: gRPC context
            
        Returns:
            AnomalyResponse with detection results
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            logger.info(
                "Processing anomaly detection request with %s series, context_length=%s forecast_horizon=%s detection_mode=%s threshold=%s sensitivity=%s",
                len(request.series),
                request.config.context_length,
                request.config.forecast_horizon,
                request.config.detection_mode,
                request.config.threshold,
                request.config.sensitivity,
            )
            
            # Parse and preprocess time series
            time_series_list = self._parse_time_series(request.series)
            processed_series = self.preprocessor.preprocess_batch(
                time_series_list,
                context_length=request.config.context_length
            )
            logger.info(
                "Preprocessed %s/%s series for anomaly detection",
                len(processed_series),
                len(time_series_list),
            )
            
            # Run anomaly detection
            results = self.detector.detect_batch(
                processed_series,
                threshold=request.config.threshold,
                detection_mode=request.config.detection_mode,
                sensitivity=request.config.sensitivity
            )
            
            # Format response
            response = self._format_anomaly_response(results, start_time)
            
            processing_time = (time.time() - start_time) * 1000
            self.total_latency += processing_time
            
            logger.info(f"Anomaly detection completed in {processing_time:.2f}ms")
            
            return response
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Anomaly detection failed: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Anomaly detection failed: {str(e)}")
            return ttm_service_pb2.AnomalyResponse()
    
    def Forecast(self, request, context):
        """
        Generate forecasts for time series
        
        Args:
            request: ForecastRequest containing time series data
            context: gRPC context
            
        Returns:
            ForecastResponse with forecast results
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            logger.info(f"Processing forecast request with {len(request.series)} series")
            
            # Parse and preprocess time series
            time_series_list = self._parse_time_series(request.series)
            processed_series = self.preprocessor.preprocess_batch(
                time_series_list,
                context_length=request.config.context_length
            )
            
            # Run forecasting
            results = self.detector.forecast_batch(
                processed_series,
                forecast_horizon=request.config.forecast_horizon,
                include_confidence=request.config.include_confidence_intervals
            )
            
            # Format response
            response = self._format_forecast_response(results, start_time)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Forecast completed in {processing_time:.2f}ms")
            
            return response
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Forecast failed: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Forecast failed: {str(e)}")
            return ttm_service_pb2.ForecastResponse()
    
    def HealthCheck(self, request, context):
        """
        Check service health
        
        Args:
            request: HealthRequest
            context: gRPC context
            
        Returns:
            HealthResponse with service status
        """
        uptime = int(time.time() - self.start_time)
        avg_latency = self.total_latency / max(self.total_requests, 1)
        
        # Determine health status
        if self.failed_requests / max(self.total_requests, 1) > 0.1:
            status = "degraded"
        else:
            status = "healthy"
        
        response = ttm_service_pb2.HealthResponse(
            status=status,
            model_version=self.model_loader.get_model_version(),
            uptime_seconds=uptime
        )
        
        if request.detailed:
            response.metrics.CopyFrom(ttm_service_pb2.HealthMetrics(
                total_requests=self.total_requests,
                failed_requests=self.failed_requests,
                avg_latency_ms=avg_latency,
                model_memory_mb=self.model_loader.get_memory_usage()
            ))
        
        return response
    
    def _parse_time_series(self, series_list) -> List[Dict]:
        """Parse protobuf time series to internal format"""
        result = []
        for series in series_list:
            result.append({
                'metric_name': series.metric_name,
                'series_id': series.series_id,
                'values': list(series.values),
                'timestamps': list(series.timestamps),
                'attributes': dict(series.attributes)
            })
        return result
    
    def _format_anomaly_response(self, results: List[Dict], start_time: float):
        """Format anomaly detection results as protobuf response"""
        response = ttm_service_pb2.AnomalyResponse()
        
        for result in results:
            anomaly_result = response.results.add()
            anomaly_result.series_id = result['series_id']
            anomaly_result.anomaly_scores.extend(result['anomaly_scores'])
            anomaly_result.is_anomaly.extend(result['is_anomaly'])
            anomaly_result.overall_score = result['overall_score']
            anomaly_result.explanation = result.get('explanation', '')
            
            # Add reconstruction data if available
            if 'reconstruction' in result:
                anomaly_result.reconstruction.reconstructed_values.extend(
                    result['reconstruction']['values']
                )
                anomaly_result.reconstruction.reconstruction_errors.extend(
                    result['reconstruction']['errors']
                )
        
        # Add metadata
        processing_time = (time.time() - start_time) * 1000
        response.metadata.processing_time_ms = int(processing_time)
        response.metadata.model_version = self.model_loader.get_model_version()
        response.metadata.batch_size = len(results)
        
        return response
    
    def _format_forecast_response(self, results: List[Dict], start_time: float):
        """Format forecast results as protobuf response"""
        response = ttm_service_pb2.ForecastResponse()
        
        for result in results:
            forecast_result = response.results.add()
            forecast_result.series_id = result['series_id']
            forecast_result.forecast_values.extend(result['forecast_values'])
            forecast_result.forecast_timestamps.extend(result['forecast_timestamps'])
            
            # Add confidence intervals if available
            if 'confidence' in result:
                forecast_result.confidence.lower_bound.extend(
                    result['confidence']['lower']
                )
                forecast_result.confidence.upper_bound.extend(
                    result['confidence']['upper']
                )
        
        # Add metadata
        processing_time = (time.time() - start_time) * 1000
        response.metadata.processing_time_ms = int(processing_time)
        response.metadata.model_version = self.model_loader.get_model_version()
        response.metadata.batch_size = len(results)
        
        return response


def serve(port: int = 50051, max_workers: int = 10):
    """
    Start the gRPC server
    
    Args:
        port: Port to listen on
        max_workers: Maximum number of worker threads
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    # Add servicer
    ttm_service_pb2_grpc.add_TTMServiceServicer_to_server(
        TTMServiceServicer(),
        server
    )
    
    # Enable reflection for grpcurl and other tools
    SERVICE_NAMES = (
        ttm_service_pb2.DESCRIPTOR.services_by_name['TTMService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Start server
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"TTM-R2 Inference Service started on port {port}")
    logger.info(f"Ready to accept requests...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(grace=5)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TTM-R2 Inference Service')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    parser.add_argument('--model', type=str, default='ibm/TTM',
                       help='Model path or HuggingFace ID')
    
    args = parser.parse_args()
    
    serve(port=args.port, max_workers=args.workers)

# Made with Bob
