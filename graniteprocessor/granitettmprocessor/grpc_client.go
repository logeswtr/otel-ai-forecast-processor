// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package granitettmprocessor

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"
	"time"

	"github.com/open-telemetry/opentelemetry-collector-contrib/processor/granitettmprocessor/internal/ttmproto"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// TTMClient wraps the gRPC client for TTM-R2 inference service
// This replaces the in-process Isolation Forest with external TTM-R2 service calls
type TTMClient struct {
	conn   *grpc.ClientConn
	client ttmproto.TTMServiceClient
	config *Config
	logger *zap.Logger
}

// NewTTMClient creates a new TTM-R2 gRPC client
func NewTTMClient(config *Config, logger *zap.Logger) (*TTMClient, error) {
	// Prepare dial options
	var opts []grpc.DialOption

	// Configure TLS
	if config.TLS.Insecure {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		tlsConfig, err := loadTLSConfig(&config.TLS)
		if err != nil {
			return nil, fmt.Errorf("failed to load TLS config: %w", err)
		}
		opts = append(opts, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))
	}

	// Add timeout
	timeout, err := config.GetTimeoutDuration()
	if err != nil {
		return nil, fmt.Errorf("invalid timeout: %w", err)
	}
	opts = append(opts, grpc.WithTimeout(timeout))

	// Establish connection
	conn, err := grpc.Dial(config.Endpoint, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to TTM-R2 service at %s: %w", config.Endpoint, err)
	}

	client := ttmproto.NewTTMServiceClient(conn)

	logger.Info("Connected to TTM-R2 service",
		zap.String("endpoint", config.Endpoint),
		zap.Bool("tls_enabled", !config.TLS.Insecure),
		zap.String("model", "IBM Granite TTM-R2"),
	)

	return &TTMClient{
		conn:   conn,
		client: client,
		config: config,
		logger: logger,
	}, nil
}

// DetectAnomalies sends a batch of time series for anomaly detection using TTM-R2
// This is the core method that replaces Isolation Forest's ProcessSample
func (c *TTMClient) DetectAnomalies(ctx context.Context, batch *TimeSeriesBatch) (*ttmproto.AnomalyResponse, error) {
	// Convert batch to protobuf request
	request := c.buildAnomalyRequest(batch)

	// Add timeout to context
	timeout, _ := c.config.GetTimeoutDuration()
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Call TTM-R2 service
	startTime := time.Now()
	response, err := c.client.DetectAnomalies(ctx, request)
	duration := time.Since(startTime)

	if err != nil {
		c.logger.Error("Failed to detect anomalies with TTM-R2",
			zap.Error(err),
			zap.String("batch_id", batch.BatchID),
			zap.Duration("duration", duration),
		)
		return nil, fmt.Errorf("TTM-R2 anomaly detection failed: %w", err)
	}

	c.logger.Debug("TTM-R2 anomaly detection completed",
		zap.String("batch_id", batch.BatchID),
		zap.Int("series_count", len(batch.Series)),
		zap.Duration("duration", duration),
		zap.Int64("processing_time_ms", response.Metadata.ProcessingTimeMs),
		zap.String("model_version", response.Metadata.ModelVersion),
	)

	return response, nil
}

// Forecast sends a batch of time series for forecasting using TTM-R2
func (c *TTMClient) Forecast(ctx context.Context, batch *TimeSeriesBatch) (*ttmproto.ForecastResponse, error) {
	// Convert batch to protobuf request
	request := c.buildForecastRequest(batch)

	// Add timeout to context
	timeout, _ := c.config.GetTimeoutDuration()
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Call TTM-R2 service
	startTime := time.Now()
	response, err := c.client.Forecast(ctx, request)
	duration := time.Since(startTime)

	if err != nil {
		c.logger.Error("Failed to forecast with TTM-R2",
			zap.Error(err),
			zap.String("batch_id", batch.BatchID),
			zap.Duration("duration", duration),
		)
		return nil, fmt.Errorf("TTM-R2 forecast failed: %w", err)
	}

	c.logger.Debug("TTM-R2 forecast completed",
		zap.String("batch_id", batch.BatchID),
		zap.Int("series_count", len(batch.Series)),
		zap.Duration("duration", duration),
	)

	return response, nil
}

// HealthCheck checks if the TTM-R2 service is healthy
func (c *TTMClient) HealthCheck(ctx context.Context) (*ttmproto.HealthResponse, error) {
	request := &ttmproto.HealthRequest{
		Detailed: true,
	}

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	response, err := c.client.HealthCheck(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("TTM-R2 health check failed: %w", err)
	}

	return response, nil
}

// Close closes the gRPC connection
func (c *TTMClient) Close() error {
	if c.conn != nil {
		c.logger.Info("Closing TTM-R2 client connection")
		return c.conn.Close()
	}
	return nil
}

// buildAnomalyRequest converts a batch to protobuf anomaly request for TTM-R2
func (c *TTMClient) buildAnomalyRequest(batch *TimeSeriesBatch) *ttmproto.AnomalyRequest {
	series := make([]*ttmproto.TimeSeries, len(batch.Series))

	for i, ts := range batch.Series {
		timestamps, values := ts.GetData()

		// Convert timestamps to Unix nanoseconds
		timestampsNano := make([]int64, len(timestamps))
		for j, t := range timestamps {
			timestampsNano[j] = t.UnixNano()
		}

		series[i] = &ttmproto.TimeSeries{
			MetricName: ts.MetricName,
			Values:     values,
			Timestamps: timestampsNano,
			Attributes: ts.Attributes,
			SeriesId:   ts.SeriesID,
		}
	}

	return &ttmproto.AnomalyRequest{
		Series: series,
		Config: &ttmproto.DetectionConfig{
			ContextLength:   int32(c.config.ContextLength),
			ForecastHorizon: int32(c.config.ForecastHorizon),
			DetectionMode:   c.config.DetectionMode,
			Threshold:       float32(c.config.AnomalyThreshold),
			Sensitivity:     c.config.Sensitivity,
		},
	}
}

// buildForecastRequest converts a batch to protobuf forecast request for TTM-R2
func (c *TTMClient) buildForecastRequest(batch *TimeSeriesBatch) *ttmproto.ForecastRequest {
	series := make([]*ttmproto.TimeSeries, len(batch.Series))

	for i, ts := range batch.Series {
		timestamps, values := ts.GetData()

		// Convert timestamps to Unix nanoseconds
		timestampsNano := make([]int64, len(timestamps))
		for j, t := range timestamps {
			timestampsNano[j] = t.UnixNano()
		}

		series[i] = &ttmproto.TimeSeries{
			MetricName: ts.MetricName,
			Values:     values,
			Timestamps: timestampsNano,
			Attributes: ts.Attributes,
			SeriesId:   ts.SeriesID,
		}
	}

	return &ttmproto.ForecastRequest{
		Series: series,
		Config: &ttmproto.ForecastConfig{
			ContextLength:              int32(c.config.ContextLength),
			ForecastHorizon:            int32(c.config.ForecastHorizon),
			IncludeConfidenceIntervals: true,
		},
	}
}

// loadTLSConfig loads TLS configuration from files
func loadTLSConfig(tlsConfig *TLSConfig) (*tls.Config, error) {
	// Load client cert and key
	cert, err := tls.LoadX509KeyPair(tlsConfig.CertFile, tlsConfig.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load client cert/key: %w", err)
	}

	// Load CA cert if provided
	var caCertPool *x509.CertPool
	if tlsConfig.CAFile != "" {
		caCert, err := os.ReadFile(tlsConfig.CAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA cert: %w", err)
		}

		caCertPool = x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse CA cert")
		}
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      caCertPool,
		MinVersion:   tls.VersionTLS12,
	}, nil
}

// Made with Bob
