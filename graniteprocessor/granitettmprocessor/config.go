// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package granitettmprocessor

import (
	"errors"
	"fmt"
	"time"

	"go.opentelemetry.io/collector/component"
)

// Config represents the configuration for the Granite TTM-R2 processor
// This processor replaces the Isolation Forest algorithm with IBM's Granite TTM-R2
// time series foundation model for more accurate anomaly detection.
type Config struct {
	// TTM-R2 Service Connection
	Endpoint string    `mapstructure:"endpoint"`
	Timeout  string    `mapstructure:"timeout"`
	TLS      TLSConfig `mapstructure:"tls"`

	// Time Series Configuration
	// ContextLength: Number of historical data points the model uses (TTM-R2 default: 512)
	ContextLength int `mapstructure:"context_length"`
	// ForecastHorizon: Number of future points to predict (TTM-R2 default: 96)
	ForecastHorizon int `mapstructure:"forecast_horizon"`
	// BufferSize: Maximum number of data points to buffer per time series
	BufferSize int `mapstructure:"buffer_size"`
	// FlushInterval: How often to send buffered data to TTM service
	FlushInterval string `mapstructure:"flush_interval"`

	// ForecastSelectors restricts which metrics receive forecast enrichment.
	// ALL metrics are always anomaly-detected regardless of this list.
	// An empty list means all metrics are also forecast-eligible.
	ForecastSelectors []MetricSelector `mapstructure:"forecast_selectors"`

	// Anomaly Detection
	// AnomalyThreshold: Score above which a point is classified as anomalous (0-1)
	AnomalyThreshold float64 `mapstructure:"anomaly_threshold"`
	// DetectionMode: "reconstruction" (compare actual vs reconstructed), "forecast" (compare actual vs predicted), or "both"
	DetectionMode string `mapstructure:"detection_mode"`
	// Sensitivity: "low", "medium", or "high" - affects threshold adjustment
	Sensitivity string `mapstructure:"sensitivity"`

	// Output Configuration
	// Mode: "enrich" (add attributes), "filter" (drop normal data), or "both"
	Mode                    string `mapstructure:"mode"`
	ScoreAttribute          string `mapstructure:"score_attribute"`
	ClassificationAttribute string `mapstructure:"classification_attribute"`
	AddForecastMetrics      bool   `mapstructure:"add_forecast_metrics"`
	// EnableForecasting: when true, the processor calls the Forecast RPC after
	// anomaly detection and enriches each datapoint with the forecast_five_mins
	// summary attributes (forecast_risk_level, forecast_risk_score, etc.).
	// Series must have at least ContextLength samples before forecasting begins;
	// during cold-start only anomaly enrichment runs.
	EnableForecasting bool `mapstructure:"enable_forecasting"`

	// Performance
	MaxConcurrentRequests int  `mapstructure:"max_concurrent_requests"`
	BatchSize             int  `mapstructure:"batch_size"`
	EnableCaching         bool `mapstructure:"enable_caching"`
}

// TLSConfig contains TLS configuration for gRPC connection
type TLSConfig struct {
	Insecure bool   `mapstructure:"insecure"`
	CertFile string `mapstructure:"cert_file"`
	KeyFile  string `mapstructure:"key_file"`
	CAFile   string `mapstructure:"ca_file"`
}

// MetricSelector defines which metrics to process
type MetricSelector struct {
	NamePattern        string   `mapstructure:"name_pattern"`
	ResourceAttributes []string `mapstructure:"resource_attributes"`
}

// createDefaultConfig returns a configuration with sensible defaults for TTM-R2
func createDefaultConfig() component.Config {
	return &Config{
		// Connection defaults
		Endpoint: "localhost:50051",
		Timeout:  "5s",
		TLS: TLSConfig{
			Insecure: true,
		},

		// Time series defaults optimized for TTM-R2
		ContextLength:   512,  // TTM-R2 optimal context window
		ForecastHorizon: 96,   // TTM-R2 optimal forecast horizon
		BufferSize:      1024, // 2x context length for safety
		FlushInterval:   "5m",

		// ForecastSelectors defaults — the 6 confirmed container metrics.
		// Unknown metrics arriving outside this list are still anomaly-detected;
		// they just don't get forecast_five_mins attributes.
		// Set to [] to forecast all metrics.
		ForecastSelectors: []MetricSelector{
			{NamePattern: "container_cpu_usage_seconds_total"},
			{NamePattern: "container_memory_working_set_bytes"},
			{NamePattern: "container_memory_usage_bytes"},
			{NamePattern: "container_fs_usage_bytes"},
			{NamePattern: "container_network_receive_bytes_total"},
			{NamePattern: "container_network_transmit_bytes_total"},
		},

		// Anomaly detection defaults
		AnomalyThreshold: 0.7,
		DetectionMode:    "reconstruction", // Reconstruction is faster than forecast
		Sensitivity:      "medium",

		// Output defaults
		Mode:                    "enrich",
		ScoreAttribute:          "ttm.anomaly_score",
		ClassificationAttribute: "ttm.is_anomaly",
		AddForecastMetrics:      false,

		// Forecasting defaults
		EnableForecasting: true, // enabled by default; set to false to disable 5-minute forecast enrichment

		// Performance defaults
		MaxConcurrentRequests: 4,
		BatchSize:             32,
		EnableCaching:         true,
	}
}

// Validate checks the configuration for logical consistency and valid parameter ranges
func (cfg *Config) Validate() error {
	if cfg.Endpoint == "" {
		return errors.New("endpoint must be specified")
	}

	if _, err := time.ParseDuration(cfg.Timeout); err != nil {
		return fmt.Errorf("timeout is not a valid duration: %w", err)
	}

	if cfg.ContextLength <= 0 {
		return errors.New("context_length must be positive")
	}

	// TTM-R2 works best with context lengths that are powers of 2
	if cfg.ContextLength < 64 {
		return errors.New("context_length should be at least 64 for TTM-R2")
	}

	if cfg.ForecastHorizon <= 0 {
		return errors.New("forecast_horizon must be positive")
	}

	if cfg.BufferSize <= 0 {
		return errors.New("buffer_size must be positive")
	}

	if cfg.BufferSize < cfg.ContextLength {
		return fmt.Errorf("buffer_size (%d) should be >= context_length (%d)", cfg.BufferSize, cfg.ContextLength)
	}

	if _, err := time.ParseDuration(cfg.FlushInterval); err != nil {
		return fmt.Errorf("flush_interval is not a valid duration: %w", err)
	}

	if cfg.AnomalyThreshold < 0.0 || cfg.AnomalyThreshold > 1.0 {
		return errors.New("anomaly_threshold must be between 0.0 and 1.0")
	}

	validModes := map[string]bool{
		"reconstruction": true,
		"forecast":       true,
		"both":           true,
	}
	if !validModes[cfg.DetectionMode] {
		return fmt.Errorf("detection_mode must be one of: reconstruction, forecast, both")
	}

	validSensitivity := map[string]bool{
		"low":    true,
		"medium": true,
		"high":   true,
	}
	if !validSensitivity[cfg.Sensitivity] {
		return fmt.Errorf("sensitivity must be one of: low, medium, high")
	}

	validOutputModes := map[string]bool{
		"enrich": true,
		"filter": true,
		"both":   true,
	}
	if !validOutputModes[cfg.Mode] {
		return fmt.Errorf("mode must be one of: enrich, filter, both")
	}

	if cfg.ScoreAttribute == "" {
		return errors.New("score_attribute must be specified")
	}

	if cfg.ClassificationAttribute == "" {
		return errors.New("classification_attribute must be specified")
	}

	if cfg.ScoreAttribute == cfg.ClassificationAttribute {
		return errors.New("score_attribute and classification_attribute must be different")
	}

	if cfg.MaxConcurrentRequests <= 0 {
		return errors.New("max_concurrent_requests must be positive")
	}

	if cfg.BatchSize <= 0 {
		return errors.New("batch_size must be positive")
	}

	// Validate TLS configuration
	if !cfg.TLS.Insecure {
		if cfg.TLS.CertFile == "" || cfg.TLS.KeyFile == "" {
			return errors.New("cert_file and key_file must be specified when TLS is enabled")
		}
	}

	return nil
}

// GetTimeoutDuration returns the timeout as a time.Duration
func (cfg *Config) GetTimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(cfg.Timeout)
}

// GetFlushIntervalDuration returns the flush interval as a time.Duration
func (cfg *Config) GetFlushIntervalDuration() (time.Duration, error) {
	return time.ParseDuration(cfg.FlushInterval)
}

// ShouldForecastMetric returns true when the metric name matches the configured
// ForecastSelectors. An empty selector list means all metrics are forecast-eligible.
// This is intentionally separate from anomaly detection — all metrics are always
// anomaly-detected; only this list controls which ones also get forecasted.
func (cfg *Config) ShouldForecastMetric(metricName string) bool {
	if len(cfg.ForecastSelectors) == 0 {
		return true
	}
	for _, selector := range cfg.ForecastSelectors {
		if cfg.matchesSelector(metricName, nil, selector) {
			return true
		}
	}
	return false
}

// matchesSelector checks if a metric matches a specific selector
func (cfg *Config) matchesSelector(metricName string, resourceAttrs map[string]string, selector MetricSelector) bool {
	// Check name pattern (simple prefix match for now, can be enhanced with regex)
	if selector.NamePattern != "" {
		// Simple contains check - can be enhanced with regex matching
		matched := false
		if selector.NamePattern == "*" {
			matched = true
		} else {
			// Simple prefix/suffix/contains matching
			matched = contains(metricName, selector.NamePattern)
		}
		if !matched {
			return false
		}
	}

	// Check resource attributes
	if len(selector.ResourceAttributes) > 0 {
		for _, requiredAttr := range selector.ResourceAttributes {
			if _, exists := resourceAttrs[requiredAttr]; !exists {
				return false
			}
		}
	}

	return true
}

// contains is a simple string matching helper
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		s[:len(substr)] == substr ||
		s[len(s)-len(substr):] == substr)
}

// Made with Bob
