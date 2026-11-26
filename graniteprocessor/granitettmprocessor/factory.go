// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package granitettmprocessor

import (
	"context"
	"errors"
	"fmt"

	"go.opentelemetry.io/collector/component"
	"go.opentelemetry.io/collector/consumer"
	"go.opentelemetry.io/collector/pdata/pmetric"
	"go.opentelemetry.io/collector/processor"
	"go.uber.org/zap"
)

var (
	typeStr   = component.MustNewType("granitettm")
	stability = component.StabilityLevelAlpha
)

// NewFactory creates a new processor factory for Granite TTM-R2
// This replaces the Isolation Forest processor with IBM's Granite TTM-R2 model
func NewFactory() processor.Factory {
	return processor.NewFactory(
		typeStr,
		createDefaultConfig,
		processor.WithMetrics(createMetricsProcessor, stability),
	)
}

// createMetricsProcessor creates a metrics processor instance
func createMetricsProcessor(
	_ context.Context,
	set processor.Settings,
	cfg component.Config,
	nextConsumer consumer.Metrics,
) (processor.Metrics, error) {
	processorCfg, ok := cfg.(*Config)
	if !ok {
		return nil, errors.New("configuration is not of type *Config")
	}

	if err := processorCfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	set.Logger.Info("Creating Granite TTM-R2 metrics processor",
		zap.String("processor_id", set.ID.String()),
		zap.String("endpoint", processorCfg.Endpoint),
		zap.Int("context_length", processorCfg.ContextLength),
		zap.Int("forecast_horizon", processorCfg.ForecastHorizon),
		zap.String("detection_mode", processorCfg.DetectionMode),
		zap.String("mode", processorCfg.Mode),
	)

	proc, err := newGraniteTTMProcessor(processorCfg, set.Logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create processor: %w", err)
	}

	return &metricsProcessor{
		graniteTTMProcessor: proc,
		nextConsumer:        nextConsumer,
		logger:              set.Logger,
	}, nil
}

// metricsProcessor wraps the core processor for metrics pipeline
type metricsProcessor struct {
	*graniteTTMProcessor
	nextConsumer consumer.Metrics
	logger       *zap.Logger
}

// Start initializes the processor
func (mp *metricsProcessor) Start(ctx context.Context, host component.Host) error {
	return mp.graniteTTMProcessor.Start(ctx, host)
}

// Shutdown gracefully stops the processor
func (mp *metricsProcessor) Shutdown(ctx context.Context) error {
	return mp.graniteTTMProcessor.Shutdown(ctx)
}

// ConsumeMetrics processes metrics through the TTM-R2 model
func (mp *metricsProcessor) ConsumeMetrics(ctx context.Context, md pmetric.Metrics) error {
	processedMetrics, err := mp.processMetrics(ctx, md)
	if err != nil {
		return err
	}
	return mp.nextConsumer.ConsumeMetrics(ctx, processedMetrics)
}

// Capabilities returns the processor capabilities
func (*metricsProcessor) Capabilities() consumer.Capabilities {
	return consumer.Capabilities{MutatesData: true}
}
