// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package granitettmprocessor

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/open-telemetry/opentelemetry-collector-contrib/processor/granitettmprocessor/internal/ttmproto"
	"go.opentelemetry.io/collector/component"
	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/pmetric"
	"go.uber.org/zap"
)

// graniteTTMProcessor is the main processor that coordinates anomaly detection
// This replaces the Isolation Forest processor with IBM Granite TTM-R2 model
type graniteTTMProcessor struct {
	config *Config
	logger *zap.Logger

	// Core components - TTM-R2 replaces Isolation Forest
	buffer    *MetricsBuffer
	ttmClient *TTMClient

	// Anomaly result cache for enrichment
	resultCache map[string]*CachedResult
	cacheMutex  sync.RWMutex
	cacheMaxAge time.Duration

	// Forecast result cache for enrichment (parallel to resultCache)
	forecastCache      map[string]*CachedForecastResult
	forecastCacheMutex sync.RWMutex

	// Background processing
	flushTicker   *time.Ticker
	cleanupTicker *time.Ticker
	stopChan      chan struct{}
	wg            sync.WaitGroup

	// Statistics
	processedMetrics  uint64
	anomaliesDetected uint64
	statsMutex        sync.Mutex
}

// CachedResult stores recent anomaly detection results from TTM-R2
type CachedResult struct {
	SeriesID       string
	AnomalyScore   float64
	IsAnomaly      bool
	PredictedValue float64
	ActualValue    float64
	Error          float64
	Timestamp      time.Time
	Explanation    string
}

// CachedForecastResult stores the compact 5-minute forecast summary for a series.
// All interesting data points are retained so Grafana can surface what is needed
// without reprocessing. Only scalars are stored — not the full future series array.
type CachedForecastResult struct {
	SeriesID           string
	RiskLevel          string  // "low" | "medium" | "high" | "critical"
	RiskScore          float64 // 0–1
	PredictedIssue     bool
	PredictedPeakValue float64
	PredictedPeakTime  int64   // Unix milliseconds
	ConfidenceLower    float64 // 95% CI lower bound at horizon end
	ConfidenceUpper    float64 // 95% CI upper bound at horizon end
	Reason             string
	Timestamp          time.Time
}

// newGraniteTTMProcessor creates a new processor instance
// This is the constructor that replaces Isolation Forest initialization
func newGraniteTTMProcessor(config *Config, logger *zap.Logger) (*graniteTTMProcessor, error) {
	// Create metrics buffer for time-series accumulation
	buffer := NewMetricsBuffer(config.ContextLength, config.BufferSize, logger)

	// Create TTM-R2 client (replaces Isolation Forest model)
	ttmClient, err := NewTTMClient(config, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create TTM-R2 client: %w", err)
	}

	// Test connection with health check
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	healthResp, err := ttmClient.HealthCheck(ctx)
	if err != nil {
		logger.Warn("TTM-R2 service health check failed, but continuing", zap.Error(err))
	} else {
		logger.Info("TTM-R2 service is healthy",
			zap.String("status", healthResp.Status),
			zap.String("model_version", healthResp.ModelVersion),
			zap.Int64("uptime_seconds", healthResp.UptimeSeconds),
		)
	}

	flushInterval, _ := config.GetFlushIntervalDuration()

	processor := &graniteTTMProcessor{
		config:        config,
		logger:        logger,
		buffer:        buffer,
		ttmClient:     ttmClient,
		resultCache:   make(map[string]*CachedResult),
		forecastCache: make(map[string]*CachedForecastResult),
		cacheMaxAge:   5 * time.Minute,
		flushTicker:   time.NewTicker(flushInterval),
		cleanupTicker: time.NewTicker(1 * time.Minute),
		stopChan:      make(chan struct{}),
	}

	return processor, nil
}

// Start initializes the processor and starts background tasks
func (p *graniteTTMProcessor) Start(_ context.Context, _ component.Host) error {
	p.logger.Info("Starting Granite TTM-R2 processor",
		zap.String("endpoint", p.config.Endpoint),
		zap.Int("context_length", p.config.ContextLength),
		zap.Int("forecast_horizon", p.config.ForecastHorizon),
		zap.String("detection_mode", p.config.DetectionMode),
		zap.String("mode", p.config.Mode),
	)

	// Start background flush loop
	p.wg.Add(1)
	go p.backgroundFlush()

	// Start cleanup loop
	p.wg.Add(1)
	go p.backgroundCleanup()

	return nil
}

// Shutdown gracefully stops the processor
func (p *graniteTTMProcessor) Shutdown(_ context.Context) error {
	p.logger.Info("Shutting down Granite TTM-R2 processor")

	// Stop tickers
	if p.flushTicker != nil {
		p.flushTicker.Stop()
	}
	if p.cleanupTicker != nil {
		p.cleanupTicker.Stop()
	}

	// Signal background goroutines to stop
	close(p.stopChan)

	// Wait for goroutines to finish
	p.wg.Wait()

	// Close TTM-R2 client
	if p.ttmClient != nil {
		if err := p.ttmClient.Close(); err != nil {
			p.logger.Error("Failed to close TTM-R2 client", zap.Error(err))
		}
	}

	// Log final statistics
	p.statsMutex.Lock()
	p.logger.Info("Processor statistics",
		zap.Uint64("processed_metrics", p.processedMetrics),
		zap.Uint64("anomalies_detected", p.anomaliesDetected),
	)
	p.statsMutex.Unlock()

	return nil
}

// processMetrics processes incoming metrics
// This is the main entry point that replaces Isolation Forest's processMetrics
func (p *graniteTTMProcessor) processMetrics(ctx context.Context, md pmetric.Metrics) (pmetric.Metrics, error) {
	// Add metrics to buffer for time-series accumulation
	if err := p.addMetricsToBuffer(md); err != nil {
		p.logger.Error("Failed to add metrics to buffer", zap.Error(err))
	}

	// Enrich metrics with cached TTM-R2 results if in enrich mode
	if p.config.Mode == "enrich" || p.config.Mode == "both" {
		p.enrichMetricsWithCache(md)
	}

	// Filter metrics if in filter mode
	if p.config.Mode == "filter" || p.config.Mode == "both" {
		md = p.filterAnomalousMetrics(md)
	}

	return md, nil
}

// addMetricsToBuffer extracts metrics and adds them to the time-series buffer.
// ALL metrics are buffered — selector filtering only applies to forecasting, not
// anomaly detection, so unknown metrics still get anomaly scores.
func (p *graniteTTMProcessor) addMetricsToBuffer(md pmetric.Metrics) error {
	for i := 0; i < md.ResourceMetrics().Len(); i++ {
		rm := md.ResourceMetrics().At(i)
		resourceAttrs := extractAttributes(rm.Resource().Attributes())

		for j := 0; j < rm.ScopeMetrics().Len(); j++ {
			sm := rm.ScopeMetrics().At(j)

			for k := 0; k < sm.Metrics().Len(); k++ {
				metric := sm.Metrics().At(k)
				// No selector check here — buffer every metric for anomaly detection.
				p.extractAndBufferMetric(metric, resourceAttrs)
			}
		}
	}

	p.statsMutex.Lock()
	p.processedMetrics++
	p.statsMutex.Unlock()

	return nil
}

// extractAndBufferMetric extracts data points from a metric and adds to buffer
func (p *graniteTTMProcessor) extractAndBufferMetric(metric pmetric.Metric, resourceAttrs map[string]string) {
	switch metric.Type() {
	case pmetric.MetricTypeGauge:
		p.bufferGaugeMetric(metric.Gauge(), metric.Name(), resourceAttrs)
	case pmetric.MetricTypeSum:
		p.bufferSumMetric(metric.Sum(), metric.Name(), resourceAttrs)
	}
}

// bufferGaugeMetric buffers gauge metric data points
func (p *graniteTTMProcessor) bufferGaugeMetric(gauge pmetric.Gauge, metricName string, resourceAttrs map[string]string) {
	for i := 0; i < gauge.DataPoints().Len(); i++ {
		dp := gauge.DataPoints().At(i)
		timestamp := time.Unix(0, int64(dp.Timestamp()))
		value := dp.DoubleValue()

		// Combine resource and data point attributes
		attrs := mergeAttributes(resourceAttrs, extractAttributes(dp.Attributes()))

		if err := p.buffer.AddDataPoint(metricName, timestamp, value, attrs); err != nil {
			p.logger.Debug("Failed to buffer data point", zap.Error(err))
		}
	}
}

// bufferSumMetric buffers sum metric data points
func (p *graniteTTMProcessor) bufferSumMetric(sum pmetric.Sum, metricName string, resourceAttrs map[string]string) {
	for i := 0; i < sum.DataPoints().Len(); i++ {
		dp := sum.DataPoints().At(i)
		timestamp := time.Unix(0, int64(dp.Timestamp()))
		value := dp.DoubleValue()

		attrs := mergeAttributes(resourceAttrs, extractAttributes(dp.Attributes()))

		if err := p.buffer.AddDataPoint(metricName, timestamp, value, attrs); err != nil {
			p.logger.Debug("Failed to buffer data point", zap.Error(err))
		}
	}
}

// backgroundFlush periodically processes buffered metrics with TTM-R2
func (p *graniteTTMProcessor) backgroundFlush() {
	defer p.wg.Done()

	for {
		select {
		case <-p.flushTicker.C:
			p.flushBuffer()
		case <-p.stopChan:
			// Final flush before shutdown
			p.flushBuffer()
			return
		}
	}
}

// flushBuffer processes ready batches from the buffer using TTM-R2
// This replaces Isolation Forest's incremental tree updates
func (p *graniteTTMProcessor) flushBuffer() {
	batches := p.buffer.GetReadyBatches(p.config.BatchSize)

	if len(batches) == 0 {
		return
	}

	p.logger.Debug("Flushing buffer to TTM-R2",
		zap.Int("num_batches", len(batches)),
		zap.Int("buffer_series_count", p.buffer.GetSeriesCount()),
	)

	// Process batches concurrently with limit
	semaphore := make(chan struct{}, p.config.MaxConcurrentRequests)
	var wg sync.WaitGroup

	for _, batch := range batches {
		wg.Add(1)
		go func(b *TimeSeriesBatch) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			p.processBatch(context.Background(), b)
		}(batch)
	}

	wg.Wait()
}

// processBatch sends a batch to TTM-R2 service and caches results
// This is the core method that replaces Isolation Forest's scoring
func (p *graniteTTMProcessor) processBatch(ctx context.Context, batch *TimeSeriesBatch) {
	// Call TTM-R2 service for anomaly detection
	response, err := p.ttmClient.DetectAnomalies(ctx, batch)
	if err != nil {
		p.logger.Error("Failed to detect anomalies with TTM-R2",
			zap.String("batch_id", batch.BatchID),
			zap.Error(err),
		)
		return
	}

	// Cache anomaly results for metric enrichment
	p.cacheResults(response)

	// Update statistics
	p.updateStatistics(response)

	// Run forecast only on the confirmed forecast-eligible series within this batch.
	// Unknown / unlisted metrics are anomaly-detected above but not forecasted.
	// Errors degrade gracefully — anomaly enrichment is always unaffected.
	if p.config.EnableForecasting {
		forecastBatch := filterBatchForForecasting(batch, p.config)
		if len(forecastBatch.Series) > 0 {
			forecastResp, fErr := p.ttmClient.Forecast(ctx, forecastBatch)
			if fErr != nil {
				p.logger.Warn("Forecast call failed, skipping forecast enrichment",
					zap.String("batch_id", batch.BatchID),
					zap.Error(fErr),
				)
			} else {
				p.cacheForecastResults(forecastResp)
			}
		}
	}
}

// filterBatchForForecasting returns a new batch containing only the series whose
// metric name matches the configured ForecastSelectors. Series not in the list
// still get anomaly detection but are skipped for forecasting.
func filterBatchForForecasting(batch *TimeSeriesBatch, cfg *Config) *TimeSeriesBatch {
	// Empty selector list means forecast everything (same as anomaly detection).
	if len(cfg.ForecastSelectors) == 0 {
		return batch
	}

	filtered := make([]*TimeSeriesData, 0, len(batch.Series))
	for _, ts := range batch.Series {
		if cfg.ShouldForecastMetric(ts.MetricName) {
			filtered = append(filtered, ts)
		}
	}
	return &TimeSeriesBatch{
		Series:    filtered,
		BatchID:   batch.BatchID,
		Timestamp: batch.Timestamp,
	}
}

// cacheResults stores TTM-R2 anomaly detection results in cache
func (p *graniteTTMProcessor) cacheResults(response *ttmproto.AnomalyResponse) {
	p.cacheMutex.Lock()
	defer p.cacheMutex.Unlock()

	now := time.Now()

	for _, result := range response.Results {
		// Calculate predicted, actual, and error from reconstruction data
		var predictedValue, actualValue, errorValue float64
		if result.Reconstruction != nil && len(result.Reconstruction.ReconstructedValues) > 0 {
			// Use the last reconstructed value as predicted
			predictedValue = result.Reconstruction.ReconstructedValues[len(result.Reconstruction.ReconstructedValues)-1]
			// Use the last reconstruction error
			if len(result.Reconstruction.ReconstructionErrors) > 0 {
				errorValue = result.Reconstruction.ReconstructionErrors[len(result.Reconstruction.ReconstructionErrors)-1]
			}
			// Calculate actual from predicted and error
			actualValue = predictedValue + errorValue
		}

		p.resultCache[result.SeriesId] = &CachedResult{
			SeriesID:       result.SeriesId,
			AnomalyScore:   float64(result.OverallScore),
			IsAnomaly:      result.OverallScore > float32(p.config.AnomalyThreshold),
			PredictedValue: predictedValue,
			ActualValue:    actualValue,
			Error:          errorValue,
			Timestamp:      now,
			Explanation:    result.Explanation,
		}
	}
}

// cacheForecastResults collapses a ForecastResponse into compact CachedForecastResult
// entries keyed by series ID. The full forecast_values arrays are not stored; only
// scalar summaries are retained so datapoint attributes stay compact.
func (p *graniteTTMProcessor) cacheForecastResults(response *ttmproto.ForecastResponse) {
	p.forecastCacheMutex.Lock()
	defer p.forecastCacheMutex.Unlock()

	now := time.Now()

	for _, result := range response.Results {
		if len(result.ForecastValues) == 0 {
			continue
		}

		// Derive peak value and its position within the forecast window.
		peakValue := result.ForecastValues[0]
		peakIdx := 0
		for i, v := range result.ForecastValues {
			if v > peakValue {
				peakValue = v
				peakIdx = i
			}
		}

		// Resolve peak timestamp (Unix ms). Fall back to now + step offset if missing.
		var peakTimeMs int64
		if peakIdx < len(result.ForecastTimestamps) {
			peakTimeMs = result.ForecastTimestamps[peakIdx] / int64(time.Millisecond)
		} else {
			peakTimeMs = now.Add(time.Duration(peakIdx) * time.Minute).UnixMilli()
		}

		// Confidence bounds at the horizon end (last step).
		last := len(result.ForecastValues) - 1
		var ciLower, ciUpper float64
		if result.Confidence != nil && last < len(result.Confidence.LowerBound) {
			ciLower = result.Confidence.LowerBound[last]
			ciUpper = result.Confidence.UpperBound[last]
		}

		fcr := evaluateForecastRisk(result.SeriesId, peakValue, ciUpper, now)
		fcr.PredictedPeakValue = peakValue
		fcr.PredictedPeakTime = peakTimeMs
		fcr.ConfidenceLower = ciLower
		fcr.ConfidenceUpper = ciUpper
		fcr.Timestamp = now

		p.forecastCache[result.SeriesId] = fcr
	}
}

// evaluateForecastRisk maps a predicted peak value to a risk level.
// Thresholds are defined per metric family and are intentionally conservative so
// operators can tune them via config in a later iteration.
func evaluateForecastRisk(seriesID string, peakValue, ciUpper float64, _ time.Time) *CachedForecastResult {
	fcr := &CachedForecastResult{SeriesID: seriesID}

	// Use the upper confidence bound when available for a more conservative estimate.
	effectiveValue := peakValue
	if ciUpper > peakValue {
		effectiveValue = ciUpper
	}

	// Classify risk by magnitude. Without per-metric limit context available at this
	// layer we use relative thresholds on the effective value. Operators with known
	// container limits should override via per-metric threshold config (future work).
	switch {
	case effectiveValue >= 0.9:
		fcr.RiskLevel = "critical"
		fcr.RiskScore = math.Min(effectiveValue, 1.0)
		fcr.PredictedIssue = true
		fcr.Reason = fmt.Sprintf("predicted value %.3f exceeds critical threshold (0.90) in 5m", effectiveValue)
	case effectiveValue >= 0.8:
		fcr.RiskLevel = "high"
		fcr.RiskScore = effectiveValue
		fcr.PredictedIssue = true
		fcr.Reason = fmt.Sprintf("predicted value %.3f exceeds high threshold (0.80) in 5m", effectiveValue)
	case effectiveValue >= 0.6:
		fcr.RiskLevel = "medium"
		fcr.RiskScore = effectiveValue
		fcr.PredictedIssue = false
		fcr.Reason = fmt.Sprintf("predicted value %.3f approaching elevated threshold (0.60) in 5m", effectiveValue)
	default:
		fcr.RiskLevel = "low"
		fcr.RiskScore = effectiveValue
		fcr.PredictedIssue = false
		fcr.Reason = fmt.Sprintf("predicted value %.3f within normal range in 5m", effectiveValue)
	}

	return fcr
}

// enrichMetricsWithCache adds TTM-R2 anomaly attributes to metrics using cached results,
// and optionally adds forecast summary attributes when EnableForecasting is set.
func (p *graniteTTMProcessor) enrichMetricsWithCache(md pmetric.Metrics) {
	p.cacheMutex.RLock()
	defer p.cacheMutex.RUnlock()

	// Take a read lock on the forecast cache only when forecasting is enabled.
	if p.config.EnableForecasting {
		p.forecastCacheMutex.RLock()
		defer p.forecastCacheMutex.RUnlock()
	}

	enrichedCount := 0

	for i := 0; i < md.ResourceMetrics().Len(); i++ {
		rm := md.ResourceMetrics().At(i)
		resourceAttrs := extractAttributes(rm.Resource().Attributes())

		for j := 0; j < rm.ScopeMetrics().Len(); j++ {
			sm := rm.ScopeMetrics().At(j)

			for k := 0; k < sm.Metrics().Len(); k++ {
				metric := sm.Metrics().At(k)

				// Process gauge metrics
				if metric.Type() == pmetric.MetricTypeGauge {
					for l := 0; l < metric.Gauge().DataPoints().Len(); l++ {
						dp := metric.Gauge().DataPoints().At(l)
						attrs := mergeAttributes(resourceAttrs, extractAttributes(dp.Attributes()))
						seriesID := p.buffer.generateSeriesID(metric.Name(), attrs)

						if result, ok := p.resultCache[seriesID]; ok {
							// Add TTM-R2 anomaly attributes (existing behavior, unchanged)
							dp.Attributes().PutDouble(p.config.ScoreAttribute, result.AnomalyScore)
							dp.Attributes().PutBool(p.config.ClassificationAttribute, result.IsAnomaly)
							dp.Attributes().PutDouble("ttm.predicted", result.PredictedValue)
							dp.Attributes().PutDouble("ttm.actual", result.ActualValue)
							dp.Attributes().PutDouble("ttm.error", result.Error)
							if result.Explanation != "" {
								dp.Attributes().PutStr("ttm.explanation", result.Explanation)
							}
							enrichedCount++
						}

						// Add forecast summary attributes on the same datapoint.
						if p.config.EnableForecasting {
							if fr, ok := p.forecastCache[seriesID]; ok {
								p.stampForecastAttributes(dp.Attributes(), fr)
							}
						}
					}
				}

				// Process sum metrics
				if metric.Type() == pmetric.MetricTypeSum {
					for l := 0; l < metric.Sum().DataPoints().Len(); l++ {
						dp := metric.Sum().DataPoints().At(l)
						attrs := mergeAttributes(resourceAttrs, extractAttributes(dp.Attributes()))
						seriesID := p.buffer.generateSeriesID(metric.Name(), attrs)

						if result, ok := p.resultCache[seriesID]; ok {
							// Add TTM-R2 anomaly attributes (existing behavior, unchanged)
							dp.Attributes().PutDouble(p.config.ScoreAttribute, result.AnomalyScore)
							dp.Attributes().PutBool(p.config.ClassificationAttribute, result.IsAnomaly)
							dp.Attributes().PutDouble("ttm.predicted", result.PredictedValue)
							dp.Attributes().PutDouble("ttm.actual", result.ActualValue)
							dp.Attributes().PutDouble("ttm.error", result.Error)
							if result.Explanation != "" {
								dp.Attributes().PutStr("ttm.explanation", result.Explanation)
							}
							enrichedCount++
						}

						// Add forecast summary attributes on the same datapoint.
						if p.config.EnableForecasting {
							if fr, ok := p.forecastCache[seriesID]; ok {
								p.stampForecastAttributes(dp.Attributes(), fr)
							}
						}
					}
				}
			}
		}
	}

	if enrichedCount > 0 {
		p.logger.Debug("Enriched metrics with TTM-R2 attributes",
			zap.Int("enriched_count", enrichedCount),
		)
	}
}

// stampForecastAttributes writes the compact forecast summary onto a datapoint's
// attribute map. All 9 attributes are always written so Grafana has the full picture.
func (p *graniteTTMProcessor) stampForecastAttributes(attrs pcommon.Map, fr *CachedForecastResult) {
	attrs.PutStr("forecast_five_mins", fr.RiskLevel)
	attrs.PutStr("forecast_risk_level", fr.RiskLevel)
	attrs.PutDouble("forecast_risk_score", fr.RiskScore)
	attrs.PutBool("forecast_predicted_issue", fr.PredictedIssue)
	attrs.PutDouble("forecast_predicted_peak_value", fr.PredictedPeakValue)
	attrs.PutInt("forecast_predicted_peak_time", fr.PredictedPeakTime)
	attrs.PutDouble("forecast_confidence_lower", fr.ConfidenceLower)
	attrs.PutDouble("forecast_confidence_upper", fr.ConfidenceUpper)
	attrs.PutStr("forecast_reason", fr.Reason)
}

// filterAnomalousMetrics removes non-anomalous metrics based on TTM-R2 results
func (p *graniteTTMProcessor) filterAnomalousMetrics(md pmetric.Metrics) pmetric.Metrics {
	// Implementation would filter based on cached results
	// This is a placeholder for filter mode
	return md
}

// backgroundCleanup periodically cleans up stale data
func (p *graniteTTMProcessor) backgroundCleanup() {
	defer p.wg.Done()

	for {
		select {
		case <-p.cleanupTicker.C:
			p.cleanup()
		case <-p.stopChan:
			return
		}
	}
}

// cleanup removes stale data from buffer and cache
func (p *graniteTTMProcessor) cleanup() {
	// Clean up stale time series
	removed := p.buffer.CleanupStale(10 * time.Minute)

	// Clean up stale anomaly cache entries
	p.cacheMutex.Lock()
	now := time.Now()
	staleCount := 0
	for seriesID, result := range p.resultCache {
		if now.Sub(result.Timestamp) > p.cacheMaxAge {
			delete(p.resultCache, seriesID)
			staleCount++
		}
	}
	p.cacheMutex.Unlock()

	// Clean up stale forecast cache entries
	staleForecastCount := 0
	if p.config.EnableForecasting {
		p.forecastCacheMutex.Lock()
		for seriesID, fr := range p.forecastCache {
			if now.Sub(fr.Timestamp) > p.cacheMaxAge {
				delete(p.forecastCache, seriesID)
				staleForecastCount++
			}
		}
		p.forecastCacheMutex.Unlock()
	}

	if removed > 0 || staleCount > 0 || staleForecastCount > 0 {
		p.logger.Debug("Cleanup completed",
			zap.Int("removed_series", removed),
			zap.Int("removed_cache_entries", staleCount),
			zap.Int("removed_forecast_cache_entries", staleForecastCount),
		)
	}
}

// updateStatistics updates processor statistics based on TTM-R2 results
func (p *graniteTTMProcessor) updateStatistics(response *ttmproto.AnomalyResponse) {
	p.statsMutex.Lock()
	defer p.statsMutex.Unlock()

	for _, result := range response.Results {
		if result.OverallScore > float32(p.config.AnomalyThreshold) {
			p.anomaliesDetected++
		}
	}
}

// Helper functions

// extractAttributes converts pcommon.Map to map[string]string
func extractAttributes(attrs pcommon.Map) map[string]string {
	result := make(map[string]string)
	attrs.Range(func(k string, v pcommon.Value) bool {
		result[k] = v.AsString()
		return true
	})
	return result
}

// mergeAttributes combines multiple attribute maps
func mergeAttributes(maps ...map[string]string) map[string]string {
	result := make(map[string]string)
	for _, m := range maps {
		for k, v := range m {
			result[k] = v
		}
	}
	return result
}

// Made with Bob
