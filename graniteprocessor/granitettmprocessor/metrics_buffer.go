// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package granitettmprocessor

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"sync"
	"time"

	"go.uber.org/zap"
)

type MetricsBuffer struct {
	// Key: series identifier (metric_name + resource_attrs hash)
	timeSeries map[string]*TimeSeriesData

	contextLength int
	maxBufferSize int
	mutex         sync.RWMutex
	logger        *zap.Logger
}

// TimeSeriesData represents a single metric time series with circular buffer
// Optimized for TTM-R2 which requires sequences of consistent length
type TimeSeriesData struct {
	MetricName string
	SeriesID   string
	Timestamps []time.Time
	Values     []float64
	Attributes map[string]string

	// Circular buffer management
	head     int
	tail     int
	size     int
	capacity int
	full     bool

	lastUpdate time.Time
	mutex      sync.RWMutex
}

// TimeSeriesBatch represents a batch of time series ready for TTM-R2 processing
type TimeSeriesBatch struct {
	Series    []*TimeSeriesData
	BatchID   string
	Timestamp time.Time
}

// NewMetricsBuffer creates a new metrics buffer
func NewMetricsBuffer(contextLength, maxBufferSize int, logger *zap.Logger) *MetricsBuffer {
	return &MetricsBuffer{
		timeSeries:    make(map[string]*TimeSeriesData),
		contextLength: contextLength,
		maxBufferSize: maxBufferSize,
		logger:        logger,
	}
}

// AddDataPoint adds a new data point to the buffer
func (mb *MetricsBuffer) AddDataPoint(
	metricName string,
	timestamp time.Time,
	value float64,
	attributes map[string]string,
) error {
	// Generate series ID
	seriesID := mb.generateSeriesID(metricName, attributes)

	mb.mutex.Lock()
	defer mb.mutex.Unlock()

	// Get or create time series
	ts, exists := mb.timeSeries[seriesID]
	if !exists {
		ts = NewTimeSeriesData(metricName, seriesID, attributes, mb.contextLength)
		mb.timeSeries[seriesID] = ts
		mb.logger.Debug("Created new time series",
			zap.String("series_id", seriesID),
			zap.String("metric_name", metricName),
		)
	}

	// Add data point to time series
	if err := ts.AddPoint(timestamp, value); err != nil {
		return fmt.Errorf("failed to add point to time series: %w", err)
	}

	return nil
}

// GetReadyBatches returns batches of time series that have enough data points for TTM-R2
func (mb *MetricsBuffer) GetReadyBatches(batchSize int) []*TimeSeriesBatch {
	mb.mutex.RLock()
	defer mb.mutex.RUnlock()

	var readySeries []*TimeSeriesData

	// Collect all series that are ready for processing
	// TTM-R2 requires at least context_length points
	for _, ts := range mb.timeSeries {
		if ts.IsReady(mb.contextLength) {
			readySeries = append(readySeries, ts)
		}
	}

	// Split into batches
	var batches []*TimeSeriesBatch
	for i := 0; i < len(readySeries); i += batchSize {
		end := i + batchSize
		if end > len(readySeries) {
			end = len(readySeries)
		}

		batch := &TimeSeriesBatch{
			Series:    readySeries[i:end],
			BatchID:   fmt.Sprintf("batch_%d_%d", time.Now().Unix(), i),
			Timestamp: time.Now(),
		}
		batches = append(batches, batch)
	}

	mb.logger.Debug("Generated batches for TTM-R2",
		zap.Int("total_series", len(readySeries)),
		zap.Int("num_batches", len(batches)),
		zap.Int("context_length", mb.contextLength),
	)

	return batches
}

// GetSeriesCount returns the number of time series in the buffer
func (mb *MetricsBuffer) GetSeriesCount() int {
	mb.mutex.RLock()
	defer mb.mutex.RUnlock()
	return len(mb.timeSeries)
}

// CleanupStale removes time series that haven't been updated recently
func (mb *MetricsBuffer) CleanupStale(maxAge time.Duration) int {
	mb.mutex.Lock()
	defer mb.mutex.Unlock()

	now := time.Now()
	removed := 0

	for seriesID, ts := range mb.timeSeries {
		if now.Sub(ts.GetLastUpdate()) > maxAge {
			delete(mb.timeSeries, seriesID)
			removed++
		}
	}

	if removed > 0 {
		mb.logger.Info("Cleaned up stale time series",
			zap.Int("removed", removed),
			zap.Int("remaining", len(mb.timeSeries)),
		)
	}

	return removed
}

// generateSeriesID creates a unique identifier for a time series
func (mb *MetricsBuffer) generateSeriesID(metricName string, attributes map[string]string) string {
	// Sort attributes for consistent hashing
	keys := make([]string, 0, len(attributes))
	for k := range attributes {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Build string representation
	var builder string
	builder = metricName
	for _, k := range keys {
		builder += fmt.Sprintf("|%s=%s", k, attributes[k])
	}

	// Hash to create compact ID
	hash := sha256.Sum256([]byte(builder))
	return hex.EncodeToString(hash[:8]) // Use first 8 bytes for shorter ID
}

// NewTimeSeriesData creates a new time series data structure
func NewTimeSeriesData(metricName, seriesID string, attributes map[string]string, capacity int) *TimeSeriesData {
	return &TimeSeriesData{
		MetricName: metricName,
		SeriesID:   seriesID,
		Timestamps: make([]time.Time, capacity),
		Values:     make([]float64, capacity),
		Attributes: attributes,
		capacity:   capacity,
		head:       0,
		tail:       0,
		size:       0,
		full:       false,
		lastUpdate: time.Now(),
	}
}

// AddPoint adds a new data point to the time series (circular buffer)
// Optimized for performance with minimal branching
func (ts *TimeSeriesData) AddPoint(timestamp time.Time, value float64) error {
	ts.mutex.Lock()
	defer ts.mutex.Unlock()

	// Write to current tail position
	ts.Timestamps[ts.tail] = timestamp
	ts.Values[ts.tail] = value

	// Advance tail with wrap-around
	ts.tail = (ts.tail + 1) % ts.capacity

	// Optimized full buffer handling
	if ts.full {
		// Buffer full: advance head to maintain sliding window
		ts.head = (ts.head + 1) % ts.capacity
	} else {
		// Buffer filling: increment size and check if now full
		ts.size++
		ts.full = (ts.size == ts.capacity)
	}

	ts.lastUpdate = time.Now()
	return nil
}

// IsReady checks if the time series has enough data points for TTM-R2 processing
func (ts *TimeSeriesData) IsReady(minPoints int) bool {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()
	return ts.size >= minPoints
}

// GetData returns a copy of the time series data in chronological order
// This is what gets sent to TTM-R2 for inference
// Optimized to minimize allocations and copies
func (ts *TimeSeriesData) GetData() ([]time.Time, []float64) {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()

	// Pre-allocate with exact size needed
	timestamps := make([]time.Time, ts.size)
	values := make([]float64, ts.size)

	if !ts.full {
		// Buffer not full: data is sequential from 0 to size
		copy(timestamps, ts.Timestamps[:ts.size])
		copy(values, ts.Values[:ts.size])
		return timestamps, values
	}

	// Buffer full: unwrap circular structure efficiently
	// Calculate split point once
	splitPoint := ts.capacity - ts.head

	// Copy older data (from head to end of array)
	copy(timestamps[:splitPoint], ts.Timestamps[ts.head:])
	copy(values[:splitPoint], ts.Values[ts.head:])

	// Copy newer data (from start of array to tail)
	copy(timestamps[splitPoint:], ts.Timestamps[:ts.tail])
	copy(values[splitPoint:], ts.Values[:ts.tail])

	return timestamps, values
}

// GetLastUpdate returns the last update time
func (ts *TimeSeriesData) GetLastUpdate() time.Time {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()
	return ts.lastUpdate
}

// GetSize returns the current number of data points
func (ts *TimeSeriesData) GetSize() int {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()
	return ts.size
}

// Made with Bob
