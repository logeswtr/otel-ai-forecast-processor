// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package granitettmprocessor provides an OpenTelemetry Collector processor
// that uses IBM's Granite TTM-R2 (Tiny Time Mixers Release 2) foundation model
// for time series anomaly detection.
//
// This processor replaces the traditional Isolation Forest algorithm with a
// state-of-the-art transformer-based approach that better captures temporal
// patterns and multivariate relationships in time series data.
//
// # Key Features
//
//   - Time series anomaly detection using IBM Granite TTM-R2 model
//   - Support for reconstruction and forecast-based detection modes
//   - Configurable context windows (optimal: 512 time steps)
//   - Batch processing for efficiency
//   - Result caching for low-latency enrichment
//   - Flexible metric selection and filtering
//
// # Architecture
//
// The processor operates in three main stages:
//
//  1. Buffering: Accumulates metrics into time series sequences
//  2. Detection: Sends batches to TTM-R2 service via gRPC
//  3. Enrichment: Adds anomaly scores and classifications to metrics
//
// # Configuration Example
//
//	processors:
//	  granitettm:
//	    endpoint: "localhost:50051"
//	    context_length: 512
//	    forecast_horizon: 96
//	    anomaly_threshold: 0.7
//	    detection_mode: "reconstruction"
//	    mode: "enrich"
//
// # Migration from Isolation Forest
//
// This processor is designed as a drop-in replacement for the Isolation Forest
// processor with the following key differences:
//
//   - Requires external TTM-R2 inference service
//   - Processes time series sequences instead of single points
//   - Higher accuracy for complex temporal patterns
//   - Slightly higher latency (amortized across batches)
//
// For detailed migration instructions, see ISOLATION_FOREST_TO_TTM_MIGRATION_GUIDE.md
//
// # References
//
//   - IBM Granite TTM-R2: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
//   - TTM Paper: https://arxiv.org/abs/2401.03955
//   - OpenTelemetry Collector: https://opentelemetry.io/docs/collector/
package granitettmprocessor // import "github.com/open-telemetry/opentelemetry-collector-contrib/processor/granitettmprocessor"

// Made with Bob
