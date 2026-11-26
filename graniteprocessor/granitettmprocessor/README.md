# Granite TTM-R2 Processor

**Status:** Alpha (metrics)  
**Model:** IBM Granite TTM-R2 (Tiny Time Mixers Release 2)

## Overview

The `granitettm` processor adds time series anomaly detection and short-horizon forecasting to OpenTelemetry metrics. It uses IBM's pre-trained [Granite TTM-R2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) foundation model, so there are no per-metric thresholds to tune and no training step.

The processor does not run the model itself. It buffers metrics into per-series time windows, sends them to a separate TTM inference service over gRPC, and writes the results back onto the metric data points as attributes.

- **Anomaly detection** runs on every Gauge and Sum metric.
- **Forecasting** produces a compact risk summary for a configurable set of metrics.
- **Pass-through on the hot path.** Metrics are never blocked on the model. Each data point is enriched from the most recent cached result for its series.

## Architecture

```
 OTLP metrics
      │
      ▼
┌──────────────────────────────────────────────────────┐
│ OpenTelemetry Collector                              │
│                                                      │
│  granitettm processor                                │
│  ┌────────────┐   ┌────────────┐   ┌──────────────┐  │
│  │ Per-series │──▶│  Batching  │──▶│  gRPC client │──┼──┐
│  │   buffer   │   │ (periodic) │   └──────────────┘  │  │
│  └────────────┘   └────────────┘                     │  │
│        ▲                                             │  │
│        │ enrich from result cache                    │  │
│  metrics in ─────────────────────────▶ metrics out   │  │
└──────────────────────────────────────────────────────┘  │
                                                          ▼
                                       ┌───────────────────────────────┐
                                       │ TTM inference service         │
                                       │ (Python / PyTorch, port 50051)│
                                       └───────────────────────────────┘
```

### How it works

1. **Buffer.** Each Gauge or Sum data point is appended to a circular buffer. The series is identified by the metric name plus the merged resource and data point attributes. All metrics are buffered, including ones not listed in `forecast_selectors`.
2. **Wait for history.** A series is sent to the model only after it holds at least `context_length` points. Until then its metrics pass through unchanged.
3. **Batch and call.** Every `flush_interval`, ready series are grouped into batches of `batch_size` and sent to the service. Up to `max_concurrent_requests` batches run in parallel.
4. **Score.** The service cleans and normalizes each series, runs the model, and returns per-point anomaly scores and an overall score (the maximum point score).
5. **Forecast (optional).** If `enable_forecasting` is on, series that match `forecast_selectors` are also sent to the `Forecast` RPC. The processor reduces the result to a risk summary.
6. **Enrich.** Results are cached for 5 minutes. As metrics flow through, matching data points get the attributes listed under [Output attributes](#output-attributes).

Buffered series that receive no data for 10 minutes are removed.

## Prerequisites

The processor needs a running TTM inference service. The service lives in the parent directory of this processor (`../`), with its own `Dockerfile` and `ttm_requirements.txt`.

```bash
cd ..
docker build -t ttm-inference-service .
docker run -p 50051:50051 ttm-inference-service
```

The container exposes gRPC on `50051`. By default it loads the model given by the `--model` argument of `server.py`; override it by changing the container command.

If the service is unreachable at startup, the processor logs a warning and keeps running. Enrichment starts once the service responds.

## Configuration

### Minimal example

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  granitettm:
    endpoint: "localhost:50051"

exporters:
  prometheus:
    endpoint: "0.0.0.0:9464"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [granitettm]
      exporters: [prometheus]
```

### Full example

```yaml
processors:
  granitettm:
    # Inference service connection
    endpoint: "ttm-service:50051"
    timeout: "10s"
    tls:
      insecure: false
      cert_file: "/path/to/client.crt"
      key_file: "/path/to/client.key"
      ca_file: "/path/to/ca.crt"

    # Time series
    context_length: 512
    forecast_horizon: 96
    buffer_size: 1024
    flush_interval: "30s"

    # Anomaly detection
    anomaly_threshold: 0.7
    detection_mode: "reconstruction"   # reconstruction | forecast | both
    sensitivity: "medium"              # low | medium | high

    # Forecasting
    enable_forecasting: true
    forecast_selectors:
      - name_pattern: "container_cpu_usage_seconds_total"
      - name_pattern: "container_memory_working_set_bytes"

    # Output
    mode: "enrich"
    score_attribute: "ttm.anomaly_score"
    classification_attribute: "ttm.is_anomaly"

    # Performance
    max_concurrent_requests: 4
    batch_size: 32
```

### Parameters

#### Connection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | string | `localhost:50051` | gRPC address of the TTM inference service |
| `timeout` | duration | `5s` | Per-request timeout |
| `tls.insecure` | bool | `true` | Disable TLS. Set to `false` outside development |
| `tls.cert_file` | string | - | Client certificate. Required with `key_file` when `tls.insecure` is `false` |
| `tls.key_file` | string | - | Client key |
| `tls.ca_file` | string | - | CA certificate |

#### Time series

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_length` | int | `512` | Points of history sent to the model. Must be at least 64 |
| `forecast_horizon` | int | `96` | Future steps to predict |
| `buffer_size` | int | `1024` | Maximum points kept per series. Must be at least `context_length` |
| `flush_interval` | duration | `5m` | How often ready series are sent to the service |

#### Anomaly detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `anomaly_threshold` | float | `0.7` | Overall score above which a series is classified as anomalous (0-1) |
| `detection_mode` | string | `reconstruction` | `reconstruction`, `forecast`, or `both` |
| `sensitivity` | string | `medium` | `low`, `medium`, or `high`. Scales the threshold used by the service (x0.5, x1.0, x1.5) |

#### Forecasting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_forecasting` | bool | `true` | Call the `Forecast` RPC and add `forecast_*` attributes |
| `forecast_selectors` | list | six container metrics (below) | Metrics eligible for forecasting. An empty list forecasts every metric |
| `forecast_selectors[].name_pattern` | string | - | Matches a metric name that equals, starts with, or ends with the pattern. `*` matches all |

Default `forecast_selectors`:
`container_cpu_usage_seconds_total`, `container_memory_working_set_bytes`, `container_memory_usage_bytes`, `container_fs_usage_bytes`, `container_network_receive_bytes_total`, `container_network_transmit_bytes_total`.

Metrics that don't match are still scored for anomalies. They just don't get `forecast_*` attributes.

#### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `enrich` | `enrich` adds attributes. `filter` and `both` are accepted, but filtering is not implemented yet |
| `score_attribute` | string | `ttm.anomaly_score` | Attribute name for the score |
| `classification_attribute` | string | `ttm.is_anomaly` | Attribute name for the boolean classification. Must differ from `score_attribute` |

#### Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_requests` | int | `4` | Maximum parallel batches in flight |
| `batch_size` | int | `32` | Series per batch |

## Detection modes

- **`reconstruction`** (default). Passes the series through the model and compares the output with the observed values. Lowest cost per series.
- **`forecast`**. Uses the first half of the window as context and compares the model's forecast with the second half. Better suited to trend changes.
- **`both`**. Averages the two scores. Highest cost.

### Scoring

For each series the service computes the absolute error between observed and predicted values, then:

```
threshold = mean(error) + 2 * std(error)
score     = sigmoid(error / threshold - 1)      # 0 to 1
```

The series' overall score is the maximum point score. The processor marks `ttm.is_anomaly` as true when the overall score is above `anomaly_threshold`. Because the threshold is relative to each series' own error distribution, one setting works across metrics with very different scales.

## Output attributes

Attributes are added to the data points of Gauge and Sum metrics.

### Anomaly attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ttm.anomaly_score` | double | Overall anomaly score (0-1, higher is more anomalous) |
| `ttm.is_anomaly` | bool | Whether the score exceeds `anomaly_threshold` |
| `ttm.predicted` | double | Model-predicted value for the latest point |
| `ttm.actual` | double | Observed value for the latest point |
| `ttm.error` | double | Prediction error for the latest point |
| `ttm.explanation` | string | Short description, for example the number of anomalous points. Omitted when empty |

The names of the first two attributes are set by `score_attribute` and `classification_attribute`.

### Forecast attributes

Added when `enable_forecasting` is `true` and the series matches `forecast_selectors`. Only a summary is attached, not the full forecast array, to keep attribute size small.

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecast_five_mins` | string | Risk level (same value as `forecast_risk_level`) |
| `forecast_risk_level` | string | `low`, `medium`, `high`, or `critical` |
| `forecast_risk_score` | double | Risk score (0-1) |
| `forecast_predicted_issue` | bool | `true` for `high` and `critical` |
| `forecast_predicted_peak_value` | double | Highest predicted value in the horizon |
| `forecast_predicted_peak_time` | int | Time of the predicted peak, Unix milliseconds |
| `forecast_confidence_lower` | double | 95% lower bound at the end of the horizon |
| `forecast_confidence_upper` | double | 95% upper bound at the end of the horizon |
| `forecast_reason` | string | Human-readable reason for the risk level |

The risk level is based on the larger of the predicted peak and the upper confidence bound:

| Value | Risk level |
|-------|------------|
| >= 0.90 | `critical` |
| >= 0.80 | `high` |
| >= 0.60 | `medium` |
| below 0.60 | `low` |

These thresholds are fixed and assume values on a 0-1 scale, such as utilization ratios. Per-metric thresholds are not configurable yet.

## Limitations

- **Cold start.** Nothing is scored until a series has `context_length` points. With a long scrape interval this can take a while.
- **Result lag.** Enrichment uses the most recent flush, so results can be up to one `flush_interval` old. Cached results expire after 5 minutes.
- **Metric types.** Only Gauge and Sum metrics are processed. Histograms and summaries are ignored.
- **Univariate.** Each series is scored independently. Correlation across metrics is not used.
- **`filter` mode** is not implemented. Use `enrich`.
- **External dependency.** The TTM inference service must be deployed and reachable.

## Performance tuning

### High throughput

```yaml
granitettm:
  buffer_size: 4096
  flush_interval: "10s"
  max_concurrent_requests: 16
  batch_size: 128
```

### Low latency

```yaml
granitettm:
  buffer_size: 512
  flush_interval: "5s"
  max_concurrent_requests: 2
  batch_size: 16
  detection_mode: "reconstruction"
```

### Fewer resources

```yaml
granitettm:
  buffer_size: 512
  batch_size: 16
  enable_forecasting: false
```

## Troubleshooting

### Cannot connect to the inference service

- Check the service is running: `docker ps | grep ttm`
- Check `endpoint`, and that the port is reachable from the Collector.
- List the exposed methods: `grpcurl -plaintext localhost:50051 list`

### No attributes appear on metrics

- Confirm the series has received at least `context_length` points. This is normal during startup.
- Reduce `context_length` (minimum 64) or `flush_interval` if metrics are sparse.
- Check the `granitettm` pipeline position. It must run before the exporter.

### No `forecast_*` attributes

- Confirm `enable_forecasting` is `true`.
- Check the metric name against `forecast_selectors`, or set `forecast_selectors: []` to forecast everything.

### Requests time out

- Increase `timeout`.
- Reduce `batch_size`.
- Scale the inference service, or run it on a GPU.

## References

- [IBM Granite TTM-R2 model](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
- [TTM paper](https://arxiv.org/abs/2401.03955)
- [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)

## License

Apache 2.0
