<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9401f20b-9b95-43bd-b514-61857ed53d23" />

# Spot Problems Before Your Pager Does: AI Anomaly Detection and Forecasting in the OpenTelemetry Collector

*Add AI-powered anomaly detection and short-term forecasting to your metrics with a single Collector processor. It needs no per-metric thresholds and no model training.*

---

## The problem: static alerts don't scale

If you run containers or microservices, you probably have alerts like these:

- "CPU above 80% for 5 minutes"
- "Memory above 90%"

They work until they don't:

- **Every metric needs its own threshold.** Hundreds of services and thousands of series means thousands of numbers to pick and maintain.
- **"Normal" changes over time.** A traffic spike at noon is fine. The same spike at 3 a.m. is not. A fixed number can't tell the difference.
- **You find out after the fact.** A threshold alert fires when the problem has already started.

What teams actually want is for the metrics pipeline to say: *"this series looks unusual compared with its own history"* and *"this one is heading toward trouble."*

That is what the **Granite TTM processor** (`granitettm`) does.

---

## What is it?

`granitettm` is an **OpenTelemetry Collector processor** that adds two things to your metrics:

| Capability | What you get |
|---|---|
| **Anomaly detection** | A score from 0 to 1 and a true/false flag on each metric data point, saying how unusual the latest behavior is for that series |
| **Short-horizon forecasting** | A compact risk summary (`low`, `medium`, `high`, `critical`) with the predicted peak, its time, and a confidence range |

It uses IBM's open-source [**Granite TTM-R2**](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) model. TTM stands for *Tiny Time Mixers*. It is a small, pre-trained foundation model for time series. In practice this means:

- **No training step.** The model is already trained. You point the processor at it and go.
- **No per-metric tuning.** Scores are relative to each series' own behavior, so one setting works for CPU, memory, network and more.
- **Light enough to run on CPU.** The model is compact, so you don't need a GPU to get started.

The processor **adds attributes to your existing metrics**. It doesn't create a new data format or require a new backend. Anything that can read OpenTelemetry metrics, such as Prometheus, Grafana, or an OTLP backend, can use the results.

---

## How it works

The Collector doesn't run the model itself. That keeps the Collector lightweight and lets you scale the AI part separately.

```
 Your apps / agents
        │  OTLP metrics
        ▼
┌──────────────────────────────────────────────┐
│ OpenTelemetry Collector                      │
│                                              │
│   granitettm processor                       │
│   ├─ keeps a rolling history per series      │
│   ├─ sends ready series in batches ──────────┼──► TTM inference service
│   └─ stamps results onto your metrics ◄──────┼───   (Python + PyTorch,
│                                              │        gRPC on port 50051)
└──────────────────────────────────────────────┘
        │  Same metrics, now with ttm.* and forecast_* attributes
        ▼
  Prometheus / Grafana / any OTLP backend
```

In plain language:

1. **Collect history.** For every metric series, the processor keeps a rolling window of recent values. A series is identified by its metric name plus its resource and data point attributes, for example one container's CPU usage.
2. **Wait until there's enough history.** By default a series needs 512 points before it is scored. Until then its metrics pass through unchanged.
3. **Send batches to the model.** On a timer (`flush_interval`), series with enough history are grouped into batches and sent to the inference service over gRPC.
4. **Clean and normalize.** The service removes bad values (NaN or infinity), evens out irregular sampling, and normalizes each series so the model sees comparable data.
5. **Score and forecast.** The model produces an anomaly score for each point and, if enabled, a forecast for the next steps.
6. **Enrich.** Results are cached for 5 minutes. As metrics flow through the Collector, matching data points get the new attributes.

**Metrics are never held up waiting for the AI.** Enrichment uses the most recent cached result, so your pipeline keeps its normal latency. If the inference service is down at startup, the processor logs a warning and carries on. Enrichment begins when the service responds.

---

## How the anomaly score works

For each series the service compares what the model *expected* with what was *actually observed*, and looks at the size of the gap:

```
threshold = mean(error) + 2 × std(error)
score     = sigmoid(error / threshold − 1)     # between 0 and 1
```

Because the threshold comes from each series' **own error distribution**, a metric measured in bytes and one measured in percent are scored on the same footing. A series is flagged as an anomaly when its overall score (the highest point score) goes above `anomaly_threshold`. The default is 0.7.

There are three detection modes:

| Mode | What it does | Best for |
|---|---|---|
| `reconstruction` (default) | Compares the model's output with the observed values | Lowest cost, good general default |
| `forecast` | Uses the first half of the window to predict the second half | Catching trend changes |
| `both` | Averages the two scores | Highest sensitivity, highest cost |

---

## Forecasting: the "what's next" view

When `enable_forecasting` is on (it is by default), series that match your `forecast_selectors` are also forecast. By default these are six container metrics:

- `container_cpu_usage_seconds_total`
- `container_memory_working_set_bytes`
- `container_memory_usage_bytes`
- `container_fs_usage_bytes`
- `container_network_receive_bytes_total`
- `container_network_transmit_bytes_total`

Every other metric still gets anomaly scores. It just doesn't get forecast attributes. Set `forecast_selectors: []` to forecast everything.

To keep your metrics small, the processor doesn't attach the full forecast array. It attaches a **summary**:

| Attribute | Meaning |
|---|---|
| `forecast_risk_level` | `low`, `medium`, `high`, or `critical` |
| `forecast_risk_score` | Risk score between 0 and 1 |
| `forecast_predicted_issue` | `true` when the risk is `high` or `critical` |
| `forecast_predicted_peak_value` | Highest predicted value in the horizon |
| `forecast_predicted_peak_time` | When that peak is expected (Unix ms) |
| `forecast_confidence_lower` / `_upper` | Approximate range at the end of the horizon |
| `forecast_reason` | A short, human-readable explanation |

The risk level is based on the larger of the predicted peak and the upper confidence bound:

| Value | Risk level |
|---|---|
| ≥ 0.90 | `critical` |
| ≥ 0.80 | `high` |
| ≥ 0.60 | `medium` |
| < 0.60 | `low` |

> **Note:** These cut-offs are fixed and assume values on a 0–1 scale, such as utilization ratios. Per-metric thresholds aren't configurable yet. See [Current limitations](#current-limitations).

---

## What you see on your metrics

Here is a data point after enrichment. The attributes are illustrative.

```
container_memory_working_set_bytes{
  ttm.anomaly_score           = 0.91
  ttm.is_anomaly              = true
  ttm.predicted               = 0.42
  ttm.actual                  = 0.78
  ttm.error                   = 0.36
  ttm.explanation             = "Detected 3/512 anomalous points (0.6%) using reconstruction method. Maximum anomaly score: 0.910"
  forecast_risk_level         = "high"
  forecast_risk_score         = 0.84
  forecast_predicted_issue    = true
  forecast_predicted_peak_value = 0.84
  forecast_reason             = "predicted value 0.840 exceeds high threshold (0.80) in 5m"
}
```

From there you can:

- Build Grafana panels that highlight series where `ttm.is_anomaly` is true.
- Alert on `forecast_predicted_issue = true` to get a warning *before* the problem lands.
- Use `ttm.explanation` and `forecast_reason` as human-readable context in dashboards and tickets.

---

## Getting started in four steps

### 1. Start the inference service

The service is a small Python app that loads the Granite TTM-R2 model and serves it over gRPC. From the `graniteprocessor/` directory:

```bash
docker build -t ttm-inference-service .
docker run -p 50051:50051 ttm-inference-service
```

### 2. Add the processor to your Collector config

The minimal setup is one line of real configuration:

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

### 3. Send metrics

Keep sending metrics as you do today. After a series has collected enough history (`context_length` points), it starts receiving `ttm.*` attributes.

### 4. Use the results

Query or visualize the new attributes in your backend, and build alerts on `ttm.is_anomaly` or `forecast_predicted_issue`.

---

## Tuning to your environment

The defaults are sensible, and a few settings make the biggest difference:

| Goal | Settings to adjust |
|---|---|
| **Get results sooner** | Lower `context_length` (minimum 64) and `flush_interval` |
| **Fewer false alarms** | Raise `anomaly_threshold`, or set `sensitivity: "low"` |
| **Catch subtler issues** | Lower `anomaly_threshold`, or set `sensitivity: "high"` |
| **Handle lots of series** | Increase `batch_size` and `max_concurrent_requests`, and scale the inference service |
| **Use fewer resources** | Set `enable_forecasting: false` and use a smaller `batch_size` |
| **Production security** | Set `tls.insecure: false` and provide `cert_file`, `key_file` and `ca_file` |

A high-throughput example:

```yaml
granitettm:
  endpoint: "ttm-service:50051"
  buffer_size: 4096
  flush_interval: "10s"
  max_concurrent_requests: 16
  batch_size: 128
```

---

## Current limitations

This processor is **alpha**. These points will help you set expectations:

- **Cold start.** Nothing is scored until a series has `context_length` points. With a long scrape interval this can take a while.
- **Result lag.** Enrichment uses the most recent flush, so results can be up to one `flush_interval` old. Cached results expire after 5 minutes.
- **Gauges and sums only.** Histograms and summaries are ignored for now.
- **One series at a time.** Each series is scored independently. Correlation across metrics isn't used.
- **Fixed forecast risk thresholds.** The 0.60, 0.80 and 0.90 cut-offs assume 0–1 values. Metrics in other units, such as raw bytes or cumulative counters, won't map meaningfully onto these levels yet. Per-metric thresholds are planned work.
- **`filter` mode isn't implemented.** Use `mode: enrich`.
- **A separate service is required.** The inference service must be deployed and reachable from the Collector.

---

## Troubleshooting quick guide

| Symptom | What to check |
|---|---|
| Can't connect to the service | Check the container is running, the `endpoint` is right, and the port is reachable. `grpcurl -plaintext localhost:50051 list` lists the available methods. |
| No `ttm.*` attributes | The series may not have `context_length` points yet. This is normal at startup. Also check that `granitettm` comes before your exporter in the pipeline. |
| No `forecast_*` attributes | Check `enable_forecasting` is `true` and that the metric name matches `forecast_selectors`. |
| Requests time out | Increase `timeout`, reduce `batch_size`, or scale the service or run it on a GPU. |

---

## Why this approach

- **It builds on OpenTelemetry.** There is no new agent and no new data format. It's a processor in the pipeline you already run.
- **It needs no ML expertise.** You don't train, label, or manage models. Granite TTM-R2 is pre-trained.
- **The Collector stays light.** Inference runs in its own service, so you can scale and update it independently.
- **It's built to degrade gracefully.** If the model service is slow or unavailable, your metrics keep flowing and are just not enriched.
- **It's open.** The model is open-source, and the processor is Apache 2.0 licensed.

---

## Learn more

- [IBM Granite TTM-R2 model on Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
- [TTM research paper](https://arxiv.org/abs/2401.03955)
- [OpenTelemetry Collector documentation](https://opentelemetry.io/docs/collector/)
- Processor reference: [`granitettmprocessor/README.md`](../graniteprocessor/granitettmprocessor/README.md)

We'd love your feedback. Try it on your own metrics, tell us which series it flags well and which it misses, and open an issue with what you find.

