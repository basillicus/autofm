# Metrics: Prometheus and Grafana

How the FM pipeline (`autofm.pipeline`) exposes its evaluation KPIs as
Prometheus metrics, so a run can be scraped into Prometheus and visualised in
Grafana like any operator-facing service dashboard.

The metrics themselves - what they mean and why these ones - are defined by the
evaluation strategy in [`SYSTEM_DESIGN.md` §Evaluation](SYSTEM_DESIGN.md#evaluation).
This doc covers only how they leave the pipeline and reach a dashboard. For
getting the pipeline running in the first place, see
[`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md).

The pipeline reports its KPIs three ways; the first two are for humans and
experiment tracking, the third is for monitoring:

- a readable summary printed at the end of a run,
- a full JSON bundle - `--metrics-out metrics.json`,
- Prometheus exposition metrics - `--prom-out autofm.prom`.

---

## What gets exported

Each run writes a `.prom` file. Every series carries the run context as labels
(`compressor`, `backbone`, `head`, `horizon`) so one dashboard can compare
configurations and forecast horizons side by side.

| Metric | Meaning |
|--------|---------|
| `autofm_eval_samples` | Samples the metrics were computed on |
| `autofm_macro_f1` | Macro-averaged F1 over fault classes |
| `autofm_weighted_f1` | Support-weighted F1 |
| `autofm_balanced_accuracy` | Balanced accuracy (multi-class) |
| `autofm_fault_precision` | Precision of fault-vs-Normal detection |
| `autofm_fault_recall` | Recall - fraction of real faults caught |
| `autofm_fault_f1` | F1 of fault detection |
| `autofm_fault_pr_auc` | PR-AUC (average precision) - the headline under imbalance |
| `autofm_fault_roc_auc` | ROC-AUC of fault detection |
| `autofm_fault_brier` | Brier score of the fault score (lower is better) |
| `autofm_fault_base_rate` | True fault base rate in the eval set |
| `autofm_mean_uncertainty` | Mean predictive uncertainty / confidence-gap |
| `autofm_class_f1{fault_class=...}` | Per-class F1, one series per fault class |

All are gauges - point-in-time values for one evaluation. Metrics that are
undefined for a run (for example the AUCs when only the Normal class is present
in the test split) are **omitted**, never emitted as a misleading `0`.

---

## Produce the file

```bash
python -m autofm.pipeline --config config_autofm.yaml \
  --prom-out artifacts/autofm.prom
```

The write is atomic (temp file plus rename), so a scrape never observes a
half-written file. Each run overwrites the same path.

---

## Scrape it with Prometheus

Because the pipeline runs and exits, use the node_exporter **textfile
collector** rather than a long-running HTTP endpoint - it is the idiomatic
pattern for a batch job that produces metrics and stops.

1. Run node_exporter pointed at a textfile directory, and write `--prom-out`
   into that same directory:

   ```bash
   node_exporter --collector.textfile.directory=/var/lib/node_exporter/textfile

   python -m autofm.pipeline --config config_autofm.yaml \
     --prom-out /var/lib/node_exporter/textfile/autofm.prom
   ```

   node_exporter then serves the metrics on its own `/metrics` (port `9100`).

2. Scrape node_exporter from Prometheus (`prometheus.yml`):

   ```yaml
   global:
     scrape_interval: 30s

   scrape_configs:
     - job_name: autofm
       static_configs:
         - targets: ["localhost:9100"]   # node_exporter
   ```

---

## Local demo stack

A self-contained laptop demo. Create the two config files and the compose file
below in a `monitoring/` directory, then bring the stack up.

`monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: autofm
    static_configs:
      - targets: ["node_exporter:9100"]
```

`monitoring/docker-compose.yml`:

```yaml
services:
  node_exporter:
    image: prom/node-exporter:latest
    command:
      - '--collector.textfile.directory=/textfile'
    volumes:
      - ../artifacts:/textfile          # pipeline writes autofm.prom here
    ports: ["9100:9100"]

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports: ["9090:9090"]
    depends_on: [node_exporter]

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Admin
    ports: ["3000:3000"]
    depends_on: [prometheus]
```

Bring it up and produce metrics:

```bash
docker compose -f monitoring/docker-compose.yml up -d
python -m autofm.pipeline --config config_autofm.yaml --prom-out artifacts/autofm.prom
```

Open Grafana at `http://localhost:3000`, add Prometheus
(`http://prometheus:9090`) as a data source, and build panels against the
`autofm_*` series.

---

## Example Grafana panels

- **Fault recall vs precision** - `autofm_fault_recall` and
  `autofm_fault_precision`. The operator's core trade-off: how many real faults
  are caught against how trustworthy each alert is.
- **PR-AUC by horizon** - `autofm_fault_pr_auc` with a `horizon` legend, to see
  how lead time trades off against confidence.
- **Per-class F1** - `autofm_class_f1` broken down by the `fault_class` label,
  as a bar gauge or heatmap.
- **Calibration** - `autofm_fault_brier` (lower is better) alongside
  `autofm_mean_uncertainty`, to watch whether confidence stays honest.

---

## Live serving (future)

For an online predictor that stays running, expose an HTTP `/metrics` endpoint
with `prometheus_client.start_http_server` instead of writing a textfile. The
metric names above are unchanged; only the transport differs. The
`prometheus_client` dependency is intentionally **not** required for the batch
textfile path documented here.
