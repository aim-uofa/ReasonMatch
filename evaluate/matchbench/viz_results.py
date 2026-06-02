from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from flask import Flask, abort, render_template_string, request, send_from_directory

from utils import compare_mappings


DEFAULT_STAGE_OPTIONS = ("one_to_one", "one_to_multi", "multi_to_multi")


INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Evaluation Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
    th { background: #f5f5f5; }
    .sample-link { font-family: monospace; }
  </style>
</head>
<body>
  <h1>Evaluation Dashboard</h1>
  <h2>Runs</h2>
  <table>
    <tr>
      <th>Run</th>
      <th>Model</th>
      <th>Total</th>
      <th>Avg Precision</th>
      <th>Avg Recall</th>
      <th>Avg F1</th>
      <th>Samples</th>
    </tr>
    {% for run in runs %}
    <tr>
      <td>{{ run.name }}</td>
      <td>{{ run.summary.config.model_name }}</td>
      <td>{{ run.summary.total_samples }}</td>
      <td>{{ '%.3f'|format(run.summary.avg_precision or 0) }}</td>
      <td>{{ '%.3f'|format(run.summary.avg_recall) }}</td>
      <td>{{ '%.3f'|format(run.summary.avg_f1 or 0.0) }}</td>
      <td><a href="{{ url_for('list_samples', run_name=run.name) }}">Browse</a></td>
    </tr>
    {% endfor %}
  </table>

  <h2>Compare Sample</h2>
  <form method="get" action="{{ url_for('compare_sample') }}">
    <label>Sample ID:
      <input type="text" name="sample_id" placeholder="uco3d_000001" required />
    </label>
    <button type="submit">Compare</button>
  </form>
</body>
</html>
"""


SAMPLES_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Samples - {{ run.name }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
    th { background: #f5f5f5; }
    .meta { font-size: 0.85rem; color: #555; }
    section { margin-bottom: 2rem; }
  </style>
</head>
<body>
  <h1>{{ run.name }} &mdash; Samples</h1>
  <p><a href="{{ url_for('index') }}">Back to runs</a></p>

  <section>
  <h2>Dataset Summary</h2>
  <table>
    <tr>
      <th>Dataset Key</th>
      <th>Samples</th>
      <th>Avg Precision</th>
      <th>Avg Recall</th>
      <th>Avg F1</th>
      <th>Avg Overlap</th>
      <th>Overlap Range</th>
    </tr>
    {% for ds_key, stats in dataset_stats.items() %}
    <tr>
      <td>{{ ds_key }}</td>
      <td>{{ stats.count }}</td>
      <td>{{ '%.3f'|format(stats.avg_precision or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_recall) }}</td>
      <td>{{ '%.3f'|format(stats.avg_f1 or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_overlap or 0) if stats.avg_overlap is not none else 'N/A' }}</td>
      <td>
        {%- if stats.overlap_rank_min is not none -%}
          {{ '%.2f'|format(stats.overlap_rank_min) }} - {{ '%.2f'|format(stats.overlap_rank_max) }}
        {%- else -%}
          N/A
        {%- endif -%}
      </td>
    </tr>
    {% endfor %}
  </table>
  <canvas id="datasetChart" height="120"></canvas>
  </section>

  <section>
  <h2>Stage Summary</h2>
  <table>
    <tr>
      <th>Stage</th>
      <th>Samples</th>
      <th>Avg Precision</th>
      <th>Avg Recall</th>
      <th>Avg F1</th>
    </tr>
    {% for stage, stats in stage_stats.items() %}
    <tr>
      <td>{{ stage }}</td>
      <td>{{ stats.count }}</td>
      <td>{{ '%.3f'|format(stats.avg_precision or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_recall or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_f1 or 0) }}</td>
    </tr>
    {% endfor %}
  </table>
  <canvas id="stageChart" height="120"></canvas>
  </section>

  <section>
  <h2>Difficulty Summary</h2>
  <table>
    <tr>
      <th>Difficulty</th>
      <th>Samples</th>
      <th>Avg Precision</th>
      <th>Avg Recall</th>
      <th>Avg F1</th>
    </tr>
    {% for diff, stats in difficulty_stats.items() %}
    <tr>
      <td>{{ diff }}</td>
      <td>{{ stats.count }}</td>
      <td>{{ '%.3f'|format(stats.avg_precision or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_recall or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_f1 or 0) }}</td>
    </tr>
    {% endfor %}
  </table>
  <canvas id="difficultyChart" height="120"></canvas>
  </section>

  <section>
  <h2>JSON Quality</h2>
  <table>
    <tr><th>Total</th><td>{{ json_quality.total }}</td></tr>
    <tr><th>Success</th><td>{{ json_quality.json_success }}</td></tr>
    <tr><th>Missing JSON</th><td>{{ json_quality.json_absent }}</td></tr>
    <tr><th>Parse Failures</th><td>{{ json_quality.json_parse_fail }}</td></tr>
    <tr><th>Request Errors</th><td>{{ json_quality.request_errors }}</td></tr>
  </table>
  </section>

  <section>
  <h2>Overlap Buckets (Overall)</h2>
  <table>
    <tr>
      <th>Bucket</th>
      <th>Samples</th>
      <th>Avg Precision</th>
      <th>Avg Recall</th>
      <th>Avg F1</th>
    </tr>
    {% for bucket, stats in overlap_buckets.items() %}
    <tr>
      <td>{{ bucket }}</td>
      <td>{{ stats.count }}</td>
      <td>{{ '%.3f'|format(stats.avg_precision or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_recall or 0) }}</td>
      <td>{{ '%.3f'|format(stats.avg_f1 or 0) }}</td>
    </tr>
    {% endfor %}
  </table>
  <canvas id="overlapChart" height="120"></canvas>
  </section>

  <section>
  <h2>Dataset Overlap Buckets</h2>
  {% for ds_key, stats in dataset_stats.items() %}
    {% if stats.overlap_buckets %}
      <h3>{{ ds_key }}</h3>
  <table>
    <tr>
      <th>Bucket</th>
      <th>Samples</th>
      <th>Avg Precision</th>
      <th>Avg Recall</th>
      <th>Avg F1</th>
    </tr>
    {% for bucket, bstats in stats.overlap_buckets.items() %}
    <tr>
      <td>{{ bucket }}</td>
      <td>{{ bstats.count }}</td>
      <td>{{ '%.3f'|format(bstats.avg_precision or 0) }}</td>
      <td>{{ '%.3f'|format(bstats.avg_recall or 0) }}</td>
      <td>{{ '%.3f'|format(bstats.avg_f1 or 0) }}</td>
    </tr>
    {% endfor %}
  </table>
    {% endif %}
  {% endfor %}
  </section>

  <section>
  <h2>Samples</h2>
  <form method="get" style="margin-bottom: 1rem;">
    <label>Dataset:
      <select name="dataset">
        <option value="">All</option>
        {% for ds in dataset_options %}
          <option value="{{ ds }}" {% if ds == current_dataset %}selected{% endif %}>{{ ds }}</option>
        {% endfor %}
      </select>
    </label>
    <button type="submit">Apply</button>
  </form>
  <form method="get" action="{{ url_for('best_samples') }}" style="margin-bottom: 1rem;">
    <input type="hidden" name="run" value="{{ run.name }}">
    <label>Metric:
      <select name="metric">
        {% for opt in ['precision','recall','f1'] %}
          <option value="{{ opt }}">{{ opt }}</option>
        {% endfor %}
      </select>
    </label>
    <label>Top:
      <input type="number" name="top" value="20" min="1" style="width:4rem;" />
    </label>
    <label>Min delta:
      <input type="number" name="delta" value="0.1" step="0.01" style="width:5rem;" />
    </label>
    <span style="margin-left:1rem;">Stages:</span>
    {% for stage in stage_options %}
      <label style="margin-left:0.5rem;">
        <input type="checkbox" name="stage" value="{{ stage }}" checked />
        {{ stage }}
      </label>
    {% endfor %}
    <button type="submit">Find standout samples</button>
  </form>
  <table>
    <tr>
      <th>Sample ID</th>
      <th>Stage</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>Compare</th>
    </tr>
    {% for sample in samples %}
    <tr>
      <td>{{ sample.meta.sample_id }}</td>
      <td>{{ sample.meta.stage }}</td>
      <td>{{ '%.3f'|format(sample.precision or 0) }}</td>
      <td>{{ '%.3f'|format(sample.recall or 0) }}</td>
      <td>{{ '%.3f'|format(sample.f1 or 0) }}</td>
      <td><a href="{{ url_for('compare_sample', sample_id=sample.meta.sample_id) }}">View</a></td>
    </tr>
    {% endfor %}
  </table>
  </section>
</body>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
  const datasetChartData = {{ dataset_chart_data|tojson }};
  const stageChartData = {{ stage_chart_data|tojson }};
  const overlapChartData = {{ overlap_chart_data|tojson }};
  const difficultyChartData = {{ difficulty_chart_data|tojson }};

  if (datasetChartData.labels.length) {
    new Chart(document.getElementById('datasetChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: datasetChartData.labels,
        datasets: [
          { label: 'Precision', data: datasetChartData.precision, backgroundColor: 'rgba(75, 192, 192, 0.6)' },
          { label: 'Recall', data: datasetChartData.recall, backgroundColor: 'rgba(255, 205, 86, 0.6)' }
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
    });
  }

  if (stageChartData.labels.length) {
    new Chart(document.getElementById('stageChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: stageChartData.labels,
        datasets: [
          { label: 'Precision', data: stageChartData.precision, backgroundColor: 'rgba(54, 162, 235, 0.6)' },
          { label: 'Recall', data: stageChartData.recall, backgroundColor: 'rgba(255, 99, 132, 0.6)' }
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
    });
  }

  if (overlapChartData.labels.length) {
    new Chart(document.getElementById('overlapChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: overlapChartData.labels,
        datasets: [
          { label: 'Precision', data: overlapChartData.precision, backgroundColor: 'rgba(153, 102, 255, 0.6)' },
          { label: 'Recall', data: overlapChartData.recall, backgroundColor: 'rgba(255, 159, 64, 0.6)' }
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
    });
  }
  if (difficultyChartData.labels.length) {
    new Chart(document.getElementById('difficultyChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: difficultyChartData.labels,
        datasets: [
          { label: 'Precision', data: difficultyChartData.precision, backgroundColor: 'rgba(201, 203, 207, 0.6)' },
          { label: 'Recall', data: difficultyChartData.recall, backgroundColor: 'rgba(102, 166, 30, 0.6)' }
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
    });
  }

</script>
</html>
"""


COMPARE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Compare {{ sample_id }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    .model-block { border: 1px solid #ccc; padding: 1rem; margin-bottom: 1rem; }
    .stats { font-weight: bold; }
    img { max-width: 480px; margin-right: 1rem; }
    .images { display: flex; }
    pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>Sample {{ sample_id }}</h1>
  <p><a href="{{ url_for('index') }}">Back</a></p>
  <form method="get" style="margin-bottom:1rem;">
    <label>Select sample:
      <select name="sample_id">
        {% for sid in all_sample_ids %}
        <option value="{{ sid }}" {% if sid == sample_id %}selected{% endif %}>{{ sid }}</option>
        {% endfor %}
      </select>
    </label>
    <button type="submit">View</button>
    {% if prev_id %}
      <a href="{{ url_for('compare_sample', sample_id=prev_id) }}">&lsaquo; Prev</a>
    {% endif %}
    {% if next_id %}
      <a href="{{ url_for('compare_sample', sample_id=next_id) }}" style="margin-left:1rem;">Next &rsaquo;</a>
    {% endif %}
  </form>
  {% if runs %}
    {% for run in runs %}
      {% set rec = run.samples.get(sample_id) %}
      {% if rec %}
      <div class="model-block">
        <div class="stats">{{ run.name }} — Prec {{ '%.3f'|format(rec.precision or 0) }}, Recall {{ '%.3f'|format(rec.recall or 0) }}, F1 {{ '%.3f'|format(rec.f1 or 0) }}</div>
        <div class="images">
          {% if rec.meta.viewA_path %}
            <div>
              <div>View A</div>
              <img src="{{ url_for('serve_media', resource=rec.meta.viewA_path) }}" alt="viewA" />
            </div>
          {% endif %}
          {% if rec.meta.viewB_path %}
            <div>
              <div>View B</div>
              <img src="{{ url_for('serve_media', resource=rec.meta.viewB_path) }}" alt="viewB" />
            </div>
          {% endif %}
        </div>
        <p>Ground truth: <code>{{ rec.gt }}</code></p>
        <p>Prediction JSON: <code>{{ rec.prediction_json_raw }}</code></p>
        <pre>{{ rec.prediction }}</pre>
      </div>
      {% endif %}
    {% endfor %}
  {% else %}
    <p>No runs loaded.</p>
  {% endif %}
</body>
</html>
"""

BEST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Top Samples - {{ run_name }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
    th { background: #f5f5f5; }
    .model-block { border: 1px solid #ccc; padding: 1rem; margin-bottom: 1rem; }
    .stats { font-weight: bold; }
    img { max-width: 420px; margin-right: 1rem; }
    .images { display: flex; }
    pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Top Samples for {{ run_name }} (metric: {{ metric }})</h1>
  <p><a href="{{ url_for('index') }}">Back</a></p>
  <form method="get" style="margin-bottom:1rem;">
    <input type="hidden" name="run" value="{{ run_name }}">
    <label>Metric:
      <select name="metric">
        {% for opt in ['precision','recall','f1'] %}
          <option value="{{ opt }}" {% if opt == metric %}selected{% endif %}>{{ opt }}</option>
        {% endfor %}
      </select>
    </label>
    <label>Top:
      <input type="number" name="top" value="{{ top_k }}" min="1" style="width:4rem;" />
    </label>
    <label>Min delta:
      <input type="number" name="delta" value="{{ '%.3f'|format(min_delta) }}" step="0.01" style="width:5rem;" />
    </label>
    <span style="margin-left:1rem;">Stages:</span>
    {% set active_stages = selected_stages if selected_stages else stage_options %}
    {% for stage in stage_options %}
      <label style="margin-left:0.5rem;">
        <input type="checkbox" name="stage" value="{{ stage }}" {% if stage in active_stages %}checked{% endif %} />
        {{ stage }}
      </label>
    {% endfor %}
    <button type="submit">Update</button>
  </form>
  {% if rows %}
  <table>
    <tr>
      <th>Sample ID</th>
      <th>Stage</th>
      <th>{{ metric }} ({{ run_name }})</th>
      <th>Next Best</th>
      <th>Delta</th>
      <th>Compare</th>
      <th>Jump</th>
    </tr>
    {% for row in rows %}
    <tr>
      <td>{{ row.sample_id }}</td>
      <td>{{ row.sample.meta.stage or 'N/A' }}</td>
      <td>{{ '%.3f'|format(row.value) }}</td>
      <td>{{ '%.3f'|format(row.next_best) }}</td>
      <td>{{ '%.3f'|format(row.delta) }}</td>
      <td><a href="{{ url_for('compare_sample', sample_id=row.sample_id) }}">View runs</a></td>
      <td><a href="#sample-{{ row.sample_id }}">scroll</a></td>
    </tr>
    {% endfor %}
  </table>

  {% for row in rows %}
  <div class="model-block" id="sample-{{ row.sample_id }}">
    <h2>{{ row.sample_id }}</h2>
    <div class="stats">{{ run_name }} &mdash; Prec {{ '%.3f'|format(row.sample.precision or 0) }}, Recall {{ '%.3f'|format(row.sample.recall or 0) }}, F1 {{ '%.3f'|format(row.sample.f1 or 0) }}</div>
    <div class="images">
      {% if row.sample.meta.viewA_path %}
        <div>
          <div>View A</div>
          <img src="{{ url_for('serve_media', resource=row.sample.meta.viewA_path) }}" alt="viewA" />
        </div>
      {% endif %}
      {% if row.sample.meta.viewB_path %}
        <div>
          <div>View B</div>
          <img src="{{ url_for('serve_media', resource=row.sample.meta.viewB_path) }}" alt="viewB" />
        </div>
      {% endif %}
    </div>
    <p>Ground truth: <code>{{ row.sample.gt }}</code></p>
    <p>Prediction JSON: <code>{{ row.sample.prediction_json_raw }}</code></p>
    <pre>{{ row.sample.prediction }}</pre>
  </div>
  {% endfor %}
  {% else %}
    <p>No samples satisfy the filter.</p>
  {% endif %}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize evaluation results in a web UI.")
    parser.add_argument("--results_dir", type=str, required=True, help="Folder containing multiple evaluation runs.")
    parser.add_argument("--testset_root", type=str, required=True, help="Root directory of the annotated testset.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def load_run(run_path: Path) -> dict | None:
    summary_path = run_path / "summary.json"
    samples_path = run_path / "samples.jsonl"
    if not summary_path.exists() or not samples_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples: Dict[str, dict] = {}
    fallback_dataset_stats: Dict[str, dict] = {}
    with samples_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            gt = record.get("gt") or {}
            parsed = record.get("prediction_parsed") if isinstance(record.get("prediction_parsed"), dict) else None
            precision, recall, f1, _ = compare_mappings(gt, parsed)
            record["precision"] = precision
            record["recall"] = recall
            record["f1"] = f1
            sample_id = record.get("meta", {}).get("sample_id")
            if sample_id:
                samples[sample_id] = record
                ds_key = record.get("meta", {}).get("dataset_key") or "unknown"
                stats = fallback_dataset_stats.setdefault(
                    ds_key, {"count": 0, "acc_sum": 0.0, "recall_sum": 0.0, "f1_sum": 0.0}
                )
                stats["count"] += 1
                stats["acc_sum"] += record.get("precision", record.get("accuracy", 0.0))
                stats["recall_sum"] += record.get("recall", 0.0)
                stats["f1_sum"] += record.get("f1", 0.0)

    for ds_key, stats in fallback_dataset_stats.items():
        count = max(stats["count"], 1)
        stats["avg_precision"] = stats["acc_sum"] / count
        stats["avg_recall"] = stats["recall_sum"] / count
        stats["avg_f1"] = stats["f1_sum"] / count
        stats.setdefault("avg_overlap", None)
        stats.setdefault("avg_overlap_rank", None)
        stats.setdefault("overlap_rank_min", None)
        stats.setdefault("overlap_rank_max", None)
        stats.setdefault("overlap_buckets", {})
        stats.setdefault("difficulties", {})
        stats.pop("acc_sum", None)
        stats.pop("recall_sum", None)
        stats.pop("f1_sum", None)

    dataset_stats = summary.get("dataset_metrics") or fallback_dataset_stats
    dataset_options = sorted(dataset_stats.keys())
    stage_options = sorted({*summary.get("stage_metrics", {}).keys(), *DEFAULT_STAGE_OPTIONS})

    return {
        "name": run_path.name,
        "summary": summary,
        "samples": samples,
        "dataset_stats": dataset_stats,
        "dataset_options": dataset_options,
        "stage_options": stage_options,
        "stage_metrics": summary.get("stage_metrics", {}),
        "json_quality": summary.get("json_quality", {}),
        "overlap_buckets": summary.get("overlap_buckets", {}),
        "difficulty_metrics": summary.get("difficulty_metrics", {}),
    }


def create_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__)
    results_dir = Path(args.results_dir)
    testset_root = Path(args.testset_root)

    runs: List[dict] = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        run = load_run(subdir)
        if run:
            runs.append(run)

    def to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [to_ns(item) for item in obj]
        return obj


    @app.route("/")
    def index() -> str:
        return render_template_string(INDEX_TEMPLATE, runs=runs)

    @app.route("/runs/<run_name>")
    def list_samples(run_name: str) -> str:
        run = next((r for r in runs if r["name"] == run_name), None)
        if not run:
            abort(404)
        dataset_filter = request.args.get("dataset") or ""
        samples = list(run["samples"].values())
        if dataset_filter:
            samples = [s for s in samples if s.get("meta", {}).get("dataset_key") == dataset_filter]
        samples.sort(key=lambda item: item.get("meta", {}).get("sample_id"))
        samples = samples[:500]

        dataset_stats = run.get("dataset_stats", {})
        stage_stats = run.get("stage_metrics", {})
        overlap_buckets = run.get("overlap_buckets", {})
        difficulty_stats = run.get("difficulty_metrics", {})
        json_quality = {
            "total": 0,
            "json_success": 0,
            "json_absent": 0,
            "json_parse_fail": 0,
            "request_errors": 0,
            **(run.get("json_quality", {}) or {}),
        }

        dataset_chart_data = {
            "labels": list(dataset_stats.keys()),
            "precision": [stats.get("avg_precision", 0.0) for stats in dataset_stats.values()],
            "recall": [stats.get("avg_recall", 0.0) for stats in dataset_stats.values()],
        }
        stage_chart_data = {
            "labels": list(stage_stats.keys()),
            "precision": [stats.get("avg_precision", 0.0) for stats in stage_stats.values()],
            "recall": [stats.get("avg_recall", 0.0) for stats in stage_stats.values()],
        }
        difficulty_chart_data = {
            "labels": list(difficulty_stats.keys()),
            "precision": [stats.get("avg_precision", 0.0) for stats in difficulty_stats.values()],
            "recall": [stats.get("avg_recall", 0.0) for stats in difficulty_stats.values()],
        }
        overlap_chart_data = {
            "labels": list(overlap_buckets.keys()),
            "precision": [stats.get("avg_precision", 0.0) for stats in overlap_buckets.values()],
            "recall": [stats.get("avg_recall", 0.0) for stats in overlap_buckets.values()],
        }

        stage_options = run.get("stage_options", list(DEFAULT_STAGE_OPTIONS))

        return render_template_string(
            SAMPLES_TEMPLATE,
            run=run,
            samples=samples,
            current_dataset=dataset_filter,
            dataset_stats=dataset_stats,
            dataset_options=run.get("dataset_options", []),
            stage_options=stage_options,
            stage_stats=stage_stats,
            difficulty_stats=difficulty_stats,
            overlap_buckets=overlap_buckets,
            dataset_chart_data=dataset_chart_data,
            stage_chart_data=stage_chart_data,
            overlap_chart_data=overlap_chart_data,
            difficulty_chart_data=difficulty_chart_data,
            json_quality=json_quality,
        )

    @app.route("/compare")
    def compare_sample() -> str:
        sample_id = request.args.get("sample_id")
        all_sample_ids = sorted({sid for run in runs for sid in run["samples"].keys()})
        if not all_sample_ids:
            abort(404, "No samples available")
        if not sample_id or sample_id not in all_sample_ids:
            sample_id = all_sample_ids[0]
        idx = all_sample_ids.index(sample_id)
        prev_id = all_sample_ids[idx - 1] if idx > 0 else None
        next_id = all_sample_ids[idx + 1] if idx + 1 < len(all_sample_ids) else None
        return render_template_string(
            COMPARE_TEMPLATE,
            sample_id=sample_id,
            runs=runs,
            all_sample_ids=all_sample_ids,
            prev_id=prev_id,
            next_id=next_id,
        )

    @app.route("/media/<path:resource>")
    def serve_media(resource: str):
        safe_path = Path(resource).as_posix()
        return send_from_directory(testset_root, safe_path)

    @app.route("/best_samples")
    def best_samples() -> str:
        target_run_name = request.args.get("run") or (runs[0]["name"] if runs else None)
        metric = request.args.get("metric", "f1").lower()
        top_k = int(request.args.get("top", 20))
        min_delta = float(request.args.get("delta", 0.0))
        if not target_run_name:
            abort(404, "No runs available")
        run = next((r for r in runs if r["name"] == target_run_name), None)
        if not run:
            abort(404, f"Run {target_run_name} not found")
        stage_options = run.get("stage_options", list(DEFAULT_STAGE_OPTIONS))
        requested_stages = [stage for stage in request.args.getlist("stage") if stage]
        if requested_stages:
            stage_filter = set(requested_stages)
        else:
            stage_filter = set(stage_options)

        valid_metrics = {"precision", "recall", "f1"}
        if metric not in valid_metrics:
            abort(400, f"metric must be one of {valid_metrics}")

        rows = []
        for sample_id, sample in run["samples"].items():
            value = sample.get(metric)
            if value is None:
                continue
            stage = sample.get("meta", {}).get("stage")
            if stage_filter and stage not in stage_filter:
                continue
            next_best = None
            for other in runs:
                if other["name"] == run["name"]:
                    continue
                other_sample = other["samples"].get(sample_id)
                if not other_sample:
                    continue
                other_value = other_sample.get(metric)
                if other_value is None:
                    continue
                if next_best is None or other_value > next_best:
                    next_best = other_value
            next_best = next_best or 0.0
            delta = value - next_best
            if delta >= min_delta:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "value": value,
                        "next_best": next_best,
                        "delta": delta,
                        "sample": sample,
                    }
                )

        rows.sort(key=lambda item: item["delta"], reverse=True)
        rows = rows[:top_k]
        rows = [
            SimpleNamespace(
                sample=to_ns(row["sample"]),
                sample_id=row["sample_id"],
                value=row["value"],
                next_best=row["next_best"],
                delta=row["delta"],
            )
            for row in rows
        ]
        return render_template_string(
            BEST_TEMPLATE,
            run_name=run["name"],
            metric=metric,
            rows=rows,
            top_k=top_k,
            min_delta=min_delta,
            stage_options=stage_options,
            selected_stages=sorted(stage_filter) if requested_stages else [],
        )

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
