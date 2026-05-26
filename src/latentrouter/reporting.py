from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from latentrouter.io import read_json, write_json


def _format_metric(value: float | int | bool | str | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if value == float("inf"):
        return "unreached"
    if value != value:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _save_overall_plot(
    run_dir: Path,
    frontier: pd.DataFrame,
    oracle: pd.DataFrame,
    single_model: pd.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 6))
    x_column = "avg_cost" if "avg_cost" in frontier.columns else "mean_cost"
    y_column = "avg_accuracy" if "avg_accuracy" in frontier.columns else "mean_quality"
    single_x = "avg_cost" if "avg_cost" in single_model.columns else "cost"
    single_y = "avg_accuracy" if "avg_accuracy" in single_model.columns else "quality"
    plt.plot(frontier[x_column], frontier[y_column], label="router", linewidth=2)
    if not oracle.empty:
        oracle_x = "avg_cost" if "avg_cost" in oracle.columns else "mean_cost"
        oracle_y = "avg_accuracy" if "avg_accuracy" in oracle.columns else "mean_quality"
        plt.plot(oracle[oracle_x], oracle[oracle_y], label="oracle", linestyle="--")
    if not single_model.empty:
        plt.scatter(single_model[single_x], single_model[single_y], label="single models", alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "frontier.png", dpi=150)
    plt.close()


def _save_slice_plot(run_dir: Path, routes: pd.DataFrame, field: str) -> None:
    if field not in routes.columns:
        return
    plt.figure(figsize=(10, 6))
    for value, frame in routes.groupby(field, dropna=False):
        frontier = frame.groupby("lambda_value", as_index=False).agg(cost=("cost", "mean"), quality=("correctness", "mean"))
        plt.plot(frontier["cost"], frontier["quality"], label=str(value))
    plt.xlabel("Mean cost")
    plt.ylabel("Mean quality")
    plt.title(f"Frontier by {field}")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(run_dir / f"frontier_{field}.png", dpi=150)
    plt.close()


def generate_report(run_dir: str | Path, aggregate_by: list[str]) -> Path:
    run_dir = Path(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    frontier = pd.read_csv(run_dir / "frontier.csv")
    single_model = pd.read_csv(run_dir / "single_model_frontier.csv")
    oracle = pd.read_csv(run_dir / "oracle_frontier.csv")
    routes = pd.read_parquet(run_dir / "routes.parquet")
    slice_metrics_path = run_dir / "slice_metrics.csv"
    slice_metrics = pd.read_csv(slice_metrics_path) if slice_metrics_path.exists() else pd.DataFrame()

    _save_overall_plot(
        run_dir,
        frontier,
        oracle,
        single_model,
        x_label=str(metrics.get("plot_x_label", "Mean cost")),
        y_label=str(metrics.get("plot_y_label", "Mean quality")),
        title=str(metrics.get("plot_title", "Cost-quality frontier")),
    )
    for field in aggregate_by:
        _save_slice_plot(run_dir, routes, field)

    summary_path = run_dir / "summary.md"
    primary_metric = str(metrics.get("primary_metric", "nAUC"))
    summary_metric_order = list(metrics.get("summary_metric_order", ["nAUC", "Ps", "QNC"]))
    slice_metric_order = list(metrics.get("slice_metric_order", [primary_metric]))
    top_slices = (
        slice_metrics.sort_values("primary_metric_value", ascending=False).head(10)
        if not slice_metrics.empty and "primary_metric_value" in slice_metrics.columns
        else pd.DataFrame()
    )
    summary_lines = [
        f"# {metrics['router_name']} evaluation",
        "",
        f"- benchmark: {metrics.get('benchmark_id', 'unknown')}",
        f"- protocol: {metrics.get('protocol_id', 'unknown')}",
        "## Overall metrics",
        f"- primary metric: {primary_metric} = {_format_metric(metrics.get('primary_metric_value'))}",
    ]
    for metric_name in summary_metric_order:
        summary_lines.append(f"- {metric_name}: {_format_metric(metrics.get(metric_name))}")
    summary_lines.extend(
        [
            f"- samples: {metrics['sample_count']}",
            f"- models: {metrics['model_count']}",
            "",
            "## Artifacts",
            "- `frontier.csv`",
            "- `routes.parquet`",
            "- `single_model_frontier.csv`",
            "- `oracle_frontier.csv`",
            "- `frontier.png`",
        ]
    )
    if not top_slices.empty:
        summary_lines.extend(["", "## Best slices"])
        for _, row in top_slices.iterrows():
            slice_bits = [f"{metric_name}={_format_metric(row.get(metric_name))}" for metric_name in slice_metric_order]
            summary_lines.append(f"- {row['slice_field']}={row['slice_value']}: " + ", ".join(slice_bits))

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_json(run_dir / "report_manifest.json", {"aggregate_by": aggregate_by})
    return summary_path
