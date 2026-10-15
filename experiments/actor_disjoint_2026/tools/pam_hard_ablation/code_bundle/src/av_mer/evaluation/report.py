"""Assemble the cross-run result table (protocol §9) from saved metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


METRIC_KEYS = ("macro_f1", "uar", "accuracy", "weighted_f1")
PROTOCOL_KEYS = ("protocol_id", "split_checksum")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_run(run_dir: str | Path) -> dict[str, Any]:
    """Collect one run's identity + val/test metrics into a flat row."""

    run_dir = Path(run_dir)
    row: dict[str, Any] = {"run": run_dir.name}
    cfg_path = run_dir / "resolved_config.json"
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        row["baseline"] = cfg.get("baseline")
        row["soft_labels"] = bool(cfg.get("loss", {}).get("soft_labels_enabled", False))
        data_config = cfg.get("data", {})
        if isinstance(data_config, dict) and data_config.get("protocol_id"):
            row["protocol_id"] = str(data_config["protocol_id"])
    reproducibility_path = run_dir / "reproducibility.json"
    if reproducibility_path.exists():
        reproducibility = _load_json(reproducibility_path)
        if reproducibility.get("split_manifest_checksum"):
            row["split_checksum"] = str(reproducibility["split_manifest_checksum"])
    for split, fname in (("val", "metrics.json"), ("test", "test_metrics.json")):
        path = run_dir / fname
        if not path.exists():
            continue
        m = _load_json(path)
        for key in METRIC_KEYS:
            if key in m:
                row[f"{split}_{key}"] = round(float(m[key]), 4)
        if isinstance(m.get("ece"), dict):
            row[f"{split}_ece"] = round(float(m["ece"]["ece"]), 4)
    return row


def build_table(run_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Build a list of rows for the given run directories."""

    return [collect_run(d) for d in run_dirs]


def validate_comparable_runs(rows: list[dict[str, Any]]) -> None:
    """Reject a controlled table that mixes or omits protocol provenance."""

    if not rows:
        return
    for key in PROTOCOL_KEYS:
        missing = [str(row.get("run", "<unknown>")) for row in rows if not row.get(key)]
        if missing:
            raise ValueError(f"Runs missing {key}: {missing}.")
        values = {str(row[key]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"Controlled comparison mixes {key} values: {sorted(values)}.")


def to_markdown(rows: list[dict[str, Any]]) -> str:
    """Render rows as a Markdown table with a stable column order."""

    if not rows:
        return "(no runs)"
    columns = ["run", "protocol_id", "split_checksum", "baseline", "soft_labels"]
    for split in ("test", "val"):
        for key in (*METRIC_KEYS, "ece"):
            columns.append(f"{split}_{key}")
    columns = [c for c in columns if any(c in r for r in rows)]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def write_report(
    run_dirs: Iterable[str | Path],
    output_path: str | Path,
    *,
    require_same_protocol: bool = True,
) -> list[dict[str, Any]]:
    """Collect runs and write a protocol-safe Markdown summary table."""

    rows = build_table(run_dirs)
    if require_same_protocol:
        validate_comparable_runs(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_markdown(rows) + "\n", encoding="utf-8")
    return rows
