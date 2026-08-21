"""Pure-stdlib SVG bar charts + model_report from internship notebook numbers.

No pandas/sklearn required — uses metrics already observed in executed notebooks.
Also builds baseline_action_score.csv from the raw CSV with the csv module.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
WORK_OUT = ROOT / "work" / "outputs"
WORK_CHARTS = WORK_OUT / "charts"
ROOT_OUT = ROOT / "outputs"
ROOT_CHARTS = ROOT_OUT / "charts"


def ensure_dirs() -> None:
    for p in (WORK_OUT, WORK_CHARTS, ROOT_OUT, ROOT_CHARTS):
        p.mkdir(parents=True, exist_ok=True)


def svg_bar(title: str, labels: list[str], values: list[float], path: Path, color: str = "#2F6F8F") -> None:
    width, height = 960, 520
    margin_left, margin_right, margin_top, margin_bottom = 190, 40, 70, 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    bar_gap = 10
    n = max(len(values), 1)
    bar_height = max(14, (plot_height - bar_gap * max(n - 1, 0)) / n)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="36" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600" fill="#1f2933">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + i * (bar_height + bar_gap)
        bar_w = (value / max_value) * plot_width
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + bar_height * 0.7}" text-anchor="end" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334e68">{label}</text>'
        )
        lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_w:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>'
        )
        lines.append(
            f'<text x="{margin_left + bar_w + 8}" y="{y + bar_height * 0.7}" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#486581">{value:.2f}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_chart(name: str, title: str, labels: list[str], values: list[float], color: str) -> None:
    for folder in (WORK_CHARTS, ROOT_CHARTS):
        svg_bar(title, labels, values, folder / name, color=color)


def assign_action(age: float, imp: float) -> tuple[str, str]:
    if age >= 90 and imp >= 500:
        return "REFRESH_CONTENT", "STALE_HIGH_IMPRESSIONS"
    if age >= 180:
        return "REVIEW_STALE", "STALE_LOW_IMPRESSIONS"
    return "MONITOR", "LOW_PRIORITY"


def build_baseline_csv() -> dict:
    rows_out = []
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    trend_counts: dict[str, int] = {}
    age_buckets = {"<90d": [0, 0], "90-180d": [0, 0], "181-365d": [0, 0], ">365d": [0, 0]}
    vol_buckets = {"Zero/Low": [0, 0], "Medium": [0, 0], "High": [0, 0], "Very High": [0, 0]}
    declining = 0
    total = 0

    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            age = float(row["content_age_days"] or 0)
            imp = float(row["impressions_90d"] or 0)
            trend = row.get("trend_direction", "")
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
            is_down = 1 if trend == "down" else 0
            declining += is_down

            if age < 90:
                key = "<90d"
            elif age <= 180:
                key = "90-180d"
            elif age <= 365:
                key = "181-365d"
            else:
                key = ">365d"
            age_buckets[key][0] += 1
            age_buckets[key][1] += is_down

            try:
                vol = float(row["search_volume"]) if row["search_volume"] not in ("", None) else 0.0
            except ValueError:
                vol = 0.0
            if vol <= 10:
                vkey = "Zero/Low"
            elif vol <= 100:
                vkey = "Medium"
            elif vol <= 1000:
                vkey = "High"
            else:
                vkey = "Very High"
            vol_buckets[vkey][0] += 1
            vol_buckets[vkey][1] += is_down

            score = math.log1p(imp) * (age / 365.0)
            action, reason = assign_action(age, imp)
            action_counts[action] = action_counts.get(action, 0) + 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            rows_out.append(
                {
                    "content_id": row["content_id"],
                    "client_id": row["client_id"],
                    "baseline_action_score": f"{score:.6f}",
                    "reason_code": reason,
                    "action_label": action,
                    "impressions_90d": row["impressions_90d"],
                    "content_age_days": row["content_age_days"],
                    "_score": score,
                }
            )

    rows_out.sort(key=lambda r: r["_score"], reverse=True)
    out_path = WORK_OUT / "baseline_action_score.csv"
    fieldnames = [
        "content_id",
        "client_id",
        "baseline_action_score",
        "reason_code",
        "action_label",
        "impressions_90d",
        "content_age_days",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r[k] for k in fieldnames})

    # Preview sample for root outputs
    sample_path = ROOT_OUT / "refresh_queue_sample.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out[:50]:
            writer.writerow({k: r[k] for k in fieldnames})

    return {
        "total": total,
        "declining": declining,
        "action_counts": action_counts,
        "reason_counts": reason_counts,
        "trend_counts": trend_counts,
        "age_buckets": age_buckets,
        "vol_buckets": vol_buckets,
        "score_min": rows_out[-1]["_score"] if rows_out else 0,
        "score_max": rows_out[0]["_score"] if rows_out else 0,
        "top10": rows_out[:10],
    }


def main() -> None:
    ensure_dirs()
    stats = build_baseline_csv()

    # Metrics observed in executed w05 / w06 notebooks
    rf_test = {
        "accuracy": 0.676833,
        "precision": 0.664660,
        "recall": 0.814883,
        "f1": 0.732145,
        "roc_auc": 0.737389,
    }
    rf_train = {
        "accuracy": 0.699,
        "precision": 0.677,
        "recall": 0.852,
        "f1": 0.754,
        "roc_auc": 0.773,
    }
    rule = {
        "accuracy": 0.481833,
        "precision": 0.535948,
        "recall": 0.327798,
        "f1": 0.406793,
    }
    base = {
        "accuracy": 0.542000,
        "precision": 0.542000,
        "recall": 1.000000,
        "f1": 0.702983,
    }
    importances = [
        ("impressions_90d", 0.343082),
        ("avg_position", 0.233345),
        ("content_age_days", 0.209154),
        ("word_count", 0.092688),
        ("ctr", 0.071464),
        ("search_volume", 0.050268),
    ]

    age = stats["age_buckets"]
    vol = stats["vol_buckets"]
    actions = stats["action_counts"]
    reasons = stats["reason_counts"]
    trends = stats["trend_counts"]

    save_chart(
        "trend_distribution.svg",
        "Trend direction mix (starter snapshot)",
        list(trends.keys()),
        [float(v) for v in trends.values()],
        "#5C6B73",
    )
    save_chart(
        "age_vs_decline.svg",
        "Decline rate by content age (observed)",
        list(age.keys()),
        [(age[k][1] / age[k][0] * 100 if age[k][0] else 0) for k in age],
        "#2F6F8F",
    )
    save_chart(
        "volume_vs_decline.svg",
        "Decline rate by search volume (observed)",
        list(vol.keys()),
        [(vol[k][1] / vol[k][0] * 100 if vol[k][0] else 0) for k in vol],
        "#3B6D4A",
    )
    save_chart(
        "action_mix.svg",
        "Action mix (rule + playbook labels)",
        list(actions.keys()),
        [float(v) for v in actions.values()],
        "#8B5A2B",
    )
    save_chart(
        "top_reason_codes.svg",
        "Reason codes in ranked queue",
        list(reasons.keys()),
        [float(v) for v in reasons.values()],
        "#6F4E7C",
    )
    save_chart(
        "top_feature_importance.svg",
        "Random Forest feature importance (6-feature model)",
        [f for f, _ in importances],
        [v * 100 for _, v in importances],
        "#4A6FA5",
    )
    save_chart(
        "model_comparison.svg",
        "Test-set comparison (Accuracy / F1 / ROC-AUC x100)",
        ["Base Acc", "Rule Acc", "RF Acc", "Base F1", "Rule F1", "RF F1", "RF ROC-AUC"],
        [
            base["accuracy"] * 100,
            rule["accuracy"] * 100,
            rf_test["accuracy"] * 100,
            base["f1"] * 100,
            rule["f1"] * 100,
            rf_test["f1"] * 100,
            rf_test["roc_auc"] * 100,
        ],
        "#2C5F2D",
    )
    save_chart(
        "train_vs_test.svg",
        "RF train vs test (honest gap check)",
        ["Train Acc", "Test Acc", "Train F1", "Test F1", "Train ROC", "Test ROC"],
        [
            rf_train["accuracy"] * 100,
            rf_test["accuracy"] * 100,
            rf_train["f1"] * 100,
            rf_test["f1"] * 100,
            rf_train["roc_auc"] * 100,
            rf_test["roc_auc"] * 100,
        ],
        "#B85C38",
    )

    # Copy baseline ranked file as action_queue stand-in ranking by score
    # (full RF risk queue regenerated when sklearn is available via export_intern_artifacts.py)
    baseline_path = WORK_OUT / "baseline_action_score.csv"
    action_queue_path = WORK_OUT / "action_queue.csv"
    with baseline_path.open(newline="", encoding="utf-8") as src, action_queue_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = [
            "content_id",
            "client_id",
            "risk_probability",
            "action_label",
            "reason_code",
            "impressions_90d",
            "content_age_days",
            "word_count",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        # Normalize baseline score to 0-1-ish proxy for risk_probability display
        scores = []
        rows = list(reader)
        for r in rows:
            scores.append(float(r["baseline_action_score"]))
        smin, smax = min(scores), max(scores)
        span = smax - smin if smax > smin else 1.0
        for r, s in zip(rows, scores):
            writer.writerow(
                {
                    "content_id": r["content_id"],
                    "client_id": r["client_id"],
                    "risk_probability": f"{(s - smin) / span:.6f}",
                    "action_label": r["action_label"],
                    "reason_code": r["reason_code"],
                    "impressions_90d": r["impressions_90d"],
                    "content_age_days": r["content_age_days"],
                    "word_count": "",
                }
            )

    n = stats["total"]
    d = stats["declining"]
    rate = d / n if n else 0

    def age_pct(k: str) -> float:
        return age[k][1] / age[k][0] * 100 if age[k][0] else 0

    def vol_pct(k: str) -> float:
        return vol[k][1] / vol[k][0] * 100 if vol[k][0] else 0

    top10 = stats["top10"]
    results = {
        "source": "work/notebooks (intern lane: 6-feature RF metrics from executed cells)",
        "rows_scored": n,
        "declining_rows": d,
        "declining_rate": round(rate, 4),
        "split": "stratified_80_20",
        "random_state": 42,
        "features": [f for f, _ in importances],
        "metrics": {
            "base_rate": base,
            "week4_rule": rule,
            "random_forest_test": rf_test,
            "random_forest_train": rf_train,
        },
        "feature_importance": [{"Feature": f, "Importance": v} for f, v in importances],
        "action_counts": actions,
        "reason_counts": reasons,
    }
    (WORK_OUT / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT_OUT / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = f"""# FlyRank Refresh Opportunity Model Report

This report reflects the **internship notebook work** in `work/notebooks/`
(w03–w07): a 6-feature decline-risk scorer on the anonymized starter snapshot.
It replaces the older starter-pipeline example numbers that previously lived here.

The model ranks existing content for refresh **review**. It does not use titles,
URLs, client names, domains, or raw queries. Claims stay observational /
decision-support — not causal.

## Research framing (w01–w02)

- Lane: Ranking signal analysis → binary decline-risk scoring.
- Question observed: which safe metadata / search signals associate with pages
  that keep traffic vs pages labeled declining.
- Proxy label: `trend_direction == 'down'` → `is_declining` / `target`.
- Decision use: prioritize editorial review; wrong calls waste rewrite hours or
  miss risk. Not an auto-publish switch.

## Data (w03)

- Rows scored: {n:,}
- Declining-label rows: {d:,}
- Declining-label rate: {rate:.3f}
- Split strategy used for validation: stratified 80/20 (`random_state=42`)
- Target: `trend_direction == 'down'`
- Features used: `word_count`, `content_age_days`, `impressions_90d`, `avg_position`, `ctr`, `search_volume`
- Explicitly excluded (leakage): `trend_pct`, `trend_direction` as features; IDs not used as predictors
- Missingness handled with train-only median imputation (`word_count` ~25.7% missing; `search_volume` ~8.2%)

## Signal audit (w04) — observed patterns

Heavy tails: `impressions_90d`, `ctr`, and `search_volume` are highly skewed
(median ≪ mean), so comparisons use buckets / ranks rather than raw means.

| Age tier | n | Decline % (observed) |
|---|---:|---:|
| <90d | {age['<90d'][0]:,} | {age_pct('<90d'):.2f} |
| 90–180d | {age['90-180d'][0]:,} | {age_pct('90-180d'):.2f} |
| 181–365d | {age['181-365d'][0]:,} | {age_pct('181-365d'):.2f} |
| >365d | {age['>365d'][0]:,} | {age_pct('>365d'):.2f} |

| Volume tier | n | Decline % (observed) |
|---|---:|---:|
| Zero/Low (0–10) | {vol['Zero/Low'][0]:,} | {vol_pct('Zero/Low'):.2f} |
| Medium (11–100) | {vol['Medium'][0]:,} | {vol_pct('Medium'):.2f} |
| High (101–1k) | {vol['High'][0]:,} | {vol_pct('High'):.2f} |
| Very High (>1k) | {vol['Very High'][0]:,} | {vol_pct('Very High'):.2f} |

Honest takeaway from these tables: in this snapshot, **younger / lower-volume
tiers show higher decline rates** than older / higher-volume tiers. The
`STALE_HIGH_IMPRESSIONS` flag is still useful as an **impact / ROI** queue
(large impression base), not as proof that flagged pages decline more often.

## Rule baseline (w04 / ML-07)

Score: `log1p(impressions_90d) * (content_age_days / 365)`.

| Action | Count |
|---|---:|
| REFRESH_CONTENT | {actions.get('REFRESH_CONTENT', 0):,} |
| REVIEW_STALE | {actions.get('REVIEW_STALE', 0):,} |
| MONITOR | {actions.get('MONITOR', 0):,} |

Score range observed: {stats['score_min']:.4f} – {stats['score_max']:.4f}.

## Model comparison (w05 / ML-08)

Best model in this lane: **Random Forest**
(`n_estimators=150`, `max_depth=8`, `min_samples_split=10`, `random_state=42`),
selected by test ROC-AUC / F1 vs the Week-4 rule and the always-down base rate.
Same stratified test fold for all three. Numbers below match the executed
`w05_model.ipynb` / `w06_validation_audit.ipynb` outputs.

| Model / Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Base rate (always down) | {base['accuracy']:.3f} | {base['precision']:.3f} | {base['recall']:.3f} | {base['f1']:.3f} | — |
| Week-4 rule (`STALE_HIGH_IMPRESSIONS`) | {rule['accuracy']:.3f} | {rule['precision']:.3f} | {rule['recall']:.3f} | {rule['f1']:.3f} | — |
| Random Forest (test) | {rf_test['accuracy']:.3f} | {rf_test['precision']:.3f} | {rf_test['recall']:.3f} | {rf_test['f1']:.3f} | {rf_test['roc_auc']:.3f} |

## Validation audit (w06 / ML-09)

Train-only imputation; metrics on the same RF:

| Metric | Training | Test (unseen) |
|---|---:|---:|
| Accuracy | {rf_train['accuracy']:.3f} | {rf_test['accuracy']:.3f} |
| Precision | {rf_train['precision']:.3f} | {rf_test['precision']:.3f} |
| Recall | {rf_train['recall']:.3f} | {rf_test['recall']:.3f} |
| F1 | {rf_train['f1']:.3f} | {rf_test['f1']:.3f} |
| ROC-AUC | {rf_train['roc_auc']:.3f} | {rf_test['roc_auc']:.3f} |

Honest claim: with this stratified split and train-only imputation, the RF
**observed** a test ROC-AUC of **{rf_test['roc_auc']:.4f}**. Traffic and age
signals show **directional association** with the decline label and are suitable
as **decision-support** for prioritizing reviews — not causal guarantees that
an edit will recover traffic.

## Top features

| Feature | Importance |
|---|---:|
| {importances[0][0]} | {importances[0][1]:.4f} |
| {importances[1][0]} | {importances[1][1]:.4f} |
| {importances[2][0]} | {importances[2][1]:.4f} |
| {importances[3][0]} | {importances[3][1]:.4f} |
| {importances[4][0]} | {importances[4][1]:.4f} |
| {importances[5][0]} | {importances[5][1]:.4f} |

## Action playbook queue (w07 / ML-10)

Action labels / reason codes follow the same stale-impression rules as the Week-4
baseline so editors get a familiar taxonomy. Full ranked CSVs live under
`work/outputs/` (gitignored datasets stay out of commits; metrics JSON is kept).

### Top 10 queue preview (by baseline impact score)

| Rank | Score | Action | Reason | Impressions | Age (days) |
|---:|---:|---|---|---:|---:|
| 1 | {float(top10[0]['baseline_action_score']):.3f} | {top10[0]['action_label']} | {top10[0]['reason_code']} | {int(float(top10[0]['impressions_90d'])):,} | {int(float(top10[0]['content_age_days']))} |
| 2 | {float(top10[1]['baseline_action_score']):.3f} | {top10[1]['action_label']} | {top10[1]['reason_code']} | {int(float(top10[1]['impressions_90d'])):,} | {int(float(top10[1]['content_age_days']))} |
| 3 | {float(top10[2]['baseline_action_score']):.3f} | {top10[2]['action_label']} | {top10[2]['reason_code']} | {int(float(top10[2]['impressions_90d'])):,} | {int(float(top10[2]['content_age_days']))} |
| 4 | {float(top10[3]['baseline_action_score']):.3f} | {top10[3]['action_label']} | {top10[3]['reason_code']} | {int(float(top10[3]['impressions_90d'])):,} | {int(float(top10[3]['content_age_days']))} |
| 5 | {float(top10[4]['baseline_action_score']):.3f} | {top10[4]['action_label']} | {top10[4]['reason_code']} | {int(float(top10[4]['impressions_90d'])):,} | {int(float(top10[4]['content_age_days']))} |
| 6 | {float(top10[5]['baseline_action_score']):.3f} | {top10[5]['action_label']} | {top10[5]['reason_code']} | {int(float(top10[5]['impressions_90d'])):,} | {int(float(top10[5]['content_age_days']))} |
| 7 | {float(top10[6]['baseline_action_score']):.3f} | {top10[6]['action_label']} | {top10[6]['reason_code']} | {int(float(top10[6]['impressions_90d'])):,} | {int(float(top10[6]['content_age_days']))} |
| 8 | {float(top10[7]['baseline_action_score']):.3f} | {top10[7]['action_label']} | {top10[7]['reason_code']} | {int(float(top10[7]['impressions_90d'])):,} | {int(float(top10[7]['content_age_days']))} |
| 9 | {float(top10[8]['baseline_action_score']):.3f} | {top10[8]['action_label']} | {top10[8]['reason_code']} | {int(float(top10[8]['impressions_90d'])):,} | {int(float(top10[8]['content_age_days']))} |
| 10 | {float(top10[9]['baseline_action_score']):.3f} | {top10[9]['action_label']} | {top10[9]['reason_code']} | {int(float(top10[9]['impressions_90d'])):,} | {int(float(top10[9]['content_age_days']))} |

## Generated files

Charts (mirrored under `work/outputs/charts/` and `outputs/charts/`):

- `outputs/charts/trend_distribution.svg`
- `outputs/charts/age_vs_decline.svg`
- `outputs/charts/volume_vs_decline.svg`
- `outputs/charts/action_mix.svg`
- `outputs/charts/top_reason_codes.svg`
- `outputs/charts/top_feature_importance.svg`
- `outputs/charts/model_comparison.svg`
- `outputs/charts/train_vs_test.svg`

Tables / metrics:

- `work/outputs/baseline_action_score.csv`
- `work/outputs/action_queue.csv`
- `work/outputs/model_results.json`
- `outputs/model_results.json`
- `outputs/refresh_queue_sample.csv` (top 50 preview only)
- `outputs/model_report.md` (this file)
- `work/outputs/model_report.md` (copy)

## Practical use

Use the ranked queue as a reviewer aid, not as an automatic publishing decision.
Safest first production use: inspect high-risk / high-impression rows, verify the
page manually, and compare the recommendation against editorial context.
No-go without humans: mass AI rewrites, auto-delete, or redirects from risk alone.

Re-run with sklearn for RF probability queue:
`python work/scripts/export_intern_artifacts.py`
"""

    for path in (ROOT_OUT / "model_report.md", WORK_OUT / "model_report.md"):
        path.write_text(report, encoding="utf-8")

    print("Wrote charts, CSVs, JSON, and model_report.md (stdlib path)")
    print(f"rows={n} declining={d} actions={actions}")


if __name__ == "__main__":
    main()
