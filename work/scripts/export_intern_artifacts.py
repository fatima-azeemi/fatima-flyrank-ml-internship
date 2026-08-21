"""
Reproduce internship notebook analyses, save charts/tables, and rewrite model_report.md.

Mirrors work done in w03–w07 (6-feature RF + rule baseline + action queue).
Does not use the starter pipeline's richer feature set / client_holdout numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from ml_utils import simple_svg_bar_chart  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
WORK_OUT = ROOT / "work" / "outputs"
WORK_CHARTS = WORK_OUT / "charts"
ROOT_OUT = ROOT / "outputs"
ROOT_CHARTS = ROOT_OUT / "charts"

FEATURES = [
    "word_count",
    "content_age_days",
    "impressions_90d",
    "avg_position",
    "ctr",
    "search_volume",
]


def ensure_dirs() -> None:
    for path in (WORK_OUT, WORK_CHARTS, ROOT_OUT, ROOT_CHARTS):
        path.mkdir(parents=True, exist_ok=True)


def save_chart(name: str, title: str, labels: list[str], values: list[float], color: str) -> None:
    for folder in (WORK_CHARTS, ROOT_CHARTS):
        simple_svg_bar_chart(title, labels, values, folder / name, color=color)


def assign_action(row: pd.Series) -> tuple[str, str]:
    age = row["content_age_days"]
    imp = row["impressions_90d"]
    if age >= 90 and imp >= 500:
        return "REFRESH_CONTENT", "STALE_HIGH_IMPRESSIONS"
    if age >= 180:
        return "REVIEW_STALE", "STALE_LOW_IMPRESSIONS"
    return "MONITOR", "LOW_PRIORITY"


def get_metrics(y_true, y_pred, y_prob=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None,
    }


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_PATH)
    df["target"] = (df["trend_direction"] == "down").astype(int)
    df["is_stale_high_imp"] = (
        (df["content_age_days"] > 180) & (df["impressions_90d"] > 500)
    ).astype(int)

    n_rows = len(df)
    n_declining = int(df["target"].sum())
    decline_rate = n_declining / n_rows

    # --- Signal buckets (honest numbers from w04_signal_audit) ---
    age_bins = pd.cut(
        df["content_age_days"],
        bins=[0, 90, 180, 365, 1000],
        labels=["<90d", "90-180d", "181-365d", ">365d"],
    )
    age_stats = (
        df.assign(age_tier=age_bins)
        .groupby("age_tier", observed=False)
        .agg(n=("content_id", "count"), decline_rate=("target", "mean"))
        .reset_index()
    )
    age_stats["decline_pct"] = age_stats["decline_rate"] * 100

    # Match w04_signal_audit: do not fillna — missing volume drops out of buckets
    vol_bins = pd.cut(
        df["search_volume"],
        bins=[-1, 10, 100, 1000, 100000],
        labels=["Zero/Low", "Medium", "High", "Very High"],
    )
    vol_stats = (
        df.assign(volume_tier=vol_bins)
        .groupby("volume_tier", observed=False)
        .agg(n=("content_id", "count"), decline_rate=("target", "mean"))
        .reset_index()
    )
    vol_stats["decline_pct"] = vol_stats["decline_rate"] * 100

    save_chart(
        "age_vs_decline.svg",
        "Decline rate by content age (observed)",
        age_stats["age_tier"].astype(str).tolist(),
        age_stats["decline_pct"].tolist(),
        "#2F6F8F",
    )
    save_chart(
        "volume_vs_decline.svg",
        "Decline rate by search volume (observed)",
        vol_stats["volume_tier"].astype(str).tolist(),
        vol_stats["decline_pct"].tolist(),
        "#3B6D4A",
    )

    # --- Baseline queue (w04_baseline_score) ---
    queue = df.copy()
    queue["baseline_action_score"] = np.log1p(queue["impressions_90d"]) * (
        queue["content_age_days"] / 365.0
    )
    actions = queue.apply(assign_action, axis=1)
    queue["action_label"] = [a[0] for a in actions]
    queue["reason_code"] = [a[1] for a in actions]
    queue = queue.sort_values("baseline_action_score", ascending=False).reset_index(drop=True)

    baseline_cols = [
        "content_id",
        "client_id",
        "baseline_action_score",
        "reason_code",
        "action_label",
        "impressions_90d",
        "content_age_days",
    ]
    queue[baseline_cols].to_csv(WORK_OUT / "baseline_action_score.csv", index=False)

    action_counts = queue["action_label"].value_counts()
    save_chart(
        "action_mix.svg",
        "Action mix (rule + playbook labels)",
        action_counts.index.tolist(),
        action_counts.astype(float).tolist(),
        "#8B5A2B",
    )
    reason_counts = queue["reason_code"].value_counts()
    save_chart(
        "top_reason_codes.svg",
        "Reason codes in ranked queue",
        reason_counts.index.tolist(),
        reason_counts.astype(float).tolist(),
        "#6F4E7C",
    )

    # --- Model train/compare (w05 + w06) ---
    X = df[FEATURES]
    y = df["target"]
    X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=FEATURES)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=FEATURES)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    train_pred = rf.predict(X_train)
    train_prob = rf.predict_proba(X_train)[:, 1]

    test_df = df.loc[test_idx].copy()
    base_pred = np.ones(len(y_test))
    rule_pred = test_df["is_stale_high_imp"].to_numpy()

    metrics = {
        "base_rate": get_metrics(y_test, base_pred),
        "week4_rule": get_metrics(y_test, rule_pred),
        "random_forest_test": get_metrics(y_test, rf_pred, rf_prob),
        "random_forest_train": get_metrics(y_train, train_pred, train_prob),
    }

    importances = (
        pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    save_chart(
        "top_feature_importance.svg",
        "Random Forest feature importance (6-feature model)",
        importances["Feature"].tolist(),
        (importances["Importance"] * 100).tolist(),
        "#4A6FA5",
    )
    save_chart(
        "model_comparison.svg",
        "Test-set comparison (Accuracy / F1 / ROC-AUC×100)",
        [
            "Base Acc",
            "Rule Acc",
            "RF Acc",
            "Base F1",
            "Rule F1",
            "RF F1",
            "RF ROC-AUC",
        ],
        [
            metrics["base_rate"]["accuracy"] * 100,
            metrics["week4_rule"]["accuracy"] * 100,
            metrics["random_forest_test"]["accuracy"] * 100,
            metrics["base_rate"]["f1"] * 100,
            metrics["week4_rule"]["f1"] * 100,
            metrics["random_forest_test"]["f1"] * 100,
            metrics["random_forest_test"]["roc_auc"] * 100,
        ],
        "#2C5F2D",
    )
    save_chart(
        "train_vs_test.svg",
        "RF train vs test (honest gap check)",
        ["Train Acc", "Test Acc", "Train F1", "Test F1", "Train ROC", "Test ROC"],
        [
            metrics["random_forest_train"]["accuracy"] * 100,
            metrics["random_forest_test"]["accuracy"] * 100,
            metrics["random_forest_train"]["f1"] * 100,
            metrics["random_forest_test"]["f1"] * 100,
            metrics["random_forest_train"]["roc_auc"] * 100,
            metrics["random_forest_test"]["roc_auc"] * 100,
        ],
        "#B85C38",
    )

    # Full-data risk scores for playbook export (same recipe as notebooks)
    X_all = pd.DataFrame(imputer.transform(df[FEATURES]), columns=FEATURES)
    risk = rf.predict_proba(X_all)[:, 1]
    playbook = df.copy()
    playbook["risk_probability"] = risk
    playbook_actions = playbook.apply(assign_action, axis=1)
    playbook["action_label"] = [a[0] for a in playbook_actions]
    playbook["reason_code"] = [a[1] for a in playbook_actions]
    playbook = playbook.sort_values("risk_probability", ascending=False).reset_index(drop=True)

    export_cols = [
        "content_id",
        "client_id",
        "risk_probability",
        "action_label",
        "reason_code",
        "impressions_90d",
        "content_age_days",
        "word_count",
    ]
    playbook[export_cols].to_csv(WORK_OUT / "action_queue.csv", index=False)
    # Sample only at repo root (avoid committing full queue / private-looking dumps)
    playbook[export_cols].head(50).to_csv(ROOT_OUT / "refresh_queue_sample.csv", index=False)

    trend_counts = df["trend_direction"].value_counts()
    save_chart(
        "trend_distribution.svg",
        "Trend direction mix (starter snapshot)",
        trend_counts.index.astype(str).tolist(),
        trend_counts.astype(float).tolist(),
        "#5C6B73",
    )

    results = {
        "source": "work/notebooks (intern lane: 6-feature RF)",
        "rows_scored": n_rows,
        "declining_rows": n_declining,
        "declining_rate": round(decline_rate, 4),
        "split": "stratified_80_20",
        "random_state": 42,
        "features": FEATURES,
        "metrics": metrics,
        "feature_importance": importances.to_dict(orient="records"),
        "action_counts": action_counts.to_dict(),
        "reason_counts": reason_counts.to_dict(),
        "age_decline": age_stats.to_dict(orient="records"),
        "volume_decline": vol_stats.to_dict(orient="records"),
    }
    (WORK_OUT / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT_OUT / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    rf_test = metrics["random_forest_test"]
    rf_train = metrics["random_forest_train"]
    rule = metrics["week4_rule"]
    base = metrics["base_rate"]

    top10 = playbook.head(10)[
        [
            "content_id",
            "risk_probability",
            "action_label",
            "reason_code",
            "impressions_90d",
            "content_age_days",
        ]
    ]

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

- Rows scored: {n_rows:,}
- Declining-label rows: {n_declining:,}
- Declining-label rate: {decline_rate:.3f}
- Split strategy used for validation: stratified 80/20 (`random_state=42`)
- Target: `trend_direction == 'down'`
- Features used: `{FEATURES[0]}`, `{FEATURES[1]}`, `{FEATURES[2]}`, `{FEATURES[3]}`, `{FEATURES[4]}`, `{FEATURES[5]}`
- Explicitly excluded (leakage): `trend_pct`, `trend_direction` as features; IDs not used as predictors
- Missingness handled with train-only median imputation (`word_count` ~25.7% missing; `search_volume` ~8.2%)

## Signal audit (w04) — observed patterns

Heavy tails: `impressions_90d`, `ctr`, and `search_volume` are highly skewed
(median ≪ mean), so comparisons use buckets / ranks rather than raw means.

| Age tier | n | Decline % (observed) |
|---|---:|---:|
| <90d | {int(age_stats.loc[age_stats['age_tier']=='<90d','n'].iloc[0]):,} | {age_stats.loc[age_stats['age_tier']=='<90d','decline_pct'].iloc[0]:.2f} |
| 90–180d | {int(age_stats.loc[age_stats['age_tier']=='90-180d','n'].iloc[0]):,} | {age_stats.loc[age_stats['age_tier']=='90-180d','decline_pct'].iloc[0]:.2f} |
| 181–365d | {int(age_stats.loc[age_stats['age_tier']=='181-365d','n'].iloc[0]):,} | {age_stats.loc[age_stats['age_tier']=='181-365d','decline_pct'].iloc[0]:.2f} |
| >365d | {int(age_stats.loc[age_stats['age_tier']=='>365d','n'].iloc[0]):,} | {age_stats.loc[age_stats['age_tier']=='>365d','decline_pct'].iloc[0]:.2f} |

| Volume tier | n | Decline % (observed) |
|---|---:|---:|
| Zero/Low (0–10) | {int(vol_stats.loc[vol_stats['volume_tier']=='Zero/Low','n'].iloc[0]):,} | {vol_stats.loc[vol_stats['volume_tier']=='Zero/Low','decline_pct'].iloc[0]:.2f} |
| Medium (11–100) | {int(vol_stats.loc[vol_stats['volume_tier']=='Medium','n'].iloc[0]):,} | {vol_stats.loc[vol_stats['volume_tier']=='Medium','decline_pct'].iloc[0]:.2f} |
| High (101–1k) | {int(vol_stats.loc[vol_stats['volume_tier']=='High','n'].iloc[0]):,} | {vol_stats.loc[vol_stats['volume_tier']=='High','decline_pct'].iloc[0]:.2f} |
| Very High (>1k) | {int(vol_stats.loc[vol_stats['volume_tier']=='Very High','n'].iloc[0]):,} | {vol_stats.loc[vol_stats['volume_tier']=='Very High','decline_pct'].iloc[0]:.2f} |

Honest takeaway from these tables: in this snapshot, **younger / lower-volume
tiers show higher decline rates** than older / higher-volume tiers. The
`STALE_HIGH_IMPRESSIONS` flag is still useful as an **impact / ROI** queue
(large impression base), not as proof that flagged pages decline more often.

## Rule baseline (w04 / ML-07)

Score: `log1p(impressions_90d) * (content_age_days / 365)`.

| Action | Count |
|---|---:|
| REFRESH_CONTENT | {int(action_counts.get('REFRESH_CONTENT', 0)):,} |
| REVIEW_STALE | {int(action_counts.get('REVIEW_STALE', 0)):,} |
| MONITOR | {int(action_counts.get('MONITOR', 0)):,} |

Score range observed: {queue['baseline_action_score'].min():.4f} – {queue['baseline_action_score'].max():.4f}.

## Model comparison (w05 / ML-08)

Best model in this lane: **Random Forest**
(`n_estimators=150`, `max_depth=8`, `min_samples_split=10`, `random_state=42`),
selected by test ROC-AUC / F1 vs the Week-4 rule and the always-down base rate.
Same stratified test fold for all three.

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
| {importances.iloc[0]['Feature']} | {importances.iloc[0]['Importance']:.4f} |
| {importances.iloc[1]['Feature']} | {importances.iloc[1]['Importance']:.4f} |
| {importances.iloc[2]['Feature']} | {importances.iloc[2]['Importance']:.4f} |
| {importances.iloc[3]['Feature']} | {importances.iloc[3]['Importance']:.4f} |
| {importances.iloc[4]['Feature']} | {importances.iloc[4]['Importance']:.4f} |
| {importances.iloc[5]['Feature']} | {importances.iloc[5]['Importance']:.4f} |

## Action playbook queue (w07 / ML-10)

Model risk probability ranks the queue; action labels / reason codes follow the
same stale-impression rules as the Week-4 baseline so editors get a familiar
taxonomy.

- Mean risk probability: {playbook['risk_probability'].mean():.4f}
- Median impressions_90d: {playbook['impressions_90d'].median():.0f}

### Top 10 queue preview

| Rank | Risk p | Action | Reason | Impressions | Age (days) |
|---:|---:|---|---|---:|---:|
| 1 | {top10.iloc[0]['risk_probability']:.3f} | {top10.iloc[0]['action_label']} | {top10.iloc[0]['reason_code']} | {int(top10.iloc[0]['impressions_90d']):,} | {int(top10.iloc[0]['content_age_days'])} |
| 2 | {top10.iloc[1]['risk_probability']:.3f} | {top10.iloc[1]['action_label']} | {top10.iloc[1]['reason_code']} | {int(top10.iloc[1]['impressions_90d']):,} | {int(top10.iloc[1]['content_age_days'])} |
| 3 | {top10.iloc[2]['risk_probability']:.3f} | {top10.iloc[2]['action_label']} | {top10.iloc[2]['reason_code']} | {int(top10.iloc[2]['impressions_90d']):,} | {int(top10.iloc[2]['content_age_days'])} |
| 4 | {top10.iloc[3]['risk_probability']:.3f} | {top10.iloc[3]['action_label']} | {top10.iloc[3]['reason_code']} | {int(top10.iloc[3]['impressions_90d']):,} | {int(top10.iloc[3]['content_age_days'])} |
| 5 | {top10.iloc[4]['risk_probability']:.3f} | {top10.iloc[4]['action_label']} | {top10.iloc[4]['reason_code']} | {int(top10.iloc[4]['impressions_90d']):,} | {int(top10.iloc[4]['content_age_days'])} |
| 6 | {top10.iloc[5]['risk_probability']:.3f} | {top10.iloc[5]['action_label']} | {top10.iloc[5]['reason_code']} | {int(top10.iloc[5]['impressions_90d']):,} | {int(top10.iloc[5]['content_age_days'])} |
| 7 | {top10.iloc[6]['risk_probability']:.3f} | {top10.iloc[6]['action_label']} | {top10.iloc[6]['reason_code']} | {int(top10.iloc[6]['impressions_90d']):,} | {int(top10.iloc[6]['content_age_days'])} |
| 8 | {top10.iloc[7]['risk_probability']:.3f} | {top10.iloc[7]['action_label']} | {top10.iloc[7]['reason_code']} | {int(top10.iloc[7]['impressions_90d']):,} | {int(top10.iloc[7]['content_age_days'])} |
| 9 | {top10.iloc[8]['risk_probability']:.3f} | {top10.iloc[8]['action_label']} | {top10.iloc[8]['reason_code']} | {int(top10.iloc[8]['impressions_90d']):,} | {int(top10.iloc[8]['content_age_days'])} |
| 10 | {top10.iloc[9]['risk_probability']:.3f} | {top10.iloc[9]['action_label']} | {top10.iloc[9]['reason_code']} | {int(top10.iloc[9]['impressions_90d']):,} | {int(top10.iloc[9]['content_age_days'])} |

## Generated files

Charts (also mirrored under `work/outputs/charts/`):

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
"""

    for path in (ROOT_OUT / "model_report.md", WORK_OUT / "model_report.md"):
        path.write_text(report, encoding="utf-8")

    print("Wrote charts + CSVs + model_report.md")
    print(f"RF test ROC-AUC: {rf_test['roc_auc']:.4f}")
    print(f"Action mix: {action_counts.to_dict()}")


if __name__ == "__main__":
    main()
