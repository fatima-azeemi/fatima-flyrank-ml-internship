# FlyRank Refresh Opportunity Model Report

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

- Rows scored: 30,000
- Declining-label rows: 16,262
- Declining-label rate: 0.542
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
| <90d | 492 | 66.87 |
| 90–180d | 11,780 | 62.56 |
| 181–365d | 11,368 | 51.49 |
| >365d | 6,360 | 42.63 |

| Volume tier | n | Decline % (observed) |
|---|---:|---:|
| Zero/Low (0–10) | 18,392 | 59.04 |
| Medium (11–100) | 6,091 | 52.06 |
| High (101–1k) | 2,489 | 50.10 |
| Very High (>1k) | 560 | 44.46 |

Honest takeaway from these tables: in this snapshot, **younger / lower-volume
tiers show higher decline rates** than older / higher-volume tiers. The
`STALE_HIGH_IMPRESSIONS` flag is still useful as an **impact / ROI** queue
(large impression base), not as proof that flagged pages decline more often.

## Rule baseline (w04 / ML-07)

Score: `log1p(impressions_90d) * (content_age_days / 365)`.

| Action | Count |
|---|---:|
| REFRESH_CONTENT | 16,726 |
| REVIEW_STALE | 8,057 |
| MONITOR | 5,217 |

Score range observed: 0.1709 – 19.3573.

## Model comparison (w05 / ML-08)

Best model in this lane: **Random Forest**
(`n_estimators=150`, `max_depth=8`, `min_samples_split=10`, `random_state=42`),
selected by test ROC-AUC / F1 vs the Week-4 rule and the always-down base rate.
Same stratified test fold for all three.

| Model / Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Base rate (always down) | 0.542 | 0.542 | 1.000 | 0.703 | — |
| Week-4 rule (`STALE_HIGH_IMPRESSIONS`) | 0.482 | 0.536 | 0.328 | 0.407 | — |
| Random Forest (test) | 0.677 | 0.665 | 0.815 | 0.732 | 0.737 |

## Validation audit (w06 / ML-09)

Train-only imputation; metrics on the same RF:

| Metric | Training | Test (unseen) |
|---|---:|---:|
| Accuracy | 0.700 | 0.677 |
| Precision | 0.679 | 0.665 |
| Recall | 0.845 | 0.815 |
| F1 | 0.753 | 0.732 |
| ROC-AUC | 0.772 | 0.737 |

Honest claim: with this stratified split and train-only imputation, the RF
**observed** a test ROC-AUC of **0.7374**. Traffic and age
signals show **directional association** with the decline label and are suitable
as **decision-support** for prioritizing reviews — not causal guarantees that
an edit will recover traffic.

## Top features

| Feature | Importance |
|---|---:|
| impressions_90d | 0.3431 |
| avg_position | 0.2333 |
| content_age_days | 0.2092 |
| word_count | 0.0927 |
| ctr | 0.0715 |
| search_volume | 0.0503 |

## Action playbook queue (w07 / ML-10)

Model risk probability ranks the queue; action labels / reason codes follow the
same stale-impression rules as the Week-4 baseline so editors get a familiar
taxonomy.

- Mean risk probability: 0.5412
- Median impressions_90d: 731

### Top 10 queue preview

| Rank | Risk p | Action | Reason | Impressions | Age (days) |
|---:|---:|---|---|---:|---:|
| 1 | 0.816 | MONITOR | LOW_PRIORITY | 397 | 96 |
| 2 | 0.816 | REFRESH_CONTENT | STALE_HIGH_IMPRESSIONS | 580 | 104 |
| 3 | 0.816 | REFRESH_CONTENT | STALE_HIGH_IMPRESSIONS | 2,229 | 557 |
| 4 | 0.814 | REFRESH_CONTENT | STALE_HIGH_IMPRESSIONS | 1,339 | 104 |
| 5 | 0.814 | MONITOR | LOW_PRIORITY | 337 | 124 |
| 6 | 0.814 | REFRESH_CONTENT | STALE_HIGH_IMPRESSIONS | 871 | 557 |
| 7 | 0.813 | MONITOR | LOW_PRIORITY | 383 | 144 |
| 8 | 0.813 | MONITOR | LOW_PRIORITY | 140 | 144 |
| 9 | 0.812 | REFRESH_CONTENT | STALE_HIGH_IMPRESSIONS | 916 | 104 |
| 10 | 0.812 | REFRESH_CONTENT | STALE_HIGH_IMPRESSIONS | 628 | 557 |

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
