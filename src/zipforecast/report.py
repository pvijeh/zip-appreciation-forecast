"""Write the evaluation tables and NYC rankings under output/."""

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from zipforecast import config
from zipforecast.model import FACTOR_HISTORY_FEATURES, _spearman

log = logging.getLogger(__name__)

MODEL_LABELS = {
    "gbm_national": "Gradient boosting, trained on all US metros",
    "gbm_recency": "Gradient boosting, all US metros, recent origins weighted more",
    "ridge_national": "Ridge regression, trained on all US metros",
    "gbm_nyc_only": "Gradient boosting, trained on NYC metro only",
    "momentum_1y": "Baseline: last year's relative growth continues",
    "momentum_5y": "Baseline: last 5 years' relative growth continues",
    "catch_up_price": "Baseline: cheaper than the metro median catches up",
    "catch_up_neighbors": "Baseline: cheaper than the 10 nearest ZIPs catches up",
}


def _fmt(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.3f}"


def _pct(log_change: float) -> str:
    return f"{100 * (np.exp(log_change) - 1):+.0f}%"


def _summary_table(summary: pd.DataFrame, horizon: int) -> str:
    s = summary[summary.horizon == horizon].copy()
    s["order"] = s["model"].map({m: i for i, m in enumerate(MODEL_LABELS)})
    s = s.sort_values("order")
    lines = [
        "| Model | Spearman, NYC ZIPs | Spearman, all ZIPs | Top-fifth minus bottom-fifth, "
        "NYC (log pts) | Worst year, NYC Spearman | Folds |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in s.iterrows():
        lines.append(
            f"| {MODEL_LABELS.get(r.model, r.model)} | {_fmt(r.spearman_nyc_mean)} | "
            f"{_fmt(r.spearman_national_mean)} | {_fmt(r.q5_q1_spread_nyc_mean)} | "
            f"{_fmt(r.spearman_nyc_min)} | {int(r.n_folds)} |"
        )
    return "\n".join(lines)


def _beats_baseline_line(summary: pd.DataFrame, horizon: int) -> str:
    s = summary[(summary.horizon == horizon) & (summary.model == "gbm_national")]
    if s.empty or pd.isna(s["best_baseline"].iloc[0]):
        return ""
    r = s.iloc[0]
    label = MODEL_LABELS[r.best_baseline].removeprefix("Baseline: ")
    return (
        f"Best baseline at this horizon: {label}. The national gradient boosting model beat it "
        f"on NYC ZIPs in {int(r.folds_gbm_beats_best_baseline)} of {int(r.folds_compared)} "
        "test origins where both were scored."
    )


def _per_year_table(results: pd.DataFrame, horizon: int) -> str:
    r = results[(results.horizon == horizon)]
    models = [m for m in MODEL_LABELS if m in set(r.model)]
    wide = r.pivot(index="test_year", columns="model", values="spearman_nyc")[models]
    lines = ["| Test origin | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for year, row in wide.iterrows():
        lines.append(f"| {year} | " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _ranking_table(ranking: pd.DataFrame, n: int) -> str:
    lines = [
        "| Rank | ZIP | City | County | ZHVI | Score (expected metro percentile) | "
        "Historical growth vs metro at that percentile | Past 5y vs metro | "
        "Price vs metro median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in ranking.head(n).iterrows():
        lines.append(
            f"| {r['rank']} | {r.zip} | {r.city} | {r.county} | ${r.zhvi:,.0f} | "
            f"{r.score:.1f} | {r.hist_relative_pct:+.1f}% | {_pct(r.mom_5y_rel)} | "
            f"{_pct(r.log_price_rel_metro)} |"
        )
    return "\n".join(lines)


def _factor_table(history: pd.DataFrame, nyc: bool) -> str:
    suffix = "_nyc" if nyc else ""
    cols = [f"{c}{suffix}" for c in FACTOR_HISTORY_FEATURES]
    lines = [
        "| Origin | " + " | ".join(f"`{c}`" for c in FACTOR_HISTORY_FEATURES) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for _, r in history.iterrows():
        lines.append(f"| {int(r.origin_year)} | " + " | ".join(_fmt(r[c])[:5] for c in cols) + " |")
    return "\n".join(lines)


def _zhvf_line(ranking: pd.DataFrame) -> str:
    zhvf = ranking["zillow_1y_forecast_pct"]
    if not zhvf.notna().any():
        return "Zillow's published 1-year forecast was not downloaded, so no comparison."
    rho = _spearman(ranking["score"], zhvf)
    top = zhvf[ranking["rank"] <= 25].mean()
    return (
        f"Comparison with Zillow's own published 12-month forecast (ZHVF) for the same ZIPs: "
        f"Spearman between our score and Zillow's forecast {_fmt(rho)} over "
        f"{int(zhvf.notna().sum())} ZIPs; Zillow expects {top:+.1f}% for our top 25 versus "
        f"{zhvf.mean():+.1f}% for all NYC-metro ZIPs. Zillow's horizon is one year, so this "
        "is a check on agreement, not on accuracy."
    )


def _importance_table(importance: pd.DataFrame, n: int = 15) -> str:
    lines = ["| Feature | Permutation importance (MSE increase) |", "|---|---|"]
    for _, r in importance.head(n).iterrows():
        lines.append(f"| `{r.feature}` | {r.importance_mean:.5f} |")
    return "\n".join(lines)


PER_HORIZON_FILES = (
    "evaluation_{h}y_by_year.csv",
    "factor_history_{h}y.csv",
    "feature_importance_{h}y.csv",
    "nyc_ranking_{h}y.csv",
    "nyc_forecast_{h}y.json",
)

JSON_FIELDS = {
    "rank": "rank",
    "zip": "zip",
    "city": "city",
    "county": "county",
    "state": "state",
    "zhvi": "zhvi",
    "score": "score",
    "nyc_percentile": "nyc_percentile",
    "hist_relative_pct": "historical_growth_vs_metro_pct",
    "zillow_1y_forecast_pct": "zillow_1y_forecast_pct",
}


def _json_value(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, float | np.floating):
        return None if np.isnan(v) else round(float(v), 2)
    return v


def _skill_for(summary: pd.DataFrame | None, horizon: int) -> dict:
    if summary is None:
        return {}
    s = summary[summary.horizon == horizon].set_index("model")
    return {
        m: {
            "spearman_nyc_mean": _json_value(s.loc[m, "spearman_nyc_mean"]),
            "spearman_nyc_min": _json_value(s.loc[m, "spearman_nyc_min"]),
            "folds": int(s.loc[m, "n_folds"]),
        }
        for m in MODEL_LABELS
        if m in s.index
    }


def write_ranking_json(
    ranking: pd.DataFrame, horizon: int, summary: pd.DataFrame | None = None
) -> None:
    """Ranked NYC-metro ZIPs plus the measured walk-forward skill needed to read them."""
    zhvf = ranking["zillow_1y_forecast_pct"]
    doc = {
        "horizon_years": horizon,
        "origin": str(ranking["origin"].iloc[0]),
        "generated": dt.date.today().isoformat(),
        "n_zips": int(len(ranking)),
        "score_meaning": (
            "Average of the gradient boosting and ridge models' predicted percentile of "
            "forward price growth within the NYC metro, 0-100. Not an absolute forecast."
        ),
        "historical_growth_vs_metro_pct_meaning": (
            "What NYC ZIPs at this predicted percentile realized over the horizon, relative "
            "to the metro, in past origins. A historical analogue, not a forecast."
        ),
        "walk_forward_skill_nyc": _skill_for(summary, horizon),
        "spearman_score_vs_zillow_1y_forecast": _json_value(
            _spearman(ranking["score"], zhvf) if zhvf.notna().any() else np.nan
        ),
        "price_vs_metro_median_pct_meaning": "ZHVI relative to the NYC-metro median ZIP.",
        "zips": [
            {
                **{out: _json_value(r[src]) for src, out in JSON_FIELDS.items()},
                "price_vs_metro_median_pct": _json_value(
                    100 * (np.exp(r["log_price_rel_metro"]) - 1)
                ),
                "past_5y_growth_vs_metro_pct": _json_value(100 * (np.exp(r["mom_5y_rel"]) - 1)),
            }
            for _, r in ranking.iterrows()
        ],
    }
    path = config.OUTPUT_DIR / f"nyc_forecast_{horizon}y.json"
    path.write_text(json.dumps(doc, indent=1))
    log.info("wrote %s", path)


def _remove_stale_horizon_files(horizons: set[int]) -> None:
    """Drop per-horizon files from earlier runs so output/ describes one evaluation."""
    for h in set(config.HORIZONS) - horizons:
        for pattern in PER_HORIZON_FILES:
            (config.OUTPUT_DIR / pattern.format(h=h)).unlink(missing_ok=True)


def write_report(
    results: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
    importance: dict[int, pd.DataFrame],
    factors: dict[int, pd.DataFrame],
    rankings: dict[int, pd.DataFrame],
    panel: pd.DataFrame,
) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.OUTPUT_DIR / "evaluation_summary.csv", index=False)
    for h, r in results.items():
        r.to_csv(config.OUTPUT_DIR / f"evaluation_{h}y_by_year.csv", index=False)
    for h, f in factors.items():
        f.to_csv(config.OUTPUT_DIR / f"factor_history_{h}y.csv", index=False)
    for h, imp in importance.items():
        imp.to_csv(config.OUTPUT_DIR / f"feature_importance_{h}y.csv", index=False)
    for h, rk in rankings.items():
        rk.to_csv(config.OUTPUT_DIR / f"nyc_ranking_{h}y.csv", index=False)
        write_ranking_json(rk, h, summary)

    latest = panel["origin"].max().date()
    first = panel["origin"].min().date()
    n_zips = panel[panel.origin == panel.origin.max()]["zip"].nunique()
    n_nyc = int(panel[(panel.origin == panel.origin.max()) & panel.is_nyc]["zip"].nunique())

    parts = [
        "# Evaluation and NYC ranking",
        "",
        f"Generated by `zipforecast all`. Zillow data through {latest}; "
        f"origins every year from {first} to {latest}; {n_zips:,} ZIPs in metros, "
        f"{n_nyc} of them in the New York metro.",
        "",
        "The target is a ZIP's log price growth over the horizon minus the average of its metro. "
        "Models are fit on each ZIP's percentile within its metro, for both features and target, "
        "so a 2005 price level and a 2026 price level are compared only to their own metro. "
        "Spearman is the rank correlation between predicted and realized relative growth on "
        "test origins the model never saw; each test origin is trained only on origins whose "
        "outcome was known by then.",
        "",
    ]
    for h in sorted(results):
        parts += [
            f"## {h}-year horizon",
            "",
            "### Walk-forward results, averaged over test origins",
            "",
            _summary_table(summary, h),
            "",
            _beats_baseline_line(summary, h),
            "",
            "### NYC Spearman by test origin",
            "",
            _per_year_table(results[h], h),
            "",
            "### One feature at a time: Spearman with realized relative growth, by origin",
            "",
            "A positive number means ZIPs high on that feature went on to beat their metro. "
            "All US metro ZIPs first, then NYC-metro ZIPs only. A model fit on the top rows "
            "of this table is scored on the bottom rows.",
            "",
            _factor_table(factors[h], nyc=False),
            "",
            "NYC-metro ZIPs only:",
            "",
            _factor_table(factors[h], nyc=True),
            "",
        ]
        if h in importance and not importance[h].empty:
            year = int(importance[h]["test_year"].iloc[0])
            parts += [
                f"### Feature importance (national gradient boosting, test origin {year})",
                "",
                _importance_table(importance[h]),
                "",
            ]
        if h in rankings:
            parts += [
                f"### Top 25 NYC-metro ZIPs, {h}-year horizon (origin {latest})",
                "",
                "Score = average of the gradient boosting and ridge models' predicted percentile "
                "of forward growth within the metro (100 = expected to beat every other ZIP). "
                "The growth column is what NYC ZIPs at that percentile realized in past "
                "origins, relative to the metro; it is a historical analogue, not a forecast of "
                f"the metro itself. Full list with more columns in `output/nyc_ranking_{h}y.csv` "
                f"and `output/nyc_forecast_{h}y.json`.",
                "",
                _zhvf_line(rankings[h]),
                "",
                _ranking_table(rankings[h], 25),
                "",
            ]
    (config.OUTPUT_DIR / "REPORT.md").write_text("\n".join(parts))
    _remove_stale_horizon_files(set(results))
    log.info("wrote %s", config.OUTPUT_DIR / "REPORT.md")
