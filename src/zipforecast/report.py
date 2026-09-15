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
    "gbm_county_target": "Gradient boosting, all US metros, target neutralized by county",
    "momentum_1y": "Baseline: last year's relative growth continues",
    "momentum_5y": "Baseline: last 5 years' relative growth continues",
    "catch_up_price": "Baseline: cheaper than the metro median catches up",
    "catch_up_neighbors": "Baseline: cheaper than the 10 nearest ZIPs catches up",
    "beta_x_metro_trend": "Baseline: 10-year beta to metro times the metro's past-year growth",
}


def _fmt(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.3f}"


def _interval(lo: float, hi: float) -> str:
    return "n/a" if pd.isna(lo) or pd.isna(hi) else f"{lo:+.2f} to {hi:+.2f}"


def _pct(log_change: float) -> str:
    return f"{100 * (np.exp(log_change) - 1):+.0f}%"


def _summary_table(summary: pd.DataFrame, horizon: int) -> str:
    s = summary[summary.horizon == horizon].copy()
    s["order"] = s["model"].map({m: i for i, m in enumerate(MODEL_LABELS)})
    s = s.sort_values("order")
    lines = [
        "| Model | Spearman, NYC ZIPs | 90% interval | Spearman, NYC within county | "
        "Spearman, all ZIPs | Top-fifth minus bottom-fifth, NYC (log pts) | "
        "Worst year, NYC Spearman | Folds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in s.iterrows():
        within = r.get("spearman_nyc_within_county_mean", np.nan)
        ci = _interval(r.get("spearman_nyc_ci_low", np.nan), r.get("spearman_nyc_ci_high", np.nan))
        lines.append(
            f"| {MODEL_LABELS.get(r.model, r.model)} | {_fmt(r.spearman_nyc_mean)} | {ci} | "
            f"{_fmt(within)} | {_fmt(r.spearman_national_mean)} | "
            f"{_fmt(r.q5_q1_spread_nyc_mean)} | {_fmt(r.spearman_nyc_min)} | {int(r.n_folds)} |"
        )
    return "\n".join(lines)


def _beats_baseline_line(summary: pd.DataFrame, horizon: int) -> str:
    s = summary[(summary.horizon == horizon) & (summary.model == "gbm_national")]
    if s.empty or pd.isna(s["best_baseline"].iloc[0]):
        return ""
    r = s.iloc[0]
    label = MODEL_LABELS[r.best_baseline].removeprefix("Baseline: ")
    text = (
        f"Best baseline at this horizon: {label}. The national gradient boosting model beat it "
        f"on NYC ZIPs in {int(r.folds_gbm_beats_best_baseline)} of {int(r.folds_compared)} "
        "test origins where both were scored."
    )
    gap = r.get("gap_vs_best_baseline", np.nan)
    if not pd.isna(gap):
        ci = _interval(r.get("gap_ci_low", np.nan), r.get("gap_ci_high", np.nan))
        verdict = (
            "the interval excludes zero"
            if r.gap_ci_low > 0 or r.gap_ci_high < 0
            else "the interval includes zero, so the two are not distinguishable on this data"
        )
        text += (
            f" Mean gap in NYC Spearman, model minus baseline: {_fmt(gap)}, block-bootstrap 90% "
            f"interval {ci} ({verdict})."
        )
    return text


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


def _beta_table(beta: pd.DataFrame) -> str:
    lines = [
        "| Origin | ZIPs | Beta vs realized, all metros | ZIPs whose metro rose | "
        "ZIPs whose metro fell | Share of ZIPs in rising metros | NYC metro growth | "
        "Beta vs realized, NYC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in beta.iterrows():
        lines.append(
            f"| {int(r.origin_year)} | {int(r.n):,} | {_fmt(r.beta_spearman_all)} | "
            f"{_fmt(r.beta_spearman_up_metros)} | {_fmt(r.beta_spearman_down_metros)} | "
            f"{100 * r.share_zips_in_up_metros:.0f}% | {_pct(r.nyc_metro_growth)} | "
            f"{_fmt(r.beta_spearman_nyc)} |"
        )
    return "\n".join(lines)


ABLATION_LABELS = {
    "full": "all features",
    "without_listing_tempo": (
        "without Zillow listing tempo (inventory, new listings, days to pending, price cuts)"
    ),
    "without_county": "without county-relative price and momentum",
    "without_beta": "without 10-year beta to metro",
    "without_regime": (
        "without regime features (mortgage rate, price-to-rent, trailing factor signs)"
    ),
    "without_all_groups": "without all four groups",
}


def _ablation_section(ablation: pd.DataFrame) -> list[str]:
    cols = [c for c in ABLATION_LABELS if c in ablation.columns]
    full = ablation["full"].mean()
    lines = [
        "| Features | Mean NYC Spearman | Change from all features | "
        "Origins where the group helped, of those where it changed the score |",
        "|---|---|---|---|",
    ]
    for c in cols:
        mean = ablation[c].mean()
        if c == "full":
            lines.append(f"| {ABLATION_LABELS[c]} | {_fmt(mean)} | | |")
            continue
        helped = int((ablation["full"] > ablation[c]).sum())
        changed = int((ablation["full"] != ablation[c]).sum())
        lines.append(
            f"| {ABLATION_LABELS[c]} | {_fmt(mean)} | {_fmt(full - mean)} | {helped} of {changed} |"
        )
    per_year = [
        "| Test origin | " + " | ".join(c.replace("without_", "no ") for c in cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for _, r in ablation.iterrows():
        per_year.append(
            f"| {int(r.test_year)} | " + " | ".join(_fmt(r[c])[:5] for c in cols) + " |"
        )
    return [
        "### Which feature groups the 1-year model uses",
        "",
        "The national gradient boosting model refit with one group of features removed, scored on "
        "the same held-out NYC origins. The change column is the group's contribution. Zillow's "
        "listing files start in 2018, so the tempo features are empty at earlier origins and "
        "removing them changes nothing there.",
        "",
        *lines,
        "",
        *per_year,
        "",
    ]


def _beta_summary(beta: pd.DataFrame) -> str:
    up = beta["beta_spearman_up_metros"].mean()
    down = beta["beta_spearman_down_metros"].mean()
    nyc_up = beta.loc[beta["nyc_metro_growth"] > 0, "beta_spearman_nyc"]
    nyc_down = beta.loc[beta["nyc_metro_growth"] < 0, "beta_spearman_nyc"]
    return (
        f"Averaged over origins: in metros that rose over the next year, ZIP beta and realized "
        f"relative growth had Spearman {_fmt(up)}; in metros that fell, {_fmt(down)}. "
        f"For NYC, the {len(nyc_up)} origins where the metro rose average {_fmt(nyc_up.mean())} "
        f"and the {len(nyc_down)} where it fell average {_fmt(nyc_down.mean())}. "
        "Beta is the slope of a ZIP's annual growth on its metro's over the trailing "
        f"{config.BETA_WINDOW_YEARS} years (at least {config.BETA_MIN_YEARS} years observed). "
        "A positive number in rising metros and a negative one in falling metros is the "
        "high-beta pattern: the ZIPs that gain the most in good years lose the most in bad ones."
    )


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
            "spearman_nyc_ci90": [
                _json_value(s.loc[m, c]) if c in s.columns else None
                for c in ("spearman_nyc_ci_low", "spearman_nyc_ci_high")
            ],
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


def _zip_rows(ranking: pd.DataFrame) -> list[str]:
    rows = []
    for _, r in ranking.iterrows():
        zhvf = r["zillow_1y_forecast_pct"]
        rows.append(
            f"| {int(r['rank'])} | {r['zip']} | {r['city']}, {r['state']} | {r['county']} | "
            f"${r['zhvi'] / 1000:,.0f}k | {_pct(r['log_price_rel_metro'])} | "
            f"{_pct(r['mom_1y_rel'])} | {r['score']:.0f} | {r['hist_relative_pct']:+.1f}% | "
            f"{'n/a' if pd.isna(zhvf) else f'{zhvf:+.1f}%'} |"
        )
    return rows


RESULTS_FILE = config.ROOT / "RESULTS.md"

# Sub-areas that get their own top-n table in RESULTS.md: (title, counties, city to bold).
SUBSETS: list[tuple[str, list[str], str | None]] = [
    (
        "Manhattan, Brooklyn and Queens",
        ["New York County", "Kings County", "Queens County"],
        None,
    ),
    ("Hudson County, NJ", ["Hudson County"], "Jersey City"),
]


def _subset_rows(ranking: pd.DataFrame, bold_city: str | None) -> list[str]:
    rows = []
    for (_, r), line in zip(ranking.iterrows(), _zip_rows(ranking), strict=True):
        cells = line.split(" | ")
        cells[1] = f"**{cells[1]}**" if r["city"] == bold_city else cells[1]
        rows.append(f"| {r['sub_rank']} " + " | ".join(cells))
    return rows


def _list_shape(rows: pd.DataFrame) -> str:
    """How the listed ZIPs sit against the metro: median price and last-year growth."""
    price = rows["log_price_rel_metro"].median()
    growth = rows["mom_1y_rel"].median()
    return (
        f"priced {_pct(price).lstrip('+-')} {'above' if price >= 0 else 'below'} the metro "
        f"median and grew {abs(100 * (np.exp(growth) - 1)):.0f} points "
        f"{'faster' if growth >= 0 else 'slower'} than the metro last year"
    )


def write_results_summary(ranking: pd.DataFrame, summary: pd.DataFrame | None, n: int = 15) -> None:
    """One-page RESULTS.md: what the 1-year model scored and the top and bottom NYC ZIPs."""
    header = [
        "| Rank | ZIP | Place | County | Home value | vs metro median | Last year vs metro | "
        "Score | Past ZIPs in this score band, next year vs metro | Zillow 12-month forecast |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    origin = ranking["origin"].iloc[0]
    top, bottom = ranking.head(n), ranking.tail(n)
    parts = ["# Results: a small one-year edge, nothing at five to ten years", ""]
    correlation = "well under +0.5"
    if summary is not None and (summary.horizon == 1).any():
        s = summary[summary.horizon == 1].set_index("model")
        gbm, mom = s.loc["gbm_national"], s.loc["momentum_1y"]
        correlation = f"{gbm['spearman_nyc_mean']:+.2f}"
        parts += [
            "The model ranks NYC-metro ZIPs by expected price growth relative to the metro over "
            f"the next year, trained on every US metro. Over {int(gbm['n_folds'])} test years it "
            "never saw, the Spearman rank correlation between its ranking and what the ZIPs then "
            "did was "
            f"{gbm['spearman_nyc_mean']:+.2f}, worst year {gbm['spearman_nyc_min']:+.2f}. "
            'The one-line rule "last year\'s growth relative to the metro continues" scores '
            f"{mom['spearman_nyc_mean']:+.2f}. The model beat that rule in "
            f"{int(gbm['folds_gbm_beats_best_baseline'])} of {int(gbm['folds_compared'])} years, "
            f"by {gbm['gap_vs_best_baseline']:+.2f} on average, 90% interval "
            f"{_interval(gbm['gap_ci_low'], gbm['gap_ci_high'])}: real, but small. In price terms, "
            f"the model's top fifth of ZIPs beat its bottom fifth by "
            f"{100 * gbm['q5_q1_spread_nyc_mean']:.1f} percentage points of growth against the "
            f"metro over the next year; the rule's top fifth beat its bottom fifth by "
            f"{100 * mom['q5_q1_spread_nyc_mean']:.1f}.",
            "",
            "At five and ten years the model has no skill, and neither did anything else we tried. "
            "Which end of the cheap-to-expensive axis wins flips every 6 to 8 years, and the data "
            "holds two such cycles. The 5- and 10-year files in `output/` are lists of the "
            "cheapest ZIPs, not forecasts.",
            "",
        ]
    parts += [
        f"Zillow data through {origin}; {len(ranking)} NYC-metro ZIPs ranked. Full tables are in "
        "[output/REPORT.md](output/REPORT.md), method and caveats in the [README](README.md). "
        "Regenerate with `zipforecast evaluate` or `zipforecast rank`.",
        "",
        f"## The {n} best and {n} worst ranked NYC-metro ZIPs for the next 12 months",
        "",
        "Score is the model's expected percentile of price growth within the NYC metro, 0 to 100. "
        "The next column is the median of what NYC ZIPs in the same 5-point score band did over "
        "the following year in past origins, relative to the metro: a historical analogue, not "
        "a forecast of this year's metro. Zillow's 12-month forecast is an absolute number, so "
        "the two columns are not comparable.",
        "",
        f"The top {n} are {_list_shape(top)}. The bottom {n} are {_list_shape(bottom)}. "
        "Read the list for what it is: mostly last year's relative growth, adjusted by how fast "
        "listings are moving.",
        "",
        f"### Top {n}",
        "",
        *header,
        *_zip_rows(top),
        "",
        f"### Bottom {n}",
        "",
        *header,
        *_zip_rows(bottom),
        "",
    ]
    for name, counties, bold in SUBSETS:
        area = ranking[ranking["county"].isin(counties)]
        sub = area.head(n).copy()
        sub["sub_rank"] = range(1, len(sub) + 1)
        note = (
            f"The first column is the rank within this area, the second the rank among all "
            f"{len(ranking)} NYC-metro ZIPs."
        )
        if bold:
            k = int((area["city"] == bold).sum())
            note = (
                f"{bold} alone has {k} ZIPs with Zillow coverage, shown in bold among the rest of "
                f"the county. {note}"
            )
        parts += [
            f"### {name}: top {len(sub)} of {len(area)}",
            "",
            note,
            "",
            "| # " + header[0],
            "|---" + header[1],
            *_subset_rows(sub, bold),
            "",
        ]
    parts += [
        f"A rank correlation of {correlation} leaves room for a fair share of the top {n} to trail "
        f"the metro next year and a fair share of the bottom {n} to beat it. This is a shortlist "
        "of where to look, not a reason to buy or sell in any one ZIP.",
        "",
    ]
    RESULTS_FILE.write_text("\n".join(parts))
    log.info("wrote %s", RESULTS_FILE)


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
    beta: pd.DataFrame | None = None,
    ablation: pd.DataFrame | None = None,
) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.OUTPUT_DIR / "evaluation_summary.csv", index=False)
    if beta is not None:
        beta.to_csv(config.OUTPUT_DIR / "beta_history_1y.csv", index=False)
    if ablation is not None:
        ablation.to_csv(config.OUTPUT_DIR / "ablation_1y.csv", index=False)
    for h, r in results.items():
        r.to_csv(config.OUTPUT_DIR / f"evaluation_{h}y_by_year.csv", index=False)
    for h, f in factors.items():
        f.to_csv(config.OUTPUT_DIR / f"factor_history_{h}y.csv", index=False)
    for h, imp in importance.items():
        imp.to_csv(config.OUTPUT_DIR / f"feature_importance_{h}y.csv", index=False)
    for h, rk in rankings.items():
        rk.to_csv(config.OUTPUT_DIR / f"nyc_ranking_{h}y.csv", index=False)
        write_ranking_json(rk, h, summary)
    if 1 in rankings:
        write_results_summary(rankings[1], summary)

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
        "outcome was known by then. The 90% interval resamples test origins in blocks as long "
        "as the horizon; with 7 to 16 origins it is a rough guide, not a precise one.",
        "",
    ]
    if beta is not None and not beta.empty:
        parts += [
            "## Does a ZIP's beta to its metro predict its relative growth?",
            "",
            _beta_summary(beta),
            "",
            _beta_table(beta),
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
        ]
        if h == 1 and ablation is not None and not ablation.empty:
            parts += _ablation_section(ablation)
        parts += [
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
