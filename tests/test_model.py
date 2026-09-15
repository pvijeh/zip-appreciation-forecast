import json

import numpy as np
import pandas as pd

from zipforecast import config, report
from zipforecast.model import (
    BASELINES,
    FEATURE_GROUPS,
    _quintile_spread,
    _spearman,
    _within_county_spearman,
    beta_history,
    block_bootstrap_ci,
    load_zhvf,
    summarize,
    walk_forward_folds,
)
from zipforecast.report import write_ranking_json, write_results_summary


def test_spearman_needs_twenty_pairs_and_ignores_nan():
    x = pd.Series(np.arange(30, dtype=float))
    y = x * 2
    y.iloc[0] = np.nan
    assert np.isclose(_spearman(x, y), 1.0)
    assert np.isnan(_spearman(x.head(10), y.head(10)))


def test_quintile_spread_is_top_minus_bottom():
    pred = pd.Series(np.arange(100, dtype=float))
    actual = pred / 10
    assert np.isclose(_quintile_spread(pred, actual), (89.5 - 9.5) / 10)


def _panel() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2022):
        for z in range(12):
            rows.append(
                {
                    "zip": f"{z:05d}",
                    "origin_year": year,
                    "metro": "M",
                    "metro_ok": True,
                    "is_nyc": z < 6,
                    "target_5y": 0.1 if year <= 2016 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_training_outcomes_are_realized_before_test_origin():
    folds = walk_forward_folds(_panel(), horizon=5)
    assert folds, "expected at least one fold"
    assert folds[0].test_year == config.FIRST_TEST_ORIGIN_YEAR
    for fold in folds:
        assert fold.train["origin_year"].max() <= fold.test_year - 5
        assert (fold.test["origin_year"] == fold.test_year).all()
    assert folds[-1].test_year == 2016


def test_one_year_folds_use_every_origin_through_last_realized_year():
    panel = _panel().rename(columns={"target_5y": "target_1y"})
    folds = walk_forward_folds(panel, horizon=1)
    assert [f.test_year for f in folds] == list(range(2010, 2017))
    for fold in folds:
        assert fold.train["origin_year"].max() == fold.test_year - 1


def test_baselines_cover_short_and_long_momentum():
    assert {"momentum_1y", "momentum_5y"} <= set(BASELINES)


def test_load_zhvf_picks_twelve_month_column(monkeypatch, tmp_path):
    path = tmp_path / "zhvf.csv"
    path.write_text(
        "RegionID,SizeRank,RegionName,RegionType,StateName,State,City,Metro,CountyName,"
        "BaseDate,2026-08-31,2026-10-31,2027-07-31\n"
        "1,1,8701,zip,NJ,NJ,Lakewood,NYC,Ocean,2026-07-31,0.1,0.5,2.3\n"
        "2,2,77494,zip,TX,TX,Katy,Houston,Fort Bend,2026-07-31,-0.1,-0.1,-0.6\n"
    )
    monkeypatch.setattr(config, "ZHVF_FILE", path)
    z = load_zhvf()
    assert z.loc["08701", "zillow_1y_forecast_pct"] == 2.3
    assert z.loc["77494", "zillow_1y_forecast_pct"] == -0.6


def test_load_zhvf_without_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ZHVF_FILE", tmp_path / "missing.csv")
    assert load_zhvf().empty


def test_ranking_json_is_ranked_and_nan_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    ranking = pd.DataFrame(
        {
            "rank": [1, 2],
            "zip": ["10001", "10002"],
            "city": ["New York", "New York"],
            "county": ["New York County", "New York County"],
            "state": ["NY", "NY"],
            "zhvi": [1e6, 9e5],
            "score": [70.0, 60.0],
            "nyc_percentile": [100.0, 50.0],
            "hist_relative_pct": [3.0, np.nan],
            "zillow_1y_forecast_pct": [np.nan, np.nan],
            "log_price_rel_metro": [0.0, np.log(0.9)],
            "mom_5y_rel": [0.0, 0.0],
            "origin": [pd.Timestamp("2026-07-31").date()] * 2,
        }
    )
    write_ranking_json(ranking, 2)
    doc = json.loads((tmp_path / "nyc_forecast_2y.json").read_text())
    assert doc["horizon_years"] == 2
    assert doc["spearman_score_vs_zillow_1y_forecast"] is None
    assert [z["rank"] for z in doc["zips"]] == [1, 2]
    assert doc["zips"][1]["historical_growth_vs_metro_pct"] is None
    assert doc["zips"][1]["price_vs_metro_median_pct"] == -10.0


def test_results_summary_lists_top_and_bottom_and_survives_missing_skill(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "RESULTS_FILE", tmp_path / "RESULTS.md")
    n = 40
    ranking = pd.DataFrame(
        {
            "rank": range(1, n + 1),
            "zip": [f"{10000 + i:05d}" for i in range(n)],
            "city": ["Town"] * n,
            "county": ["Kings County"] * n,
            "state": ["NY"] * n,
            "zhvi": np.linspace(2e6, 3e5, n),
            "score": np.linspace(80, 20, n),
            "hist_relative_pct": np.linspace(4, -4, n),
            "zillow_1y_forecast_pct": [np.nan] + [1.0] * (n - 1),
            "log_price_rel_metro": np.linspace(0.5, -0.5, n),
            "mom_1y_rel": np.linspace(0.05, -0.05, n),
            "origin": [pd.Timestamp("2026-07-31").date()] * n,
        }
    )
    ranking.loc[ranking.index[-2:], ["county", "city"]] = [
        ["Hudson County", "Hoboken"],
        ["Hudson County", "Jersey City"],
    ]
    write_results_summary(ranking, None)
    text = (tmp_path / "RESULTS.md").read_text()
    assert "Spearman" not in text
    assert "| 1 | 10000 |" in text and "| 40 | 10039 |" in text
    assert "| 16 | 10015 |" not in text and "| 25 | 10024 |" not in text
    assert "| n/a |" in text
    assert "### Manhattan, Brooklyn and Queens: top 15 of 38" in text
    assert "### Hudson County, NJ: top 2 of 2" in text
    assert "Jersey City alone has 1 ZIPs" in text
    assert "| 2 | 40 | **10039** |" in text

    summary = pd.DataFrame(
        {
            "horizon": [1, 1],
            "model": ["gbm_national", "momentum_1y"],
            "spearman_nyc_mean": [0.44, 0.39],
            "spearman_nyc_min": [-0.03, -0.02],
            "q5_q1_spread_nyc_mean": [0.056, 0.050],
            "n_folds": [16, 16],
            "folds_gbm_beats_best_baseline": [11.0, np.nan],
            "folds_compared": [16.0, np.nan],
            "gap_vs_best_baseline": [0.049, np.nan],
            "gap_ci_low": [0.008, np.nan],
            "gap_ci_high": [0.088, np.nan],
        }
    )
    write_results_summary(ranking, summary)
    text = (tmp_path / "RESULTS.md").read_text()
    assert "+0.44, worst year -0.03" in text
    assert "11 of 16 years" in text
    assert "+0.01 to +0.09" in text


def _results(rows):
    return pd.DataFrame(
        rows, columns=["horizon", "model", "test_year", "spearman_nyc", "spearman_national"]
    ).assign(q5_q1_spread_national=0.0, q5_q1_spread_nyc=0.0)


def test_summary_counts_wins_only_over_folds_both_models_scored():
    rows = [
        (1, "gbm_national", 2010, 0.5, 0.1),
        (1, "momentum_1y", 2010, 0.2, 0.1),
        (1, "gbm_national", 2011, 0.1, 0.1),
        (1, "momentum_1y", 2011, 0.4, 0.1),
        (1, "gbm_national", 2012, 0.3, 0.1),
        (1, "gbm_national", 2013, 0.3, 0.1),
        (1, "momentum_1y", 2013, np.nan, 0.1),
    ]
    s = summarize(_results(rows)).set_index("model")
    assert s.loc["gbm_national", "best_baseline"] == "momentum_1y"
    assert s.loc["gbm_national", "folds_gbm_beats_best_baseline"] == 1
    assert s.loc["gbm_national", "folds_compared"] == 2
    assert s.loc["gbm_national", "n_folds"] == 4
    assert np.isclose(s.loc["gbm_national", "gap_vs_best_baseline"], 0.0)


def test_block_bootstrap_ci_brackets_the_mean_and_shrinks_with_signal():
    noisy = pd.Series([0.4, -0.3, 0.5, -0.2, 0.3, -0.4, 0.2, -0.1])
    lo, hi = block_bootstrap_ci(noisy, block=1)
    assert lo < noisy.mean() < hi
    assert lo < 0 < hi
    steady = pd.Series([0.30, 0.32, 0.29, 0.31, 0.30, 0.33, 0.28, 0.31])
    lo, hi = block_bootstrap_ci(steady, block=2)
    assert 0.27 < lo <= steady.mean() <= hi < 0.34
    assert block_bootstrap_ci(pd.Series([0.1, np.nan]), block=1) == (np.nan, np.nan)
    assert block_bootstrap_ci(pd.Series(np.arange(7.0)), block=10) == (np.nan, np.nan)


def test_within_county_spearman_ignores_between_county_spread():
    n = 40
    rng = np.random.default_rng(0)
    within = rng.normal(size=2 * n)
    test = pd.DataFrame(
        {
            "county": ["Kings"] * n + ["Bergen"] * n,
            "target_1y": np.r_[within[:n] + 10, within[n:]],
        }
    )
    pred = pd.Series(np.r_[within[:n] - 10, within[n:]], index=test.index)
    assert _spearman(pred, test["target_1y"]) < 0
    assert np.isclose(_within_county_spearman(pred, test, "target_1y"), 1.0)


def test_beta_history_splits_zips_by_their_metro_direction():
    rng = np.random.default_rng(1)
    n = 60
    beta = rng.uniform(0.5, 1.5, size=2 * n)
    metro = ["up"] * n + ["down"] * n
    metro_growth = np.r_[np.full(n, 0.1), np.full(n, -0.1)]
    # ZIP growth = beta times metro growth: high beta wins in up metros, loses in down ones.
    fwd = beta * metro_growth
    panel = pd.DataFrame(
        {
            "origin_year": 2015,
            "metro": metro,
            "metro_ok": True,
            "is_nyc": [True] * n + [False] * n,
            "beta_10y": beta,
            "fwd_1y": fwd,
        }
    )
    panel["target_1y"] = fwd - panel.groupby("metro")["fwd_1y"].transform("mean")
    b = beta_history(panel, 1)
    assert len(b) == 1
    r = b.iloc[0]
    assert r.beta_spearman_up_metros > 0.99
    assert r.beta_spearman_down_metros < -0.99
    assert r.share_zips_in_up_metros == 0.5
    assert r.nyc_metro_growth > 0


def test_rank_drops_summary_from_a_different_panel(monkeypatch, tmp_path, caplog):
    from zipforecast.cli import _load_summary

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    panel = pd.DataFrame({"origin": pd.to_datetime(["2026-07-31", "2027-07-31"])})
    assert _load_summary(panel) is None
    summary = pd.DataFrame({"horizon": [1], "model": ["gbm_national"], "panel_origin": ""})
    summary.assign(panel_origin="2026-07-31").to_csv(tmp_path / "evaluation_summary.csv")
    assert _load_summary(panel) is None
    assert "2026-07-31" in caplog.text and "2027-07-31" in caplog.text
    summary.assign(panel_origin="2027-07-31").to_csv(tmp_path / "evaluation_summary.csv")
    assert len(_load_summary(panel)) == 1


def test_feature_groups_split_new_features_from_the_rest():
    def group(f):
        return [g for g, member in FEATURE_GROUPS.items() if member(f)]

    assert group("inventory_yoy") == ["listing_tempo"]
    assert group("inv_per_1k_units") == ["listing_tempo"]
    assert group("mom_1y_rel_county") == ["county"]
    assert group("beta_10y") == ["beta"]
    assert group("rg_rate_change_1y") == ["regime"]
    for base in ("mom_1y", "log_price_rel_metro", "rent_yield", "median_income"):
        assert group(base) == []
