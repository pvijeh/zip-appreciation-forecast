import json

import numpy as np
import pandas as pd

from zipforecast import config
from zipforecast.model import BASELINES, _quintile_spread, _spearman, load_zhvf, walk_forward_folds
from zipforecast.report import write_ranking_json


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
