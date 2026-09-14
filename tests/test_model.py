import numpy as np
import pandas as pd

from zipforecast import config
from zipforecast.model import _quintile_spread, _spearman, walk_forward_folds


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
