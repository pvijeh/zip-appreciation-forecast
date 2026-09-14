import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from zipforecast import config
from zipforecast.panel import (
    _acs_vintage_for,
    _beta_to_metro,
    _demean_by,
    _forward_log_change,
    _grouped_spearman,
    _load_fred,
    _log_change,
    _origins,
)


def _values() -> pd.DataFrame:
    months = pd.date_range("2000-01-31", "2010-12-31", freq="ME")
    base = np.linspace(100, 200, len(months))
    data = {m: [b, 2 * b] for m, b in zip(months, base, strict=True)}
    return pd.DataFrame(data, index=["a", "b"])


def test_origins_use_latest_month_from_2000():
    origins = _origins(_values())
    assert origins[0] == pd.Timestamp("2000-12-31")
    assert origins[-1] == pd.Timestamp("2010-12-31")
    assert all(o.month == 12 for o in origins)


def test_log_change_and_forward_are_inverse_views():
    v = _values()
    origin = pd.Timestamp("2005-12-31")
    back = _log_change(v, origin, 3)
    fwd = _forward_log_change(v, pd.Timestamp("2002-12-31"), 3)
    pd.testing.assert_series_equal(back, fwd)
    assert np.isclose(back["a"], np.log(v.loc["a", origin] / v.loc["a", "2002-12-31"]))


def test_log_change_missing_month_is_nan():
    v = _values()
    assert _log_change(v, pd.Timestamp("2001-12-31"), 5).isna().all()
    assert _forward_log_change(v, pd.Timestamp("2008-12-31"), 5).isna().all()


def test_demean_by_group():
    s = pd.Series([1.0, 3.0, 10.0, 20.0])
    g = pd.Series(["x", "x", "y", "y"])
    assert _demean_by(s, g).tolist() == [-1.0, 1.0, -5.0, 5.0]


def test_acs_vintage_respects_publication_lag():
    years = [2011, 2012, 2013]
    lag = config.ACS_PUBLICATION_LAG_YEARS
    assert _acs_vintage_for(2011 + lag - 1, years) is None
    assert _acs_vintage_for(2011 + lag, years) == 2011
    assert _acs_vintage_for(2030, years) == 2013


def test_beta_to_metro_recovers_slope_and_needs_enough_years():
    months = pd.date_range("2000-01-31", "2012-12-31", freq="ME")
    metro_growth = np.tile([0.05, -0.02, 0.08, 0.01, -0.04, 0.06, 0.03], 3)[: len(months) // 12]
    zips = {"steady": 1.0, "amplifier": 1.8, "damper": 0.2}
    rows = {}
    for name, beta in zips.items():
        annual = np.r_[0, beta * metro_growth[: len(months) // 12 - 1]]
        levels = np.exp(np.cumsum(np.repeat(annual, 12) / 12))
        rows[name] = 100 * levels[: len(months)]
    values = pd.DataFrame(rows, index=months).T
    metro = pd.Series("m", index=values.index)
    beta = _beta_to_metro(values, pd.Timestamp("2012-12-31"), metro)
    assert beta["amplifier"] > beta["steady"] > beta["damper"]
    assert beta["amplifier"] > 1.4 and beta["damper"] < 0.5
    early = _beta_to_metro(values, pd.Timestamp("2004-12-31"), metro)
    assert early.isna().all()


def test_grouped_spearman_matches_scipy_and_skips_small_groups():
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(size=70))
    b = pd.Series(a + rng.normal(size=70))
    groups = pd.Series(["big"] * 55 + ["small"] * 15)
    out = _grouped_spearman(a, b, groups)
    expected = spearmanr(a[:55], b[:55]).statistic
    assert np.isclose(out["big"], expected)
    assert np.isnan(out["small"])


def test_load_fred_takes_last_observation_in_month(monkeypatch, tmp_path):
    path = tmp_path / "fred.csv"
    path.write_text("observation_date,MORTGAGE30US\n2020-01-02,3.7\n2020-01-30,3.5\n2020-02-06,.\n")
    monkeypatch.setattr(config, "FRED_FILES", {"mortgage_rate": path})
    s = _load_fred("mortgage_rate")
    assert list(s.index) == [pd.Timestamp("2020-01-31")]
    assert s.iloc[0] == 3.5
