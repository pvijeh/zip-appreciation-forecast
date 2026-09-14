import numpy as np
import pandas as pd

from zipforecast import config
from zipforecast.panel import (
    _acs_vintage_for,
    _demean_by,
    _forward_log_change,
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
