"""Walk-forward evaluation and final ranking of ZIPs by expected relative appreciation."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zipforecast import config
from zipforecast.panel import model_features

log = logging.getLogger(__name__)

RANDOM_STATE = 7


def make_gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def make_ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


BASELINES = {
    # Continue the ZIP's own outperformance of its metro over the past five years.
    "momentum_5y": lambda df: df["mom_5y_rel"],
    # Cheaper-than-metro ZIPs catch up.
    "catch_up_price": lambda df: -df["log_price_rel_metro"],
    # Cheaper-than-neighbors ZIPs catch up.
    "catch_up_neighbors": lambda df: -df["nbr_log_price_gap"],
}


@dataclass
class Fold:
    horizon: int
    test_year: int
    train: pd.DataFrame
    test: pd.DataFrame


def walk_forward_folds(panel: pd.DataFrame, horizon: int) -> list[Fold]:
    target = f"target_{horizon}y"
    usable = panel[panel["metro_ok"] & panel[target].notna()]
    years = sorted(usable["origin_year"].unique())
    folds = []
    for test_year in years:
        if test_year < config.FIRST_TEST_ORIGIN_YEAR:
            continue
        # Training targets must be fully realized before the test origin.
        train = usable[usable["origin_year"] <= test_year - horizon]
        test = usable[usable["origin_year"] == test_year]
        if train.empty or test.empty:
            continue
        folds.append(Fold(horizon, test_year, train, test))
    return folds


def _spearman(pred: pd.Series, actual: pd.Series) -> float:
    mask = pred.notna() & actual.notna()
    if mask.sum() < 20:
        return np.nan
    return float(spearmanr(pred[mask], actual[mask]).statistic)


def _quintile_spread(pred: pd.Series, actual: pd.Series) -> float:
    """Realized relative growth of the predicted top fifth minus the predicted bottom fifth."""
    mask = pred.notna() & actual.notna()
    if mask.sum() < 50:
        return np.nan
    p, a = pred[mask], actual[mask]
    hi, lo = p.quantile(0.8), p.quantile(0.2)
    return float(a[p >= hi].mean() - a[p <= lo].mean())


def _score(name: str, pred: pd.Series, test: pd.DataFrame, target: str) -> dict:
    nyc = test["is_nyc"]
    return {
        "model": name,
        "spearman_national": _spearman(pred, test[target]),
        "spearman_nyc": _spearman(pred[nyc], test.loc[nyc, target]),
        "q5_q1_spread_national": _quintile_spread(pred, test[target]),
        "q5_q1_spread_nyc": _quintile_spread(pred[nyc], test.loc[nyc, target]),
        "n_test": int(test[target].notna().sum()),
        "n_test_nyc": int(test.loc[nyc, target].notna().sum()),
    }


def _usable_features(train: pd.DataFrame, features: list[str]) -> list[str]:
    """Features with at least two distinct values in the training rows; early origins have
    no ACS columns at all, and the gradient boosting binner rejects an all-NaN column."""
    return [c for c in features if train[c].nunique(dropna=True) >= 2]


def _recency_weights(train: pd.DataFrame) -> np.ndarray:
    """Halve an origin's weight every RECENCY_HALF_LIFE_YEARS before the newest training origin."""
    age = train["origin_year"].max() - train["origin_year"]
    return np.power(0.5, age / config.RECENCY_HALF_LIFE_YEARS).to_numpy()


def _fit_predict(
    model, train: pd.DataFrame, test: pd.DataFrame, features, horizon: int, recency: bool = False
):
    """Fit on the within-metro percentile target; predictions are expected percentiles."""
    fit_target = f"target_{horizon}y_pct"
    train = train[train[fit_target].notna()]
    features = _usable_features(train, features)
    kwargs = {"sample_weight": _recency_weights(train)} if recency else {}
    model.fit(train[features], train[fit_target], **kwargs)
    return pd.Series(model.predict(test[features]), index=test.index)


def evaluate(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    features = model_features(panel)
    target = f"target_{horizon}y"
    rows = []
    for fold in walk_forward_folds(panel, horizon):
        test = fold.test
        log.info(
            "h=%d test origin %d: train %d rows (origins <= %d), test %d rows (%d NYC)",
            horizon,
            fold.test_year,
            len(fold.train),
            fold.test_year - horizon,
            len(test),
            int(test["is_nyc"].sum()),
        )
        preds = {
            "gbm_national": _fit_predict(make_gbm(), fold.train, test, features, horizon),
            "gbm_recency": _fit_predict(
                make_gbm(), fold.train, test, features, horizon, recency=True
            ),
            "ridge_national": _fit_predict(make_ridge(), fold.train, test, features, horizon),
        }
        nyc_train = fold.train[fold.train["is_nyc"]]
        if nyc_train[target].notna().sum() >= 500:
            preds["gbm_nyc_only"] = _fit_predict(make_gbm(), nyc_train, test, features, horizon)
        for name, fn in BASELINES.items():
            preds[name] = fn(test)
        for name, pred in preds.items():
            row = _score(name, pred, test, target)
            row.update(horizon=horizon, test_year=fold.test_year, n_train=len(fold.train))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["spearman_national", "spearman_nyc", "q5_q1_spread_national", "q5_q1_spread_nyc"]
    grouped = results.groupby(["horizon", "model"])
    summary = grouped[metrics].mean()
    summary.columns = [f"{c}_mean" for c in metrics]
    for m in metrics:
        summary[f"{m}_min"] = grouped[m].min()
    summary["n_folds"] = grouped.size()
    summary["share_folds_gbm_beats_momentum"] = np.nan
    for (h, model), _ in summary.iterrows():
        if model == "gbm_national":
            g = results[(results.horizon == h) & (results.model == "gbm_national")].set_index(
                "test_year"
            )["spearman_nyc"]
            b = results[(results.horizon == h) & (results.model == "momentum_5y")].set_index(
                "test_year"
            )["spearman_nyc"]
            both = pd.concat([g, b], axis=1, keys=["g", "b"]).dropna()
            summary.loc[(h, model), "share_folds_gbm_beats_momentum"] = (
                (both["g"] > both["b"]).mean() if len(both) else np.nan
            )
    return summary.reset_index()


FACTOR_HISTORY_FEATURES = [
    "log_price_rel_metro",
    "nbr_log_price_gap",
    "price_to_income",
    "rent_yield",
    "log_income",
    "bachelors_share",
    "mom_1y",
    "mom_5y",
    "unemployment_rate",
]


def factor_history(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Spearman between each single feature and realized relative growth, by origin year.

    Rows are origins with a realized outcome; a sign that flips between rows is the reason a
    model fit on earlier origins can rank later ones backwards."""
    target = f"target_{horizon}y"
    usable = panel[panel["metro_ok"] & panel[target].notna()]
    rows = []
    for year, g in usable.groupby("origin_year"):
        row = {"origin_year": int(year), "n": len(g)}
        for c in FACTOR_HISTORY_FEATURES:
            row[c] = _spearman(g[f"cs_{c}"], g[target])
            row[f"{c}_nyc"] = _spearman(g.loc[g["is_nyc"], f"cs_{c}"], g.loc[g["is_nyc"], target])
        rows.append(row)
    return pd.DataFrame(rows)


def feature_importance(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Permutation importance on the most recent walk-forward fold, national test set."""
    features = model_features(panel)
    target = f"target_{horizon}y_pct"
    folds = walk_forward_folds(panel, horizon)
    if not folds:
        return pd.DataFrame()
    fold = folds[-1]
    model = make_gbm()
    train = fold.train[fold.train[target].notna()]
    features = _usable_features(train, features)
    model.fit(train[features], train[target])
    test = fold.test[fold.test[target].notna()]
    if len(test) > 20000:
        test = test.sample(20000, random_state=RANDOM_STATE)
    imp = permutation_importance(
        model,
        test[features],
        test[target],
        n_repeats=3,
        random_state=RANDOM_STATE,
        scoring="neg_mean_squared_error",
    )
    return (
        pd.DataFrame(
            {
                "feature": features,
                "importance_mean": imp.importances_mean,
                "importance_std": imp.importances_std,
                "test_year": fold.test_year,
                "horizon": horizon,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _percentile_to_relative_growth(
    train: pd.DataFrame, target: str, pct: pd.Series, bins: int = 20
) -> pd.Series:
    """Median realized metro-relative log growth of NYC ZIPs that landed in each percentile
    band of their metro, over all realized training origins."""
    nyc = train[train["is_nyc"] & train[target].notna()]
    band = np.minimum((nyc[f"{target}_pct"] * bins).astype(int), bins - 1)
    by_band = nyc.groupby(band)[target].median().reindex(range(bins)).interpolate()
    return pd.Series(by_band.to_numpy()[np.minimum((pct * bins).astype(int), bins - 1)], pct.index)


RANKING_CONTEXT = [
    "zip",
    "city",
    "county",
    "state",
    "zhvi",
    "mom_5y_rel",
    "log_price_rel_metro",
    "nbr_log_price_gap",
    "rent_yield",
    "price_to_income_rel",
    "bachelors_share_rel",
    "d5_log_income_rel",
    "d5_log_housing_units_rel",
]


def rank_nyc(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fit on every fully realized origin, score the latest origin, keep NYC-metro ZIPs.

    The score is the model's expected within-metro percentile of forward growth (0-100). The
    last column converts that into log growth relative to the metro using the realized spread
    between percentiles in the training data, so the number is a historical analogue, not a
    forecast of the metro's own path.
    """
    features = model_features(panel)
    target = f"target_{horizon}y"
    latest_year = panel["origin_year"].max()
    train = panel[panel["metro_ok"] & panel[target].notna()]
    latest = panel[(panel["origin_year"] == latest_year) & panel["is_nyc"]].copy()
    log.info(
        "h=%d final fit: %d rows from origins %d..%d, scoring %d NYC ZIPs at %s",
        horizon,
        len(train),
        train["origin_year"].min(),
        train["origin_year"].max(),
        len(latest),
        latest["origin"].iloc[0].date(),
    )
    features = _usable_features(train, features)
    fit_target = f"{target}_pct"
    gbm = make_gbm().fit(train[features], train[fit_target])
    ridge = make_ridge().fit(train[features], train[fit_target])
    latest["pred_gbm"] = 100 * gbm.predict(latest[features])
    latest["pred_ridge"] = 100 * ridge.predict(latest[features])
    latest["score"] = 0.5 * (latest["pred_gbm"] + latest["pred_ridge"])
    latest["rank"] = latest["score"].rank(ascending=False, method="first").astype(int)
    latest["nyc_percentile"] = 100 * latest["score"].rank(pct=True)
    latest["hist_relative_pct"] = 100 * (
        np.exp(_percentile_to_relative_growth(train, target, latest["score"] / 100)) - 1
    )
    latest["zhvi"] = np.exp(latest["log_zhvi"]).round(0)
    latest["origin"] = latest["origin"].dt.date
    cols = [
        "rank",
        "score",
        "nyc_percentile",
        "hist_relative_pct",
        "pred_gbm",
        "pred_ridge",
        *RANKING_CONTEXT,
        "origin",
    ]
    cols = [c for c in cols if c in latest.columns]
    return latest.sort_values("rank")[cols].reset_index(drop=True)
