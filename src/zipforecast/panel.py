"""Build the ZIP x origin-year panel: features known at the origin, forward relative growth."""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from zipforecast import config

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
ID_COLUMNS = ["zip", "origin", "origin_year", "metro", "state", "county", "city"]


def _load_zillow_wide(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (meta, values) with values indexed by zip, columns = month-end Timestamps."""
    raw = pd.read_csv(path, dtype={"RegionName": str})
    raw["zip"] = raw["RegionName"].str.zfill(5)
    month_cols = [c for c in raw.columns if c[:2] in ("19", "20") and c[4] == "-"]
    meta = raw.set_index("zip")[["State", "City", "Metro", "CountyName"]].rename(
        columns={"State": "state", "City": "city", "Metro": "metro", "CountyName": "county"}
    )
    values = raw.set_index("zip")[month_cols]
    values.columns = pd.to_datetime(values.columns)
    return meta, values


def _origins(values: pd.DataFrame) -> list[pd.Timestamp]:
    """Yearly origins on the calendar month of the latest Zillow release, 2000 onward."""
    last = values.columns.max()
    return [c for c in values.columns if c.month == last.month and c.year >= 2000]


def _log_change(values: pd.DataFrame, origin: pd.Timestamp, years: int) -> pd.Series:
    past = origin - pd.DateOffset(years=years)
    past = past + pd.offsets.MonthEnd(0)
    if past not in values.columns:
        return pd.Series(np.nan, index=values.index)
    return np.log(values[origin]) - np.log(values[past])


def _forward_log_change(values: pd.DataFrame, origin: pd.Timestamp, years: int) -> pd.Series:
    future = origin + pd.DateOffset(years=years)
    future = future + pd.offsets.MonthEnd(0)
    if future not in values.columns:
        return pd.Series(np.nan, index=values.index)
    return np.log(values[future]) - np.log(values[origin])


def _demean_by(series: pd.Series, groups: pd.Series) -> pd.Series:
    return series - series.groupby(groups).transform("mean")


def _neighbor_features(frame: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Gap between a ZIP and its k nearest ZIPs (by ZCTA centroid) on price level and momentum."""
    out = pd.DataFrame(index=frame.index)
    out["nbr_log_price_gap"] = np.nan
    out["nbr_mom_5y_gap"] = np.nan
    out["nbr_dist_km"] = np.nan
    have = frame.index.intersection(coords.index)
    have = have[frame.loc[have, "log_zhvi"].notna()]
    if len(have) <= config.NEIGHBOR_COUNT:
        return out
    xy = np.deg2rad(coords.loc[have, ["lat", "lon"]].to_numpy())
    tree = BallTree(xy, metric="haversine")
    dist, idx = tree.query(xy, k=config.NEIGHBOR_COUNT + 1)
    dist, idx = dist[:, 1:] * EARTH_RADIUS_KM, idx[:, 1:]
    log_price = frame.loc[have, "log_zhvi"].to_numpy()
    mom = frame.loc[have, "mom_5y"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out.loc[have, "nbr_log_price_gap"] = log_price - np.nanmean(log_price[idx], axis=1)
        out.loc[have, "nbr_mom_5y_gap"] = mom - np.nanmean(mom[idx], axis=1)
    out.loc[have, "nbr_dist_km"] = dist.mean(axis=1)
    return out


def _acs_features(acs: pd.DataFrame) -> pd.DataFrame:
    """Ratios from one ACS vintage, indexed by zcta."""
    a = acs.set_index("zcta")
    f = pd.DataFrame(index=a.index)
    f["log_pop"] = np.log(a["pop"].where(a["pop"] > 0))
    f["log_income"] = np.log(a["median_income"].where(a["median_income"] > 0))
    f["log_acs_rent"] = np.log(a["acs_median_rent"].where(a["acs_median_rent"] > 0))
    f["log_housing_units"] = np.log(a["housing_units"].where(a["housing_units"] > 0))
    f["median_age"] = a["median_age"]
    f["median_year_built"] = a["median_year_built"].where(a["median_year_built"] > 1000)
    f["young_adult_share"] = (
        a[["male_25_29", "male_30_34", "female_25_29", "female_30_34"]].sum(axis=1) / a["pop"]
    )
    f["owner_share"] = a["owner_occupied"] / a["occupied_units"]
    f["vacancy_share"] = a["vacant_units"] / a["housing_units"]
    f["bachelors_share"] = a["edu_bachelors_plus"] / a["edu_total"]
    f["inflow_share"] = (
        a[["moved_other_county", "moved_other_state", "moved_abroad"]].sum(axis=1)
        / a["mobility_total"]
    )
    f["unemployment_rate"] = a["unemployed"] / a["labor_force"]
    f["transit_share"] = a["transit_commuters"] / a["commuters"]
    return f.replace([np.inf, -np.inf], np.nan)


def _load_acs() -> dict[int, pd.DataFrame]:
    out = {}
    for path in sorted(config.ACS_DIR.glob("acs5_*.parquet")):
        year = int(path.stem.split("_")[1])
        out[year] = _acs_features(pd.read_parquet(path))
    return out


def _acs_vintage_for(origin_year: int, available: list[int]) -> int | None:
    latest_safe = origin_year - config.ACS_PUBLICATION_LAG_YEARS
    usable = [y for y in available if y <= latest_safe]
    return max(usable) if usable else None


ACS_CHANGE_COLUMNS = [
    "log_pop",
    "log_income",
    "log_housing_units",
    "bachelors_share",
    "young_adult_share",
    "inflow_share",
    "owner_share",
]


def build_panel() -> pd.DataFrame:
    meta, zhvi = _load_zillow_wide(config.ZHVI_FILE)
    _, zori = _load_zillow_wide(config.ZORI_FILE)
    zori = zori.reindex(index=zhvi.index)

    gaz = pd.read_csv(config.GAZETTEER_FILE, sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    coords = gaz.set_index(gaz["GEOID"].str.zfill(5))[["INTPTLAT", "INTPTLONG"]]
    coords.columns = ["lat", "lon"]

    acs = _load_acs()
    acs_years = sorted(acs)
    log.info("ACS vintages: %s", acs_years)

    origins = _origins(zhvi)
    log.info("origins: %s .. %s (%d)", origins[0].date(), origins[-1].date(), len(origins))

    frames = []
    for origin in origins:
        f = meta.copy()
        f["origin"] = origin
        f["origin_year"] = origin.year
        f["log_zhvi"] = np.log(zhvi[origin])
        f = f[f["log_zhvi"].notna() & f["metro"].notna()]
        metro = f["metro"]

        for k in (1, 3, 5):
            f[f"mom_{k}y"] = _log_change(zhvi, origin, k).reindex(f.index)
            f[f"mom_{k}y_rel"] = _demean_by(f[f"mom_{k}y"], metro)
        f["log_price_rel_metro"] = f["log_zhvi"] - f.groupby("metro")["log_zhvi"].transform(
            "median"
        )
        f["metro_log_price_median"] = f.groupby("metro")["log_zhvi"].transform("median")
        f["metro_log_price_spread"] = f.groupby("metro")["log_zhvi"].transform("std")
        f["metro_mom_5y"] = f.groupby("metro")["mom_5y"].transform("mean")
        f["metro_mom_1y"] = f.groupby("metro")["mom_1y"].transform("mean")
        f["metro_log_n_zips"] = np.log(f.groupby("metro")["log_zhvi"].transform("size"))

        if origin in zori.columns:
            rent_yield = 12 * zori[origin].reindex(f.index) / zhvi[origin].reindex(f.index)
            f["rent_yield"] = rent_yield
            f["rent_yield_rel"] = _demean_by(rent_yield, metro)
            f["rent_mom_3y"] = _log_change(zori, origin, 3).reindex(f.index)
            f["rent_mom_3y_rel"] = _demean_by(f["rent_mom_3y"], metro)
        else:
            for col in ("rent_yield", "rent_yield_rel", "rent_mom_3y", "rent_mom_3y_rel"):
                f[col] = np.nan

        f = f.join(_neighbor_features(f, coords))

        vintage = _acs_vintage_for(origin.year, acs_years)
        f["acs_vintage"] = vintage
        if vintage is not None:
            cur = acs[vintage].reindex(f.index)
            f = f.join(cur)
            f["price_to_income"] = f["log_zhvi"] - f["log_income"]
            f["price_to_income_rel"] = _demean_by(f["price_to_income"], metro)
            f["log_income_rel"] = _demean_by(f["log_income"], metro)
            f["bachelors_share_rel"] = _demean_by(f["bachelors_share"], metro)
            f["young_adult_share_rel"] = _demean_by(f["young_adult_share"], metro)
            f["acs_rent_to_price"] = f["log_acs_rent"] + np.log(12) - f["log_zhvi"]
            prev_vintage = vintage - 5
            if prev_vintage in acs:
                prev = acs[prev_vintage].reindex(f.index)
                for col in ACS_CHANGE_COLUMNS:
                    f[f"d5_{col}"] = cur[col] - prev[col]
                    f[f"d5_{col}_rel"] = _demean_by(f[f"d5_{col}"], metro)

        for h in config.HORIZONS:
            fwd = _forward_log_change(zhvi, origin, h).reindex(f.index)
            f[f"fwd_{h}y"] = fwd
            f[f"target_{h}y"] = _demean_by(fwd, metro)
        frames.append(f.reset_index())

    panel = pd.concat(frames, ignore_index=True)
    n_metro = panel.groupby(["origin", "metro"])["zip"].transform("size")
    panel["metro_ok"] = n_metro >= config.MIN_ZIPS_PER_METRO
    panel["is_nyc"] = panel["metro"] == config.NYC_METRO
    panel = _add_cross_sectional_ranks(panel)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PANEL_FILE, index=False)
    log.info("panel: %d rows, %d columns -> %s", len(panel), panel.shape[1], config.PANEL_FILE)
    return panel


def _add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each ZIP-level feature and target within its (origin, metro) group.

    Price levels, momentum and incomes drift a lot from one decade to the next; a model fit on
    2000-2005 raw values sees numbers it never met when scoring 2026. Within-metro percentiles
    keep only the cross-sectional information, which is all a metro-relative target can use.
    """
    groups = panel.groupby(["origin", "metro"], sort=False)
    # Demeaned "_rel" columns rank identically to their raw counterparts within a metro.
    zip_level = [
        c
        for c in raw_feature_columns(panel)
        if not c.startswith("metro_") and not c.endswith("_rel")
    ]
    ranked = groups[zip_level].rank(pct=True)
    ranked.columns = [f"cs_{c}" for c in zip_level]
    targets = {}
    for h in config.HORIZONS:
        targets[f"target_{h}y_pct"] = groups[f"fwd_{h}y"].rank(pct=True)
    return pd.concat([panel, ranked, pd.DataFrame(targets)], axis=1)


def raw_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Features in original units, used by the rule-of-thumb baselines and the ranking table."""
    skip = set(ID_COLUMNS) | {"metro_ok", "is_nyc", "acs_vintage"}
    return [
        c
        for c in panel.columns
        if c not in skip
        and not c.startswith(("fwd_", "target_", "cs_"))
        and panel[c].dtype != object
    ]


def model_features(panel: pd.DataFrame) -> list[str]:
    """Within-metro percentile features the models are fit on."""
    return [c for c in panel.columns if c.startswith("cs_")]
