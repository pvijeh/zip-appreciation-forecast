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


def _month_end(origin: pd.Timestamp, years: int) -> pd.Timestamp:
    return origin - pd.DateOffset(years=years) + pd.offsets.MonthEnd(0)


def _beta_to_metro(zhvi: pd.DataFrame, origin: pd.Timestamp, metro: pd.Series) -> pd.Series:
    """Slope of a ZIP's annual log growth on its metro's mean annual log growth, over the
    BETA_WINDOW_YEARS years ending at the origin. Above 1: the ZIP amplifies its metro's
    swings in both directions; below 1: it damps them."""
    cols = []
    for k in range(config.BETA_WINDOW_YEARS):
        end, start = _month_end(origin, k), _month_end(origin, k + 1)
        if end in zhvi.columns and start in zhvi.columns:
            cols.append(np.log(zhvi[end]) - np.log(zhvi[start]))
    if len(cols) < config.BETA_MIN_YEARS:
        return pd.Series(np.nan, index=metro.index)
    g = pd.concat(cols, axis=1).reindex(metro.index)
    m = g.groupby(metro).transform("mean")
    g = g.where(m.notna())
    m = m.where(g.notna())
    gc = g.sub(g.mean(axis=1), axis=0)
    mc = m.sub(m.mean(axis=1), axis=0)
    beta = (gc * mc).sum(axis=1, min_count=1) / (mc * mc).sum(axis=1, min_count=1)
    beta[g.notna().sum(axis=1) < config.BETA_MIN_YEARS] = np.nan
    return beta.replace([np.inf, -np.inf], np.nan)


def _load_fred(name: str) -> pd.Series:
    """One FRED series as a month-end indexed Series (last observation in each month)."""
    raw = pd.read_csv(config.FRED_FILES[name], na_values=".")
    raw.columns = ["date", "value"]
    s = pd.Series(raw["value"].to_numpy(), index=pd.to_datetime(raw["date"])).dropna()
    return s.groupby(s.index + pd.offsets.MonthEnd(0)).last()


def _macro_regime(zhvi: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    """Per-origin national series the model can use to tell which phase of the cycle it is in.

    These are the same for every ZIP at an origin, so they carry no ranking information on
    their own; they only help through interactions with ZIP-level features."""
    if not all(p.exists() for p in config.FRED_FILES.values()):
        log.warning("FRED files missing; regime features left empty")
        return pd.DataFrame(index=pd.Index(origins, name="origin"))
    rate = _load_fred("mortgage_rate")
    cpi = _load_fred("rent_cpi")
    natl = np.log(zhvi.median(axis=0))
    price_to_rent = natl - np.log(cpi.reindex(natl.index))
    window = 12 * config.PRICE_TO_RENT_MEAN_YEARS
    dev = price_to_rent - price_to_rent.rolling(window, min_periods=36).mean()
    out = pd.DataFrame(index=pd.Index(origins, name="origin"))
    out["rg_rate_level"] = rate.reindex(origins).to_numpy()
    out["rg_rate_change_1y"] = (
        out["rg_rate_level"] - rate.reindex([_month_end(o, 1) for o in origins]).to_numpy()
    )
    out["rg_natl_mom_1y"] = [
        natl.get(o, np.nan) - natl.get(_month_end(o, 1), np.nan) for o in origins
    ]
    out["rg_price_to_rent_dev"] = dev.reindex(origins).to_numpy()
    return out


def _load_tempo() -> dict[str, pd.DataFrame]:
    out = {}
    for name, path in config.ZILLOW_TEMPO_FILES.items():
        if path.exists():
            out[name] = _load_zillow_wide(path)[1]
        else:
            log.warning("%s missing; market-tempo features left empty", path.name)
    return out


TEMPO_COLUMNS = [
    "inventory_yoy",
    "new_listings_yoy",
    "inv_to_new_listings",
    "days_pending",
    "days_pending_yoy",
    "price_cut_share",
    "price_cut_share_yoy",
]


def _tempo_features(
    tempo: dict[str, pd.DataFrame], origin: pd.Timestamp, index: pd.Index
) -> pd.DataFrame:
    """Zillow listing-market series at the origin: how fast homes sell and how much supply."""
    f = pd.DataFrame(np.nan, index=index, columns=TEMPO_COLUMNS)
    prev = _month_end(origin, 1)

    def at(name, when):
        v = tempo.get(name)
        if v is None or when not in v.columns:
            return pd.Series(np.nan, index=index)
        return v[when].reindex(index)

    inv, inv_prev = at("inventory", origin), at("inventory", prev)
    new, new_prev = at("new_listings", origin), at("new_listings", prev)
    f["inventory_yoy"] = np.log(inv.where(inv > 0)) - np.log(inv_prev.where(inv_prev > 0))
    f["new_listings_yoy"] = np.log(new.where(new > 0)) - np.log(new_prev.where(new_prev > 0))
    f["inv_to_new_listings"] = inv / new.where(new > 0)
    f["days_pending"] = at("days_pending", origin)
    f["days_pending_yoy"] = f["days_pending"] - at("days_pending", prev)
    f["price_cut_share"] = at("price_cut_share", origin)
    f["price_cut_share_yoy"] = f["price_cut_share"] - at("price_cut_share", prev)
    return f.replace([np.inf, -np.inf], np.nan)


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
    tempo = _load_tempo()
    macro = _macro_regime(zhvi, origins)

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
        f["beta_10y"] = _beta_to_metro(zhvi, origin, metro)

        # County within metro: Manhattan and Pike County PA share a metro but little else.
        county = f["metro"] + "|" + f["county"].fillna("")
        county_price = f.groupby(county)["log_zhvi"].transform("median")
        f["log_price_rel_county"] = f["log_zhvi"] - county_price
        f["county_log_price_rel_metro"] = county_price - f["metro_log_price_median"]
        f["mom_1y_rel_county"] = _demean_by(f["mom_1y"], county)
        f["county_mom_1y_rel"] = f.groupby(county)["mom_1y"].transform("mean") - f["metro_mom_1y"]
        f["county_mom_5y_rel"] = f.groupby(county)["mom_5y"].transform("mean") - f["metro_mom_5y"]

        f = f.join(_tempo_features(tempo, origin, f.index))
        for col in macro.columns:
            f[col] = macro.loc[origin, col]

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
            inv = tempo.get("inventory")
            if inv is not None and origin in inv.columns:
                f["inv_per_1k_units"] = (
                    1000 * inv[origin].reindex(f.index) / np.exp(f["log_housing_units"])
                )
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
    panel = _add_trailing_factor_signs(panel)
    panel = _add_cross_sectional_ranks(panel)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PANEL_FILE, index=False)
    log.info("panel: %d rows, %d columns -> %s", len(panel), panel.shape[1], config.PANEL_FILE)
    return panel


def _grouped_spearman(a: pd.Series, b: pd.Series, groups) -> pd.Series:
    """Spearman of a and b within each group, as a Series indexed by group."""
    mask = a.notna() & b.notna()
    ra = a[mask].groupby(groups[mask]).rank()
    rb = b[mask].groupby(groups[mask]).rank()
    g = groups[mask]
    ra = ra - ra.groupby(g).transform("mean")
    rb = rb - rb.groupby(g).transform("mean")
    cov = (ra * rb).groupby(g).sum()
    var = np.sqrt((ra * ra).groupby(g).sum() * (rb * rb).groupby(g).sum())
    n = mask.groupby(groups).sum()
    return (cov / var.where(var > 0)).where(n >= 20)


TRAILING_FACTORS = {
    # Did cheaper-than-metro ZIPs beat their metro over the past year?
    "cheap": lambda p: -p["log_price_rel_metro"],
    # Did last year's winners keep winning?
    "mom": lambda p: p["mom_1y_rel"],
}


def _add_trailing_factor_signs(panel: pd.DataFrame) -> pd.DataFrame:
    """Realized one-year Spearman of the cheap and momentum factors, lagged one origin so
    it is known at forecast time: nationally, over the last three years, and per metro."""
    ok = panel["metro_ok"]
    year = panel["origin_year"]
    metro_year = panel["metro"] + "|" + year.astype(str)
    for name, fn in TRAILING_FACTORS.items():
        factor = fn(panel).where(ok)
        realized = panel["fwd_1y"].where(ok)
        national = _grouped_spearman(factor, realized, year)
        national.index = national.index + 1
        panel[f"rg_{name}_sign_1y"] = year.map(national)
        rolling = national.rolling(3, min_periods=2).mean()
        panel[f"rg_{name}_sign_3y"] = year.map(rolling)
        by_metro = _grouped_spearman(factor, realized, metro_year)
        shifted = {}
        for key, v in by_metro.items():
            m, y = key.rsplit("|", 1)
            shifted[f"{m}|{int(y) + 1}"] = v
        panel[f"rg_metro_{name}_sign_1y"] = metro_year.map(shifted)
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
        if not c.startswith(("metro_", "rg_")) and not c.endswith("_rel")
    ]
    ranked = groups[zip_level].rank(pct=True)
    ranked.columns = [f"cs_{c}" for c in zip_level]
    # County-neutral target variant: percentile within (origin, metro, county).
    county_groups = panel.groupby(["origin", "metro", panel["county"].fillna("")], sort=False)
    targets = {}
    for h in config.HORIZONS:
        targets[f"target_{h}y_pct"] = groups[f"fwd_{h}y"].rank(pct=True)
        targets[f"target_{h}y_county_pct"] = county_groups[f"fwd_{h}y"].rank(pct=True)
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
    """Within-metro percentile features plus the raw regime series the models are fit on."""
    return [c for c in panel.columns if c.startswith(("cs_", "rg_"))]
