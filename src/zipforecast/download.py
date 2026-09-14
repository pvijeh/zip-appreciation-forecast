"""Fetch the raw inputs: Zillow series by ZIP, FRED macro series, Census ACS 5-year by ZCTA,
ZCTA centroids."""

import io
import logging
import os
import zipfile
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import pandas as pd

from zipforecast import config

log = logging.getLogger(__name__)

CENSUS_API = "https://api.census.gov/data/{year}/acs/acs5"
ACS_MISSING_SENTINEL = -666666666


def census_api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY", "")
    # Strip stray punctuation from a pasted key; the API rejects anything but the 40 hex chars.
    return key.strip().rstrip(".,;")


def _remote_last_modified(url: str, client: httpx.Client) -> float | None:
    resp = client.head(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    header = resp.headers.get("last-modified")
    if header is None:
        return None
    return parsedate_to_datetime(header).timestamp()


def _is_current(url: str, dest: Path, client: httpx.Client, mutable: bool) -> bool:
    """Whether the local copy can be reused.

    Immutable inputs (ACS vintages, gazetteer) are current once present. Mutable inputs
    (Zillow republishes the same URL every month) are current only if the server's
    Last-Modified matches the mtime stamped on the local file at download time.
    """
    if not dest.exists():
        return False
    if not mutable:
        return True
    remote = _remote_last_modified(url, client)
    return remote is not None and abs(dest.stat().st_mtime - remote) < 1


def _download(url: str, dest: Path, client: httpx.Client, mutable: bool = False) -> None:
    if _is_current(url, dest, client, mutable):
        log.info("cached %s", dest.name)
        return
    log.info("downloading %s", url)
    with client.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
        header = resp.headers.get("last-modified")
    if header is not None:
        stamp = parsedate_to_datetime(header).timestamp()
        os.utime(tmp, (stamp, stamp))
    tmp.replace(dest)


def download_zillow(client: httpx.Client) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    _download(config.ZHVI_URL, config.ZHVI_FILE, client, mutable=True)
    _download(config.ZORI_URL, config.ZORI_FILE, client, mutable=True)
    _download(config.ZHVF_URL, config.ZHVF_FILE, client, mutable=True)
    for name, url in config.ZILLOW_TEMPO_URLS.items():
        _download(url, config.ZILLOW_TEMPO_FILES[name], client, mutable=True)


def download_fred(client: httpx.Client) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, series in config.FRED_SERIES.items():
        url = config.FRED_URL.format(series=series)
        _download(url, config.FRED_FILES[name], client, mutable=True)


def download_gazetteer(client: httpx.Client) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.GAZETTEER_FILE.exists():
        log.info("cached %s", config.GAZETTEER_FILE.name)
        return
    resp = client.get(config.GAZETTEER_URL, follow_redirects=True, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        config.GAZETTEER_FILE.write_bytes(zf.read(name))


def _available_variables(client: httpx.Client, year: int) -> set[str]:
    resp = client.get(CENSUS_API.format(year=year) + "/variables.json", timeout=120)
    resp.raise_for_status()
    return set(resp.json()["variables"])


def _acs_query(client: httpx.Client, year: int, codes: list[str]) -> pd.DataFrame:
    params = {
        "get": ",".join(codes),
        "for": "zip code tabulation area:*",
        "key": census_api_key(),
    }
    resp = client.get(CENSUS_API.format(year=year), params=params, timeout=300)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={"zip code tabulation area": "zcta"})
    keep = ["zcta", *codes]
    return df[keep]


def download_acs_year(client: httpx.Client, year: int) -> Path | None:
    """One ACS 5-year vintage for every ZCTA, normalized to the column names in config."""
    config.ACS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.ACS_DIR / f"acs5_{year}.parquet"
    if dest.exists():
        log.info("cached %s", dest.name)
        return dest
    try:
        available = _available_variables(client, year)
    except httpx.HTTPStatusError as exc:
        log.warning("ACS %d not available (%s)", year, exc.response.status_code)
        return None

    edu_table = next((t for t in config.ACS_EDU_TABLES if f"{t}_001E" in available), None)
    if edu_table is None:
        raise RuntimeError(f"no educational attainment table found for ACS {year}")
    edu_codes = [c for cols in config.ACS_EDU_TABLES[edu_table].values() for c in cols]

    base_codes = list(config.ACS_VARIABLES.values())
    missing = [c for c in base_codes if c not in available]
    if missing:
        raise RuntimeError(f"ACS {year} lacks variables {missing}")

    # The API caps a request at 50 variables; the base list plus B15002 fits in two calls.
    frames = [
        _acs_query(client, year, base_codes),
        _acs_query(client, year, edu_codes),
    ]
    df = frames[0].merge(frames[1], on="zcta", how="outer")

    numeric = df.drop(columns=["zcta"]).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.mask(numeric <= ACS_MISSING_SENTINEL)
    out = pd.DataFrame({"zcta": df["zcta"].str.zfill(5), "acs_year": year})
    for name, code in config.ACS_VARIABLES.items():
        out[name] = numeric[code]
    for name, cols in config.ACS_EDU_TABLES[edu_table].items():
        out[name] = numeric[cols].sum(axis=1, min_count=1)
    out.to_parquet(dest, index=False)
    log.info("ACS %d: %d ZCTAs", year, len(out))
    return dest


def download_all() -> None:
    with httpx.Client() as client:
        download_zillow(client)
        download_fred(client)
        download_gazetteer(client)
        if not census_api_key():
            raise SystemExit("CENSUS_API_KEY is not set; get a free key at api.census.gov")
        for year in config.ACS_YEARS:
            download_acs_year(client, year)
