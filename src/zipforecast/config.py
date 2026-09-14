from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ACS_DIR = RAW_DIR / "acs"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "output"

ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
ZORI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv"
)
# Zillow's own published forecast: % change in ZHVI over the next 1, 3 and 12 months by ZIP.
ZHVF_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvf_growth/"
    "Zip_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
# Zillow market-tempo series by ZIP, monthly from 2018-03: for-sale inventory, mean days to
# pending, share of listings with a price cut, new listings.
ZILLOW_TEMPO_URLS = {
    "inventory": (
        "https://files.zillowstatic.com/research/public_csvs/invt_fs/"
        "Zip_invt_fs_uc_sfrcondo_sm_month.csv"
    ),
    "days_pending": (
        "https://files.zillowstatic.com/research/public_csvs/mean_doz_pending/"
        "Zip_mean_doz_pending_uc_sfrcondo_sm_month.csv"
    ),
    "price_cut_share": (
        "https://files.zillowstatic.com/research/public_csvs/perc_listings_price_cut/"
        "Zip_perc_listings_price_cut_uc_sfrcondo_sm_month.csv"
    ),
    "new_listings": (
        "https://files.zillowstatic.com/research/public_csvs/new_listings/"
        "Zip_new_listings_uc_sfrcondo_sm_month.csv"
    ),
}
# FRED series for the regime features: 30-year mortgage rate (weekly) and rent CPI (monthly).
FRED_SERIES = {"mortgage_rate": "MORTGAGE30US", "rent_cpi": "CUSR0000SEHA"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/"
    "2023_Gaz_zcta_national.zip"
)

ZHVI_FILE = RAW_DIR / "zhvi_zip.csv"
ZORI_FILE = RAW_DIR / "zori_zip.csv"
ZHVF_FILE = RAW_DIR / "zhvf_zip.csv"
ZILLOW_TEMPO_FILES = {name: RAW_DIR / f"zillow_{name}_zip.csv" for name in ZILLOW_TEMPO_URLS}
FRED_FILES = {name: RAW_DIR / f"fred_{name}.csv" for name in FRED_SERIES}
GAZETTEER_FILE = RAW_DIR / "zcta_gazetteer.txt"
PANEL_FILE = PROCESSED_DIR / "panel.parquet"

NYC_METRO = "New York-Newark-Jersey City, NY-NJ-PA"

# ACS 5-year vintages with ZCTA geography. Vintage Y is published in December of Y+1.
ACS_YEARS = range(2011, 2025)
ACS_PUBLICATION_LAG_YEARS = 2  # at an origin in year T, vintage T-2 is the latest safe one

# Column name -> ACS 5-year variable code.
ACS_VARIABLES = {
    "pop": "B01003_001E",
    "median_age": "B01002_001E",
    "male_25_29": "B01001_011E",
    "male_30_34": "B01001_012E",
    "female_25_29": "B01001_035E",
    "female_30_34": "B01001_036E",
    "median_income": "B19013_001E",
    "acs_median_value": "B25077_001E",
    "acs_median_rent": "B25064_001E",
    "housing_units": "B25001_001E",
    "vacant_units": "B25002_003E",
    "occupied_units": "B25003_001E",
    "owner_occupied": "B25003_002E",
    "mobility_total": "B07001_001E",
    "moved_other_county": "B07001_049E",
    "moved_other_state": "B07001_065E",
    "moved_abroad": "B07001_081E",
    "labor_force": "B23025_003E",
    "unemployed": "B23025_005E",
    "commuters": "B08301_001E",
    "transit_commuters": "B08301_010E",
    "median_year_built": "B25035_001E",
}

# Educational attainment, population 25+. B15003 starts with the 2012 vintage; 2011 only has
# B15002 (split by sex). Both reduce to a total and a bachelor's-or-higher count.
ACS_EDU_TABLES = {
    "B15003": {
        "edu_total": ["B15003_001E"],
        "edu_bachelors_plus": ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"],
    },
    "B15002": {
        "edu_total": ["B15002_001E"],
        "edu_bachelors_plus": [
            "B15002_015E",
            "B15002_016E",
            "B15002_017E",
            "B15002_018E",
            "B15002_032E",
            "B15002_033E",
            "B15002_034E",
            "B15002_035E",
        ],
    },
}

HORIZONS = (1, 2, 3, 5, 10)
FIRST_TEST_ORIGIN_YEAR = 2010
NEIGHBOR_COUNT = 10
RECENCY_HALF_LIFE_YEARS = 4
MIN_ZIPS_PER_METRO = 10
BETA_WINDOW_YEARS = 10
BETA_MIN_YEARS = 6
PRICE_TO_RENT_MEAN_YEARS = 10
BOOTSTRAP_DRAWS = 2000
