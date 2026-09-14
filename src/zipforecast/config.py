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
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/"
    "2023_Gaz_zcta_national.zip"
)

ZHVI_FILE = RAW_DIR / "zhvi_zip.csv"
ZORI_FILE = RAW_DIR / "zori_zip.csv"
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

HORIZONS = (5, 10)
FIRST_TEST_ORIGIN_YEAR = 2010
NEIGHBOR_COUNT = 10
RECENCY_HALF_LIFE_YEARS = 4
MIN_ZIPS_PER_METRO = 10
