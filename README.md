# zip-appreciation-forecast

Ranks the 831 ZIP codes in the New York metro by how much their home values are expected to
gain, relative to the metro, over the next 5 and 10 years. Trained on 21,556 ZIPs across all US
metros using Zillow home values and rents plus Census ACS demographics. All data is public; the
only credential is a free Census API key.

## What the walk-forward test found

The learned models rank NYC ZIPs backwards more often than forwards. Averaged over 12 test
origins (2010 to 2021, 5-year horizon), the gradient boosting model trained on all US metros has
a Spearman rank correlation of -0.12 with what happened. The one-line rule "the cheapest ZIPs in
the metro catch up" has +0.29 over the same 12 origins ([output/REPORT.md](output/REPORT.md)
has every table).

The reason is in the "one feature at a time" table of that report. Almost every feature is a
version of one question, is this ZIP cheap or expensive for its metro, and which answer wins
flips every 6 to 8 years:

| Origins | Who beat the NYC metro over the next 5 years |
|---|---|
| 2000 to 2003 | cheap ZIPs (Spearman of price level with growth: -0.37 to -0.55) |
| 2005 to 2012 | expensive ZIPs (+0.13 to +0.56) |
| 2013 to 2020 | cheap ZIPs again (-0.10 to -0.74) |
| 2021 | mixed; richer ZIPs with high rent yield and 1-year momentum (+0.31, +0.48, +0.59) |

A model can only learn from origins whose 5-year outcome is already known. So at any point it is
fit on the previous phase of the cycle and applied to the next one. Weighting recent origins more
(half-life 4 years) or training on NYC ZIPs only does not fix this; both are in the table.

Since 2015 three features have kept one sign in NYC. ZIPs that are cheap relative to local
income win, ZIPs with high rent yield win, and ZIPs that gained the most in the past year win.
The ranking in `output/nyc_ranking_5y.csv` is what the models produce today. It is close to a
list of the cheapest ZIPs in the metro, and it has no demonstrated skill. Of the top 25, eight
are in Pike County PA, eight in Sussex County NJ, three in Ocean County NJ; one is in the Bronx.

## Run it

```bash
uv sync
export CENSUS_API_KEY=...   # free key from https://api.census.gov/data/key_signup.html
uv run zipforecast all      # download, build panel, evaluate, rank
```

Or step by step: `zipforecast download`, `zipforecast panel`, `zipforecast evaluate`,
`zipforecast rank`. Outputs go to `output/`:

| File | Contents |
|---|---|
| `REPORT.md` | all tables below, plus the top 25 NYC ZIPs per horizon |
| `evaluation_summary.csv` | mean and worst-year Spearman and top-fifth minus bottom-fifth spread, per model and horizon |
| `evaluation_{5,10}y_by_year.csv` | the same per test origin |
| `factor_history_{5,10}y.csv` | Spearman of each single feature with realized relative growth, per origin, all US and NYC |
| `feature_importance_{5,10}y.csv` | permutation importance of the national gradient boosting model on the latest test origin |
| `nyc_ranking_{5,10}y.csv` | every NYC-metro ZIP with score, rank and the features behind it |

## How it is built

**Data.**

| Source | What | Coverage |
|---|---|---|
| Zillow ZHVI | home values, monthly, by ZIP | 2000 on |
| Zillow ZORI | rents, monthly, by ZIP | 2015 on |
| Census ACS 5-year | population, income, rent, housing units, vacancy, tenure, education, age of housing, migration, unemployment, transit commuting; by ZCTA | vintages 2011 to 2024 |
| Census ZCTA gazetteer | centroid of each ZCTA, for nearest-neighbor features | 2023 |

ACS is used with a 2-year publication lag, so the 2013 origin sees the 2011 vintage.

**Panel.** One row per ZIP per July origin, 2000 to 2026, 480,889 rows. Features known at the
origin: 1-, 3- and 5-year price momentum, 3-year rent momentum, rent yield, price relative to
the metro median, price and momentum gaps to the 10 nearest ZIPs, and the ACS ratios above.

**Target.** The ZIP's log price change over the horizon minus the average for its metro. This
removes the national and metro cycle; the model only has to rank ZIPs within a metro.

**Features and target as within-metro percentiles.** A 2005 price level and a 2026 price level
are only compared to their own metro in the same year, so the model never sees a raw number it
was not trained on.

**Walk-forward test.** For a test origin such as 2016 at a 5-year horizon, the model is fit only
on origins up to 2011, whose outcomes were known by 2016. Test origins run 2010 to 2021 (5-year)
and 2010 to 2016 (10-year). Scored on all US metro ZIPs and on NYC ZIPs separately.

**Models.** `HistGradientBoostingRegressor` and ridge regression on all US metros, gradient
boosting with recent origins weighted more, gradient boosting on NYC only, and three rules:
5-year momentum continues, cheaper than metro median catches up, cheaper than the 10 nearest
ZIPs catches up.

## Layout

```
src/zipforecast/
  config.py    paths, URLs, ACS variable list, horizons, NYC metro name
  download.py  Zillow CSVs, ACS by year (one Parquet each), gazetteer
  panel.py     panel construction, features, targets, within-metro percentiles
  model.py     folds, models, baselines, metrics, factor history, final ranking
  report.py    output/REPORT.md and CSVs
  cli.py       zipforecast download | panel | evaluate | rank | all
tests/         unit tests for the panel helpers and metrics
scripts/       prose-check.mjs, run with `node scripts/prose-check.mjs` before editing this README
data/          raw and processed data, not committed
output/        committed so the latest run is readable without running anything
```

## Things this does not do yet

- No NYC-only data (DOB permits, subway distance, rezonings, flood zones). The national test
  suggests adding features will not help until the cycle-phase problem is handled.
- No attempt to predict which phase comes next. A macro feature (mortgage rate change, national
  price-to-rent, the trailing sign of the cheap-catches-up factor) is the obvious next experiment.
- Rent appreciation is not a target; ZORI only starts in 2015, too short for a 5-year horizon.
- Zillow revises history. The panel uses today's vintage, so old origins look cleaner than they
  did at the time.
