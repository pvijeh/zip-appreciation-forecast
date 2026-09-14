# zip-appreciation-forecast

Ranks the 831 ZIP codes in the New York metro by how much their home values are expected to
gain, relative to the metro, over the next 1, 2, 3, 5 and 10 years. Trained on 21,556 ZIPs
across all US metros using Zillow home values and rents plus Census ACS demographics. All data
is public; the only credential is a free Census API key.

## What the walk-forward test found

At one year the models have measurable skill. At five and ten years they rank NYC ZIPs
backwards. [output/REPORT.md](output/REPORT.md) has every table.

| Horizon | Test origins | Gradient boosting, all US metros | Best one-line rule | Model beat the rule in |
|---|---|---|---|---|
| 1 year | 16 (2010 to 2025) | +0.41 | last year's growth continues, +0.39 | 8 of 16 origins |
| 2 years | 15 | +0.38 | last year's growth continues, +0.40 | 6 of 15 |
| 3 years | 14 | +0.28 | last year's growth continues, +0.36 | 4 of 14 |
| 5 years | 12 | -0.12 | cheaper than metro median catches up, +0.29 | 2 of 12 |
| 10 years | 7 | -0.26 | cheaper than metro median catches up, +0.54 | 0 of 7 |

Numbers are the Spearman rank correlation between the predicted and the realized ranking of NYC
ZIPs, averaged over test origins the model never saw. The 1-year model was positive in 14 of 16
origins; its worst origin was 2010 at -0.06. But a spreadsheet column sorted by last year's
growth relative to the metro gets the same ranking, and the model beat that column in only half
the origins.

The long horizons fail for a reason the "one feature at a time" table in the report shows.
Almost every feature is a version of one question, is this ZIP cheap or expensive for its metro,
and which answer wins flips every 6 to 8 years:

| Origins | Who beat the NYC metro over the next 5 years |
|---|---|
| 2000 to 2003 | cheap ZIPs (Spearman of price level with growth: -0.37 to -0.55) |
| 2005 to 2012 | expensive ZIPs (+0.13 to +0.56) |
| 2013 to 2020 | cheap ZIPs again (-0.10 to -0.74) |
| 2021 | mixed; richer ZIPs with high rent yield and 1-year momentum (+0.31, +0.48, +0.59) |

A model can only learn from origins whose outcome is already known. At a 5-year horizon it is
fit on the previous phase of the cycle and applied to the next one. Weighting recent origins more
(half-life 4 years) or training on NYC ZIPs only does not fix this; both are in the report.

## Which features carried signal

Spearman of each feature alone with realized growth relative to the metro, averaged over
2010 to the last realized origin; "stable" is the share of origins where the sign held. All US
metro ZIPs. The full per-origin tables, and the NYC-only versions, are in the report.

| Feature | 1 year | 3 years | 5 years | Stable at 1 year |
|---|---|---|---|---|
| 1-year momentum, higher wins | +0.21 | +0.19 | +0.14 | 100% |
| rent yield (ZORI/ZHVI), higher wins | +0.18 | +0.26 | +0.32 | 82% |
| bachelor's share, higher loses | -0.16 | -0.24 | -0.30 | 69% |
| median income, higher loses | -0.13 | -0.20 | -0.24 | 77% |
| price vs metro median, cheaper wins | -0.12 | -0.23 | -0.33 | 62% |
| 5-year momentum | +0.12 | +0.06 | -0.03 | 81% |
| price gap vs 10 nearest ZIPs, cheaper wins | -0.11 | -0.20 | -0.27 | 69% |
| price to income, higher loses | -0.10 | -0.20 | -0.28 | 85% |
| unemployment rate, higher wins | +0.10 | +0.17 | +0.21 | 69% |

Momentum is the only feature whose sign never flipped. The cheapness features get stronger with
horizon and were all one sign from 2013 to 2020, which is why the 5- and 10-year rules look
good over that window and why a model trained on it fails when the phase turns. Rent yield only
exists from 2015 (ZORI), so its history is one phase.

## Comparison with Zillow's published forecast

Zillow publishes a 12-month home value forecast by ZIP (ZHVF). For the 830 NYC-metro ZIPs both
cover, the Spearman between our 1-year score and Zillow's forecast is +0.47. Zillow expects
+2.8% for our top 25 versus +2.4% for the metro. This is a check that two methods agree, not a
test of either; Zillow's forecast is absolute growth and ours is rank within the metro.

## The rankings

`output/nyc_forecast_{1,2,3,5,10}y.json` lists every NYC-metro ZIP, ranked, with the score, the
walk-forward skill of every model at that horizon, Zillow's 1-year forecast for the same ZIP,
and what ZIPs at that score realized in past origins. The score is the model's expected
percentile of growth within the metro, 0 to 100, not a growth number. The 1-year list is a
momentum list: Ocean, Bergen, Monmouth and Union County NJ suburbs that beat the metro in
2025 to 2026. The 3-year list is mostly the cheapest ZIPs (Sussex NJ, Pike PA); read it with the
3-year row of the first table in mind.

## Run it

```bash
uv sync
export CENSUS_API_KEY=...   # free key from https://api.census.gov/data/key_signup.html
uv run zipforecast all      # download, build panel, evaluate, rank
```

Or step by step: `zipforecast download`, `zipforecast panel`, `zipforecast evaluate`,
`zipforecast rank`. The download is about 160 MB; the evaluation refits the models for every
fold and takes roughly an hour. Outputs go to `output/`:

| File | Contents |
|---|---|
| `REPORT.md` | all tables, plus the top 25 NYC ZIPs per horizon |
| `evaluation_summary.csv` | mean and worst-year Spearman and top-fifth minus bottom-fifth spread, per model and horizon |
| `evaluation_{h}y_by_year.csv` | the same per test origin |
| `factor_history_{h}y.csv` | Spearman of each single feature with realized relative growth, per origin, all US and NYC |
| `feature_importance_{h}y.csv` | permutation importance of the national gradient boosting model on the latest test origin |
| `nyc_ranking_{h}y.csv` | every NYC-metro ZIP with score, rank and the features behind it |
| `nyc_forecast_{h}y.json` | the same ranking with the skill numbers and Zillow's forecast, for other programs |

## How it is built

**Data.**

| Source | What | Coverage |
|---|---|---|
| Zillow ZHVI | home values, monthly, by ZIP | 2000 on |
| Zillow ZORI | rents, monthly, by ZIP | 2015 on |
| Zillow ZHVF | Zillow's own 12-month forecast, by ZIP | latest month only |
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
on origins up to 2011, whose outcomes were known by 2016. Test origins start in 2010 and end at
the last origin whose outcome is known: 2025 for 1 year, 2021 for 5 years, 2016 for 10 years.
Scored on all US metro ZIPs and on NYC ZIPs separately.

**Models.** `HistGradientBoostingRegressor` and ridge regression on all US metros, gradient
boosting with recent origins weighted more, gradient boosting on NYC only, and four rules:
1-year momentum continues, 5-year momentum continues, cheaper than metro median catches up,
cheaper than the 10 nearest ZIPs catches up.

## Layout

```
src/zipforecast/
  config.py    paths, URLs, ACS variable list, horizons, NYC metro name
  download.py  Zillow CSVs, ACS by year (one Parquet each), gazetteer
  panel.py     panel construction, features, targets, within-metro percentiles
  model.py     folds, models, baselines, metrics, factor history, ZHVF, final ranking
  report.py    output/REPORT.md, CSVs and JSON
  cli.py       zipforecast download | panel | evaluate | rank | all
tests/         unit tests for the panel helpers, metrics, ZHVF parsing and JSON output
scripts/       prose-check.mjs, run with `node scripts/prose-check.mjs` before editing this README
data/          raw and processed data, not committed
output/        committed so the latest run is readable without running anything
```

## Things this does not do yet

- No NYC-only data (DOB permits, subway distance, rezonings, flood zones). Nothing in the
  national test suggests more features would beat 1-year momentum at short horizons.
- No attempt to predict which phase of the cycle comes next. A macro feature (mortgage rate
  change, national price-to-rent, the trailing sign of the cheap-catches-up factor) is the
  obvious next experiment for the 5-year horizon.
- Overlapping windows. Sixteen 1-year folds are close to sixteen independent tests; twelve
  5-year folds are closer to two or three.
- Rent appreciation is not a target; ZORI only starts in 2015.
- Zillow revises history. The panel uses today's vintage, so old origins look cleaner than they
  did at the time.
