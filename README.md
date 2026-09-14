# zip-appreciation-forecast

Ranks the 831 ZIP codes in the New York metro by how much their home values are expected to
gain, relative to the metro, over the next 1, 2, 3, 5 and 10 years. Trained on 21,556 ZIPs
across all US metros using Zillow home values, rents and listing counts, Census ACS demographics
and two FRED series. All data is public; the only credential is a free Census API key.
[RESULTS.md](RESULTS.md) is the one-page version: the score, and the 15 best and 15 worst
ranked ZIPs for the next 12 months.

## What the walk-forward test found

At one year the model beats the best one-line rule in 11 of 16 test years, and the margin is
small: +0.44 against +0.39. At five and ten years it ranks NYC ZIPs backwards.
[output/REPORT.md](output/REPORT.md) has every table.

| Horizon | Test origins | Gradient boosting, all US metros (90% interval) | Best one-line rule | Model beat the rule in |
|---|---|---|---|---|
| 1 year | 16 (2010 to 2025) | +0.44 (+0.36 to +0.51) | last year's growth continues, +0.39 | 11 of 16 origins |
| 2 years | 15 | +0.40 (+0.27 to +0.51) | last year's growth continues, +0.40 | 8 of 15 |
| 3 years | 14 | +0.36 (+0.27 to +0.44) | last year's growth continues, +0.36 | 6 of 14 |
| 5 years | 12 | -0.00 (-0.08 to +0.06) | cheaper than metro median catches up, +0.29 | 4 of 12 |
| 10 years | 7 | -0.23 | cheaper than metro median catches up, +0.54 | 0 of 7 |

Numbers are the Spearman rank correlation between the predicted and the realized ranking of NYC
ZIPs, averaged over test origins the model never saw. The interval resamples test origins in
blocks as long as the horizon; seven 10-year origins are too few for one. At one year the gap
between the model and the momentum rule is +0.05 with a 90% interval of +0.01 to +0.09, so the
model's edge is probably real and certainly small. In money terms, the top fifth of the 1-year
ranking beat the bottom fifth by 5.6 log points against the metro over the next year, on
average; a spreadsheet column sorted by last year's relative growth gets 5.0.

## Why the long horizons fail

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
(half-life 4 years), training on NYC ZIPs only, and the regime features below do not fix this;
all three are in the report.

## Where the one-year edge comes from

Zillow's for-sale listing files. Refitting the model with one group of features removed and
scoring the same held-out NYC origins:

| Features removed | Mean NYC Spearman | Change | Origins where the group helped |
|---|---|---|---|
| none | +0.44 | | |
| Zillow listing tempo: inventory, new listings, days to pending, share of listings with a price cut | +0.42 | -0.02 | 7 of the 7 origins where the files exist (2019 to 2025) |
| county-relative price and momentum | +0.44 | +0.00 | 9 of 16 |
| 10-year beta to the metro | +0.44 | -0.00 | 9 of 16 |
| regime: mortgage rate, national price-to-rent, which way the cheap-catches-up factor went last year | +0.44 | -0.00 | 9 of 16 |
| all four groups | +0.41 | -0.03 | 9 of 16 |

The listing files start in 2018, so they are empty at the first nine test origins and the mean
understates them. Over 2019 to 2025 alone the model scores +0.45 with them and +0.41 without.
The share of listings with a price cut is the single strongest of these features: Spearman -0.21
with next year's relative growth, same sign in all 8 origins it covers. The other three groups
help in some years and hurt in others and average to nothing.

Two variants that did not help. One neutralizes the target by county instead of metro, so the
model ranks ZIPs against their borough or county. It scores +0.32 on the metro-wide ranking and
+0.34 within county; the metro-relative model scores +0.44 and +0.32. The county version gives up
the between-county spread, which the model does predict, for a small within-county gain.
The other, the regime features, was meant to tell the model which phase of the cycle it is in.
With them the 5-year model went from -0.12 to -0.00, still no skill.

## Does a ZIP have a beta to its metro?

Yes, and it is visible one year ahead. Beta here is the slope of a ZIP's annual growth on its
metro's over the trailing 10 years, computed at each origin from data known by then. Splitting
the next year by whether the ZIP's metro rose or fell:

| Next year the metro | Spearman of beta with relative growth, all US | Sign held in | NYC |
|---|---|---|---|
| rose | +0.14 | 17 of 20 origins | +0.25 over 14 origins |
| fell | -0.19 | 18 of 19 origins | -0.12 over 6 origins |

High-beta ZIPs beat their metro in up years and trail it in down years, as a stock beta would
predict. Two limits. Beta is not the whole story: in 2013 to 2021 cheap ZIPs caught up without
giving it back, which is drift, not beta. And knowing the pattern does not help unless you know
which way the metro goes next; a rule that multiplies beta by the metro's past-year growth
scores +0.29 at one year, below plain momentum. Per-origin table in the report and in
`output/beta_history_1y.csv`.

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
| 10-year beta to metro, higher wins | +0.11 | +0.18 | +0.21 | 69% |
| share of listings with a price cut, higher loses (2018 on) | -0.21 | -0.18 | -0.16 | 100% |
| for-sale inventory, 1-year change, higher loses (2019 on) | -0.13 | -0.11 | -0.14 | 100% |

Momentum and the two listing features are the only ones whose sign never flipped, and the listing
features have 7 or 8 origins of history. The cheapness features get stronger with horizon and
were all one sign from 2013 to 2020, which is why the 5- and 10-year rules look good over that
window and why a model trained on it fails when the phase turns. Rent yield only exists from
2015 (ZORI), so its history is one phase.

## Comparison with Zillow's published forecast

Zillow publishes a 12-month home value forecast by ZIP (ZHVF). For the 830 NYC-metro ZIPs both
cover, the Spearman between our 1-year score and Zillow's forecast is +0.59. Zillow expects
+3.0% for our top 25 versus +2.4% for the metro. This is a check that two methods agree, not a
test of either; Zillow's forecast is absolute growth and ours is rank within the metro.

## The rankings

`output/nyc_forecast_{1,2,3,5,10}y.json` lists every NYC-metro ZIP, ranked, with the score, the
walk-forward skill of every model at that horizon, Zillow's 1-year forecast for the same ZIP,
and what ZIPs at that score realized in past origins. The score is the model's expected
percentile of growth within the metro, 0 to 100, not a growth number. The 1-year list is
expensive commuter suburbs (Pelham, Summit, Westfield, Rumson, Chatham) where the market is tight.
The top 25 averaged 8.5% growth last year against 4.5% for the median NYC ZIP; inventory is down
19% against up 2%; 10% of listings took a price cut against 15%. The 3-year list is cheaper
Suffolk and Passaic County ZIPs (Shirley, Mastic, Wanaque); at that horizon the model beat the
momentum rule in 6 of 14 origins, so the list is no better than sorting by last year's growth.

## Run it

```bash
uv sync
export CENSUS_API_KEY=...   # free key from https://api.census.gov/data/key_signup.html
uv run zipforecast all      # download, build panel, evaluate, rank
```

Or step by step: `zipforecast download`, `zipforecast panel`, `zipforecast evaluate`,
`zipforecast rank`. The download is about 250 MB; the evaluation refits the models for every
fold and takes one to two hours. Outputs go to `output/`:

| File | Contents |
|---|---|
| `REPORT.md` | all tables, plus the top 25 NYC ZIPs per horizon |
| `evaluation_summary.csv` | mean and worst-year Spearman and top-fifth minus bottom-fifth spread, per model and horizon |
| `evaluation_{h}y_by_year.csv` | the same per test origin |
| `factor_history_{h}y.csv` | Spearman of each single feature with realized relative growth, per origin, all US and NYC |
| `feature_importance_{h}y.csv` | permutation importance of the national gradient boosting model on the latest test origin |
| `ablation_1y.csv` | 1-year NYC Spearman per test origin with each feature group removed |
| `beta_history_1y.csv` | Spearman of trailing beta with next-year relative growth, split by whether the metro rose or fell |
| `nyc_ranking_{h}y.csv` | every NYC-metro ZIP with score, rank and the features behind it |
| `nyc_forecast_{h}y.json` | the same ranking with the skill numbers and Zillow's forecast, for other programs |

## How it is built

**Data.**

| Source | What | Coverage |
|---|---|---|
| Zillow ZHVI | home values, monthly, by ZIP | 2000 on |
| Zillow ZORI | rents, monthly, by ZIP | 2015 on |
| Zillow ZHVF | Zillow's own 12-month forecast, by ZIP | latest month only |
| Zillow listings | for-sale inventory, new listings, mean days to pending, share of listings with a price cut; monthly, by ZIP | 2018 on |
| FRED | 30-year mortgage rate (MORTGAGE30US), rent CPI (CUSR0000SEHA) | 2000 on |
| Census ACS 5-year | population, income, rent, housing units, vacancy, tenure, education, age of housing, migration, unemployment, transit commuting; by ZCTA | vintages 2011 to 2024 |
| Census ZCTA gazetteer | centroid of each ZCTA, for nearest-neighbor features | 2023 |

ACS is used with a 2-year publication lag, so the 2013 origin sees the 2011 vintage.

**Panel.** One row per ZIP per July origin, 2000 to 2026, 480,889 rows. Every feature is known
at the origin.

| Group | Features |
|---|---|
| price and rent | 1-, 3- and 5-year price momentum, 3-year rent momentum, rent yield |
| relative price | price against the metro median, against the county, and against the 10 nearest ZIPs; momentum gap to the 10 nearest ZIPs |
| ACS | the ratios in the data table above, 2-year lag |
| beta | slope of the ZIP's annual growth on its metro's, trailing 10 years |
| listing tempo | inventory, new listings, days to pending, price-cut share, their 1-year changes, inventory per 1,000 housing units |
| regime, same for every ZIP at an origin | mortgage rate and its 1-year change, national 1-year momentum, national price-to-rent against its 10-year mean |
| regime, per metro | whether cheap ZIPs beat the metro over the past year, and whether the past year's winners did (sign only) |

**Target.** The ZIP's log price change over the horizon minus the average for its metro. This
removes the national and metro cycle; the model only has to rank ZIPs within a metro, which is
why it can train on every US metro and score NYC. NYC alone is 831 ZIPs sharing one cycle, about
two of them since 2000; too few to fit on.

**Features and target as within-metro percentiles.** A 2005 price level and a 2026 price level
are only compared to their own metro in the same year, so the model never sees a raw number it
was not trained on.

**Walk-forward test.** For a test origin such as 2016 at a 5-year horizon, the model is fit only
on origins up to 2011, whose outcomes were known by 2016. Test origins start in 2010 and end at
the last origin whose outcome is known: 2025 for 1 year, 2021 for 5 years, 2016 for 10 years.
Scored on all US metro ZIPs and on NYC ZIPs separately.

**Models.** `HistGradientBoostingRegressor` and ridge regression on all US metros; gradient
boosting with recent origins weighted more, on NYC only, and with the target neutralized by
county. Five one-line rules: 1-year momentum continues, 5-year momentum continues, cheaper than
metro median catches up, cheaper than the 10 nearest ZIPs catches up, beta times the metro's
past-year growth.

## Layout

```
src/zipforecast/
  config.py    paths, URLs, ACS variable list, FRED series, horizons, NYC metro name
  download.py  Zillow CSVs, FRED CSVs, ACS by year (one Parquet each), gazetteer
  panel.py     panel construction, features, beta, regime columns, targets, percentiles
  model.py     folds, models, baselines, metrics, bootstrap, factor and beta history, ablation, ranking
  report.py    output/REPORT.md, CSVs and JSON
  cli.py       zipforecast download | panel | evaluate | rank | all
tests/         unit tests for the panel helpers, metrics, ZHVF parsing and JSON output
scripts/       prose-check.mjs, run with `node scripts/prose-check.mjs` before editing this README
data/          raw and processed data, not committed
output/        committed so the latest run is readable without running anything
```

## Things this does not do yet

- No NYC-only data (DOB permits, subway distance, rezonings, flood zones). The one group of
  features that helped at one year was market tempo, so listing-level data is the better bet
  than more demographics.
- No cycle timing. The regime features were the attempt and did not work at 5 years.
- Overlapping windows. Sixteen 1-year folds are close to sixteen independent tests; twelve
  5-year folds are closer to two or three.
- Rent appreciation is not a target; ZORI only starts in 2015.
- Zillow revises history. The panel uses today's vintage, so old origins look cleaner than they
  did at the time.
