# Results: a small one-year edge, nothing at five to ten years

The model ranks NYC-metro ZIPs by expected price growth relative to the metro over the next year, trained on every US metro. Over 16 test years it never saw, the Spearman rank correlation between its ranking and what the ZIPs then did was +0.44, worst year -0.03. The one-line rule "last year's growth relative to the metro continues" scores +0.39. The model beat that rule in 11 of 16 years, by +0.05 on average, 90% interval +0.01 to +0.09: real, but small. In price terms, the model's top fifth of ZIPs beat its bottom fifth by 5.6 percentage points of growth against the metro over the next year; the rule's top fifth beat its bottom fifth by 5.0.

At five and ten years the model has no skill, and neither did anything else we tried. Which end of the cheap-to-expensive axis wins flips every 6 to 8 years, and the data holds two such cycles. The 5- and 10-year files in `output/` are lists of the cheapest ZIPs, not forecasts.

Zillow data through 2026-07-31; 831 NYC-metro ZIPs ranked. Full tables are in [output/REPORT.md](output/REPORT.md), method and caveats in the [README](README.md). Regenerate with `zipforecast evaluate` or `zipforecast rank`.

## The 15 best and 15 worst ranked NYC-metro ZIPs for the next 12 months

Score is the model's expected percentile of price growth within the NYC metro, 0 to 100. The next column is the median of what NYC ZIPs in the same 5-point score band did over the following year in past origins, relative to the metro: a historical analogue, not a forecast of this year's metro. Zillow's 12-month forecast is an absolute number, so the two columns are not comparable.

The top 15 are priced 73% above the metro median and grew 5 points faster than the metro last year. The bottom 15 are priced 17% below the metro median and grew 5 points slower than the metro last year. Read the list for what it is: mostly last year's relative growth, adjusted by how fast listings are moving.

### Top 15

| Rank | ZIP | Place | County | Home value | vs metro median | Last year vs metro | Score | Past ZIPs in this score band, next year vs metro | Zillow 12-month forecast |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10803 | Pelham, NY | Westchester County | $1,458k | +96% | +7% | 73 | +1.8% | +3.7% |
| 2 | 07711 | Allenhurst, NJ | Monmouth County | $2,552k | +243% | +7% | 72 | +1.8% | +3.6% |
| 3 | 07901 | Summit, NJ | Union County | $1,421k | +91% | +7% | 72 | +1.8% | +3.3% |
| 4 | 07090 | Westfield, NJ | Union County | $1,329k | +79% | +4% | 72 | +1.8% | +2.9% |
| 5 | 07760 | Rumson, NJ | Monmouth County | $1,893k | +155% | +5% | 71 | +1.8% | +3.5% |
| 6 | 07928 | Chatham, NJ | Morris County | $1,398k | +88% | +5% | 71 | +1.8% | +3.3% |
| 7 | 08735 | Lavallette, NJ | Ocean County | $1,223k | +65% | +6% | 70 | +1.8% | +3.0% |
| 8 | 07974 | New Providence, NJ | Union County | $1,022k | +38% | +3% | 70 | +1.3% | +2.9% |
| 9 | 08736 | Manasquan, NJ | Monmouth County | $1,234k | +66% | +6% | 70 | +1.3% | +3.3% |
| 10 | 07719 | Wall, NJ | Monmouth County | $838k | +13% | +1% | 70 | +1.3% | +2.7% |
| 11 | 10580 | Rye, NY | Westchester County | $2,360k | +218% | +5% | 70 | +1.3% | +3.8% |
| 12 | 07040 | Maplewood, NJ | Essex County | $977k | +31% | +1% | 69 | +1.3% | +2.7% |
| 13 | 07446 | Ramsey, NJ | Bergen County | $899k | +21% | +4% | 69 | +1.3% | +2.8% |
| 14 | 07456 | Ringwood, NJ | Passaic County | $618k | -17% | +2% | 69 | +1.3% | +2.6% |
| 15 | 11219 | New York, NY | Kings County | $1,286k | +73% | +3% | 68 | +1.3% | +2.9% |

### Bottom 15

| Rank | ZIP | Place | County | Home value | vs metro median | Last year vs metro | Score | Past ZIPs in this score band, next year vs metro | Zillow 12-month forecast |
|---|---|---|---|---|---|---|---|---|---|
| 817 | 10040 | New York, NY | New York County | $484k | -35% | -4% | 25 | -2.0% | +0.9% |
| 818 | 07017 | East Orange, NJ | Essex County | $490k | -34% | -5% | 25 | -2.5% | +2.3% |
| 819 | 07108 | Newark, NJ | Essex County | $504k | -32% | -5% | 25 | -2.5% | +2.0% |
| 820 | 07105 | Newark, NJ | Essex County | $558k | -25% | -8% | 24 | -2.5% | +1.5% |
| 821 | 10029 | New York, NY | New York County | $769k | +3% | -5% | 24 | -2.5% | +1.7% |
| 822 | 07029 | Harrison, NJ | Hudson County | $584k | -21% | -4% | 24 | -2.5% | +2.0% |
| 823 | 10030 | New York, NY | New York County | $727k | -2% | -7% | 24 | -2.5% | +1.3% |
| 824 | 10033 | New York, NY | New York County | $613k | -17% | -3% | 22 | -2.5% | +1.7% |
| 825 | 10035 | New York, NY | New York County | $709k | -5% | -6% | 22 | -2.5% | +1.1% |
| 826 | 07112 | Newark, NJ | Essex County | $496k | -33% | -3% | 22 | -2.5% | +2.0% |
| 827 | 11101 | New York, NY | Queens County | $1,153k | +55% | -4% | 21 | -2.5% | +1.1% |
| 828 | 07102 | Newark, NJ | Essex County | $405k | -46% | -12% | 21 | -2.5% | -0.3% |
| 829 | 10018 | New York, NY | New York County | $1,328k | +79% | -8% | 20 | -2.5% | +0.2% |
| 830 | 10280 | New York, NY | New York County | $822k | +11% | -3% | 17 | -3.0% | +2.1% |
| 831 | 10036 | New York, NY | New York County | $997k | +34% | -8% | 17 | -3.0% | +0.6% |

A rank correlation of +0.44 leaves room for a fair share of the top 15 to trail the metro next year and a fair share of the bottom 15 to beat it. This is a shortlist of where to look, not a reason to buy or sell in any one ZIP.
