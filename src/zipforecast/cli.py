import argparse
import logging

import pandas as pd

from zipforecast import config
from zipforecast.download import download_all
from zipforecast.model import (
    evaluate,
    factor_history,
    feature_importance,
    rank_nyc,
    summarize,
)
from zipforecast.panel import build_panel
from zipforecast.report import write_report


def _load_panel() -> pd.DataFrame:
    if not config.PANEL_FILE.exists():
        raise SystemExit("no panel yet; run `zipforecast panel` first")
    return pd.read_parquet(config.PANEL_FILE)


def cmd_download(_args) -> None:
    download_all()


def cmd_panel(_args) -> None:
    build_panel()


def cmd_evaluate(args) -> None:
    panel = _load_panel()
    results = {h: evaluate(panel, h) for h in args.horizons}
    summary = summarize(pd.concat(results.values(), ignore_index=True))
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    importance = {h: feature_importance(panel, h) for h in args.horizons}
    factors = {h: factor_history(panel, h) for h in args.horizons}
    rankings = {h: rank_nyc(panel, h) for h in args.horizons}
    write_report(results, summary, importance, factors, rankings, panel)


def cmd_rank(args) -> None:
    panel = _load_panel()
    for h in args.horizons:
        ranking = rank_nyc(panel, h)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(config.OUTPUT_DIR / f"nyc_ranking_{h}y.csv", index=False)
        print(ranking.head(25).to_string(index=False))


def cmd_all(args) -> None:
    download_all()
    build_panel()
    cmd_evaluate(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs, which would include the Census API key.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="zipforecast")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, doc in [
        ("download", cmd_download, "fetch Zillow, ACS and gazetteer files into data/raw"),
        ("panel", cmd_panel, "build the ZIP x origin-year panel into data/processed"),
        ("evaluate", cmd_evaluate, "walk-forward evaluation, rankings and output/REPORT.md"),
        ("rank", cmd_rank, "fit on all realized origins and rank NYC-metro ZIPs"),
        ("all", cmd_all, "download, panel, evaluate"),
    ]:
        p = sub.add_parser(name, help=doc)
        p.add_argument(
            "--horizons",
            type=int,
            nargs="+",
            default=list(config.HORIZONS),
            help="forecast horizons in years (default: %(default)s)",
        )
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)
