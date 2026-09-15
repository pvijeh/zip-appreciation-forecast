import argparse
import logging

import pandas as pd

from zipforecast import config
from zipforecast.download import download_all
from zipforecast.model import (
    beta_history,
    evaluate,
    factor_history,
    feature_group_ablation,
    feature_importance,
    rank_nyc,
    summarize,
)
from zipforecast.panel import build_panel
from zipforecast.report import write_ranking_json, write_report, write_results_summary

log = logging.getLogger(__name__)


def _load_panel(horizons: list[int]) -> pd.DataFrame:
    if not config.PANEL_FILE.exists():
        raise SystemExit("no panel yet; run `zipforecast panel` first")
    panel = pd.read_parquet(config.PANEL_FILE)
    missing = [h for h in horizons if f"target_{h}y" not in panel.columns]
    if missing:
        raise SystemExit(
            f"panel has no targets for horizons {missing}; it was built by an older version, "
            "run `zipforecast panel` to rebuild it"
        )
    return panel


def _load_summary(panel: pd.DataFrame) -> pd.DataFrame | None:
    """The last evaluation's summary, so `rank` keeps the skill metadata in the JSON files.

    Only if it was computed on this panel: a rebuilt panel has a new latest origin and revised
    history, and skill measured on the old one must not be attached to the new rankings."""
    path = config.OUTPUT_DIR / "evaluation_summary.csv"
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    current = str(panel["origin"].max().date())
    stored = summary["panel_origin"].iloc[0] if "panel_origin" in summary.columns else None
    if stored != current:
        log.warning(
            "evaluation_summary.csv is from panel origin %s, panel is %s; "
            "run `zipforecast evaluate` to attach skill metadata to the JSON files",
            stored,
            current,
        )
        return None
    return summary


def cmd_download(_args) -> None:
    download_all()


def cmd_panel(_args) -> None:
    build_panel()


def cmd_evaluate(args) -> None:
    panel = _load_panel(args.horizons)
    results = {h: evaluate(panel, h) for h in args.horizons}
    summary = summarize(pd.concat(results.values(), ignore_index=True))
    summary["panel_origin"] = str(panel["origin"].max().date())
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    importance = {h: feature_importance(panel, h) for h in args.horizons}
    factors = {h: factor_history(panel, h) for h in args.horizons}
    rankings = {h: rank_nyc(panel, h) for h in args.horizons}
    beta = beta_history(panel, 1) if 1 in args.horizons else None
    ablation = feature_group_ablation(panel, 1) if 1 in args.horizons else None
    write_report(results, summary, importance, factors, rankings, panel, beta, ablation)


def cmd_rank(args) -> None:
    panel = _load_panel(args.horizons)
    summary = _load_summary(panel)
    for h in args.horizons:
        ranking = rank_nyc(panel, h)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(config.OUTPUT_DIR / f"nyc_ranking_{h}y.csv", index=False)
        write_ranking_json(ranking, h, summary)
        if h == 1:
            write_results_summary(ranking, summary)
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
            choices=config.HORIZONS,
            default=list(config.HORIZONS),
            help="forecast horizons in years (default: %(default)s)",
        )
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)
