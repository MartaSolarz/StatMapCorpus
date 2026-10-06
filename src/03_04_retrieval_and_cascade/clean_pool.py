import argparse
from pathlib import Path

import polars as pl

import config


def main():
    parser = argparse.ArgumentParser(
        description="Normalise URLs, drop duplicate URLs and excluded domains from the Stage 2 predictions")
    parser.add_argument("--input", required=True,
                        help="predictions written by src/02_classifier/3_inference/apply_model_large_scale.py")
    parser.add_argument("--output", default=str(config.PREDICTIONS_FILE))
    args = parser.parse_args()

    df = pl.read_parquet(args.input)
    print(f"input: {len(df):,} records")

    df = df.with_columns(
        pl.col("url")
        .str.replace(r"^https?://", "")
        .str.replace(r"^www\.", "")
        .str.strip_chars("/")
        .alias("url_norm")
    )
    df = df.with_columns(pl.col("url_norm").str.extract(r"^([^/]+)", 1).alias("domain"))

    before = len(df)
    df = df.sort("pred_proba", descending=True).unique(subset=["url_norm"], keep="first")
    print(f"duplicate URLs removed: {before - len(df):,}")

    before = len(df)
    df = df.filter(~pl.col("domain").is_in(config.EXCLUDE_DOMAINS))
    print(f"records of excluded domains removed: {before - len(df):,}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.select(["uid", "url", "url_norm", "domain", "pred_proba", "score"]).write_parquet(output)
    print(f"output: {len(df):,} records, {df['domain'].n_unique():,} domains -> {output}")


if __name__ == "__main__":
    main()
