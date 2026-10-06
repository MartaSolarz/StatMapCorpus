import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

SEED = 42
N_RANDOM = 143
N_RESERVE = 200
N_CODED = 157


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the 300-map test-retest sample")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="main corpus table (parquet)")
    parser.add_argument("--coded", type=Path, required=True,
                        help="frozen list of the uids in the coder validation sample")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus, columns=["uid", "is_duplicate_byte"])
    coded = [line.strip() for line in args.coded.read_text().splitlines()
             if line.strip()]
    assert len(coded) == len(set(coded)) == N_CODED, \
        f"expected {N_CODED} unique uids, got {len(coded)}"
    assert corpus["uid"].isin(coded).sum() == N_CODED, \
        "not every validation map is in the corpus"

    pool = (corpus.loc[~corpus["is_duplicate_byte"].astype(bool), "uid"]
            .loc[lambda s: ~s.isin(coded)]
            .sort_values()
            .to_numpy())
    print(f"drawing pool: {len(pool)} maps "
          f"(corpus without byte duplicates, without the {N_CODED} validation maps)")

    order = np.random.default_rng(SEED).permutation(len(pool))
    drawn = pool[order[:N_RANDOM]]
    reserve = pool[order[N_RANDOM:N_RANDOM + N_RESERVE]]

    rows = [{"uid": uid, "layer": "validation", "order": i}
            for i, uid in enumerate(coded)]
    rows += [{"uid": uid, "layer": "random", "order": i}
             for i, uid in enumerate(drawn)]
    sample = pd.DataFrame(rows)
    assert len(sample) == N_CODED + N_RANDOM and sample["uid"].is_unique

    args.out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out / "sample_uids.csv", index=False)
    pd.DataFrame({"uid": reserve, "order": range(len(reserve))}).to_csv(
        args.out / "sample_uids_reserve.csv", index=False)
    print(f"wrote sample_uids.csv ({len(sample)}) "
          f"and sample_uids_reserve.csv ({len(reserve)})")


if __name__ == "__main__":
    main()
