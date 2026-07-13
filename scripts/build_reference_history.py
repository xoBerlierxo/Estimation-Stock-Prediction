"""Generate the small history table the inference service uses to reconstruct
lag/rolling features for a prediction request.

Run once (and re-run whenever `data/raw/train.csv` changes):

    python scripts/build_reference_history.py
"""

from __future__ import annotations

from pathlib import Path

from src.data.load import load_raw_train
from src.utils.logging import get_logger

logger = get_logger(__name__)

RAW_TRAIN_PATH = Path("data/raw/train.csv")
OUT_PATH = Path("app/data/reference_history.parquet")
HISTORY_WEEKS = 12  # must be >= the largest lag/rolling window used in training


def main() -> None:
    df = load_raw_train(RAW_TRAIN_PATH)
    df = df.sort_values(["store_id", "sku_id", "week"])

    history = (
        df.groupby(["store_id", "sku_id"], group_keys=False)
        .tail(HISTORY_WEEKS)[
            [
                "store_id",
                "sku_id",
                "week",
                "units_sold",
                "total_price",
                "base_price",
                "is_featured_sku",
                "is_display_sku",
            ]
        ]
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(OUT_PATH, index=False)
    logger.info(
        "Wrote %d rows (%d store/sku combos x up to %d weeks) to %s",
        len(history),
        history.groupby(["store_id", "sku_id"]).ngroups,
        HISTORY_WEEKS,
        OUT_PATH,
    )


if __name__ == "__main__":
    main()
