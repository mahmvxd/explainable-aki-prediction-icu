import os
import pandas as pd
from src.config import Config
from src.io_utils import ensure_dirs
from src.build_dataset import build_dataset

def main():
    cfg = Config()
    ensure_dirs(cfg.OUTPUT_DATA_DIR, cfg.OUTPUT_MODEL_DIR, cfg.OUTPUT_REPORT_DIR)

    dataset = build_dataset(
        folders=[cfg.TRAIN_A_DIR, cfg.TRAIN_B_DIR],
        label_mode=cfg.AKI_LABEL_MODE,
        horizon_hours=cfg.HORIZON_HOURS,
        windows=cfg.WINDOWS,
        min_history_hours=cfg.MIN_HISTORY_HOURS,
        max_patients=None  # set e.g. 200 for quick testing
    )

    out_path = os.path.join(cfg.OUTPUT_DATA_DIR, f"aki_dataset_h{cfg.HORIZON_HOURS}_{cfg.AKI_LABEL_MODE}.parquet")
    dataset.to_parquet(out_path, index=False)
    print("Saved:", out_path)
    print("Shape:", dataset.shape)
    print("Target prevalence:", dataset["AKI_within_horizon"].mean())

if __name__ == "__main__":
    main()