import os
import pandas as pd
from src.config import Config
from src.io_utils import ensure_dirs
from src.train import train_models

def main():
    cfg = Config()
    ensure_dirs(cfg.OUTPUT_DATA_DIR, cfg.OUTPUT_MODEL_DIR, cfg.OUTPUT_REPORT_DIR)

    data_path = os.path.join(cfg.OUTPUT_DATA_DIR, f"aki_dataset_h{cfg.HORIZON_HOURS}_{cfg.AKI_LABEL_MODE}.parquet")
    dataset = pd.read_parquet(data_path)

    results = train_models(dataset, output_dir=cfg.OUTPUT_MODEL_DIR, seed=cfg.RANDOM_SEED)

    out_csv = os.path.join(cfg.OUTPUT_REPORT_DIR, "train_results.csv")
    results.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()