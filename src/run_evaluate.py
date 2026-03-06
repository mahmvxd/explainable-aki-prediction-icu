import os
import json
import pandas as pd
from src.config import Config
from src.io_utils import ensure_dirs
from src.evaluate import evaluate_model

def main():
    cfg = Config()
    ensure_dirs(cfg.OUTPUT_DATA_DIR, cfg.OUTPUT_MODEL_DIR, cfg.OUTPUT_REPORT_DIR)

    data_path = os.path.join(cfg.OUTPUT_DATA_DIR, f"aki_dataset_h{cfg.HORIZON_HOURS}_{cfg.AKI_LABEL_MODE}.parquet")
    dataset = pd.read_parquet(data_path)

    for model_name in ["logreg", "hgb"]:
        model_path = os.path.join(cfg.OUTPUT_MODEL_DIR, f"{model_name}.joblib")
        report = evaluate_model(dataset, model_path=model_path, seed=cfg.RANDOM_SEED)

        out_json = os.path.join(cfg.OUTPUT_REPORT_DIR, f"eval_{model_name}.json")
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)

        print(model_name, "->", report)
        print("Saved:", out_json)

if __name__ == "__main__":
    main()