import numpy as np
import pandas as pd


BASE_COLS = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BUN","Creatinine","Glucose","Lactate","WBC","Platelets","Hgb","Hct",
    "Potassium","Chloride","Calcium","Magnesium","Phosphate",
    "Bilirubin_total","AST",
    "Age","Gender","HospAdmTime","ICULOS"
]

WINDOWS = [1, 3, 6, 12, 24, 48]


def build_demo_feature_row(user_input: dict, feature_names: list) -> pd.DataFrame:
    """
    Build a single-row dataframe with exactly the same columns expected by the trained model.
    Since the demo only has current values (not full history), we approximate rolling stats
    by repeating the current value across all engineered windows.
    """

    row = {}

    # 1. base columns
    for col in BASE_COLS:
        row[col] = user_input.get(col, np.nan)

    # 2. missing indicators
    for col in BASE_COLS:
        row[f"{col}_isna"] = int(pd.isna(row[col]))

    # 3. engineered features
    for col in BASE_COLS:
        val = row[col]

        for k in WINDOWS:
            # no historical sequence in demo, so deltas default to 0 if value present
            row[f"{col}_delta_{k}h"] = 0.0 if not pd.isna(val) else np.nan
            row[f"{col}_mean_{k}h"] = val
            row[f"{col}_min_{k}h"] = val
            row[f"{col}_max_{k}h"] = val

    # 4. kidney-specific extras
    cr = row.get("Creatinine", np.nan)
    row["cr_rollmin_48h"] = cr
    row["cr_above_rollmin_48h"] = 0.0 if not pd.isna(cr) else np.nan
    row["cr_slope_6h"] = 0.0 if not pd.isna(cr) else np.nan
    row["cr_slope_12h"] = 0.0 if not pd.isna(cr) else np.nan

    # 5. build dataframe
    df = pd.DataFrame([row])

    # 6. ensure all expected columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    # 7. keep exact training order
    df = df[feature_names]

    return df