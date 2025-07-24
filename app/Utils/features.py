import pandas as pd
import numpy as np

def extract_features_from_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes solar wind data as a DataFrame and computes 4 physics-informed features.

    Expected Columns:
        - timestamp (datetime)
        - proton_density
        - proton_speed
        - proton_temperature
        - alpha_density

    Automatically resamples to 5-minute averages if input interval < 5min.
    Raises error if input interval > 5min.
    """

    required_cols = {"timestamp", "proton_density", "proton_speed", "proton_temperature", "alpha_density"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Please ensure your CSV contains all the following columns:\n\n"
                         + "\n".join(required_cols) +
                         "\n\nEach row should represent ≤5-minute interval measurements.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])  # drop invalid timestamps
    df = df.sort_values("timestamp").reset_index(drop=True)

    diffs = df["timestamp"].diff().dropna().dt.total_seconds()
    if not diffs.empty:
        mode_interval = diffs.mode()[0]
        if mode_interval < 300:
            # Resample to 5min
            df = df.set_index("timestamp").resample("5min").mean().dropna().reset_index()
        elif mode_interval > 300:
            raise ValueError(f"Your data is too sparse (interval ≈ {int(mode_interval)}s). "
                             f"Please provide higher-resolution data (≤5min).")

    df = df.rename(columns={
        "proton_density": "Np",
        "proton_speed": "Vp",
        "proton_temperature": "Tp",
        "alpha_density": "Alpha"
    })

    df["alpha_proton_ratio"] = df["Alpha"] / df["Np"].replace(0, np.nan)
    df["vp_std_15min"] = df["Vp"].rolling(window=3, center=True).std()
    df["alpha_over_vpstd"] = df["alpha_proton_ratio"] / df["vp_std_15min"].replace(0, np.nan)
    df["alpha_tp_ratio"] = df["Alpha"] / df["Tp"].replace(0, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "alpha_proton_ratio", "vp_std_15min", "alpha_over_vpstd", "alpha_tp_ratio"
    ])

    if df.empty:
        return pd.DataFrame([{
            "alpha_proton_ratio": np.nan,
            "vp_std_15min": np.nan,
            "alpha_over_vpstd": np.nan,
            "alpha_tp_ratio": np.nan
        }])

    return pd.DataFrame([{
        "alpha_proton_ratio": df["alpha_proton_ratio"].mean(),
        "vp_std_15min": df["vp_std_15min"].mean(),
        "alpha_over_vpstd": df["alpha_over_vpstd"].mean(),
        "alpha_tp_ratio": df["alpha_tp_ratio"].mean()
    }])
