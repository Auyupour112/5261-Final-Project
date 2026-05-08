import pandas as pd
import numpy as np
import os
from scipy import stats
from statsmodels.tsa.stattools import acf

data_dir = "data_tech_2016_2025"
output_dir = "data_with_signals"
os.makedirs(output_dir, exist_ok=True)

momentum_windows = [30, 60, 90]
ma_pairs = [(10, 30), (20, 100), (50, 200)]

for file in os.listdir(data_dir):
    if not file.endswith(".csv"):
        continue

    symbol = file.split("_")[0]
    path = os.path.join(data_dir, file)

    df = pd.read_csv(path)

    if "date" not in df.columns:
        print(symbol, "missing date column, skipped")
        continue

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "adjclose" not in df.columns:
        if "close" in df.columns:
            df["adjclose"] = df["close"]
        else:
            print(symbol, "missing price column, skipped")
            continue

    # daily returns
    df["ret_d1"] = df["adjclose"].pct_change()
    df["log_ret_d1"] = np.log1p(df["ret_d1"])

    # momentum signals
    # use yesterday's momentum to avoid using future information
    for w in momentum_windows:
        df[f"momentum_{w}d"] = df["adjclose"].pct_change(w)

        df[f"signal_momentum_{w}d"] = (
            df[f"momentum_{w}d"].shift(1) > 0
        ).astype(int)

    # moving average signals
    # use yesterday's moving average comparison
    for short_ma, long_ma in ma_pairs:
        df[f"ma_{short_ma}"] = df["adjclose"].rolling(short_ma).mean()
        df[f"ma_{long_ma}"] = df["adjclose"].rolling(long_ma).mean()

        df[f"signal_ma_{short_ma}_{long_ma}"] = (
            df[f"ma_{short_ma}"].shift(1) > df[f"ma_{long_ma}"].shift(1)
        ).astype(int)

    # combined signals: momentum signal and MA signal are both 1
    for w in momentum_windows:
        for short_ma, long_ma in ma_pairs:
            df[f"signal_combined_mom{w}_ma{short_ma}_{long_ma}"] = (
                (df[f"signal_momentum_{w}d"] == 1) &
                (df[f"signal_ma_{short_ma}_{long_ma}"] == 1)
            ).astype(int)

    # simple statistical check for daily log returns
    returns = df["log_ret_d1"].dropna()

    if len(returns) > 30:
        acf_vals = acf(returns, nlags=5)
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        print("\n", symbol)
        print("ACF first 5 lags:", np.round(acf_vals[:5], 4))
        print("t-stat:", round(t_stat, 4))
        print("p-value:", round(p_value, 4))

    out_path = os.path.join(output_dir, f"{symbol}_signals.csv")
    df.to_csv(out_path, index=False)

    print(symbol, "processed", out_path)

print("All signal files generated.")