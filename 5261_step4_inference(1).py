# One-sample t-test + Paired t-test + 95% CI

import os
import numpy as np
import pandas as pd
from scipy import stats


BT_DIR = "bt_outputs"
OUT_DIR = "inference_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RETURNS_PATH = os.path.join(BT_DIR, "daily_returns_all.csv")
SUMMARY_PATH = os.path.join(BT_DIR, "summary_all.csv")


def mean_ci(series, alpha=0.05):
    x = pd.Series(series).dropna()
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    lower = mean - tcrit * se
    upper = mean + tcrit * se
    return lower, upper


def one_sample_ttest(series):
    x = pd.Series(series).dropna()
    n = len(x)
    if n < 2:
        return {
            "n_obs": n,
            "mean_daily_return": np.nan,
            "t_stat": np.nan,
            "p_value_two_sided": np.nan,
            "p_value_one_sided_positive": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
        }

    t_stat, p_two = stats.ttest_1samp(x, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    ci_l, ci_u = mean_ci(x)

    return {
        "n_obs": n,
        "mean_daily_return": x.mean(),
        "t_stat": t_stat,
        "p_value_two_sided": p_two,
        "p_value_one_sided_positive": p_one,
        "ci95_lower": ci_l,
        "ci95_upper": ci_u,
    }


def paired_ttest(strategy_series, benchmark_series):
    temp = pd.concat([strategy_series, benchmark_series], axis=1).dropna()
    temp.columns = ["strategy", "benchmark"]
    diff = temp["strategy"] - temp["benchmark"]

    n = len(diff)
    if n < 2:
        return {
            "n_obs_paired": n,
            "mean_daily_diff_vs_buyhold": np.nan,
            "paired_t_stat": np.nan,
            "paired_p_value_two_sided": np.nan,
            "paired_p_value_one_sided_positive": np.nan,
            "paired_ci95_lower": np.nan,
            "paired_ci95_upper": np.nan,
        }

    t_stat, p_two = stats.ttest_1samp(diff, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    ci_l, ci_u = mean_ci(diff)

    return {
        "n_obs_paired": n,
        "mean_daily_diff_vs_buyhold": diff.mean(),
        "paired_t_stat": t_stat,
        "paired_p_value_two_sided": p_two,
        "paired_p_value_one_sided_positive": p_one,
        "paired_ci95_lower": ci_l,
        "paired_ci95_upper": ci_u,
    }


if __name__ == "__main__":
    if not os.path.exists(RETURNS_PATH):
        raise FileNotFoundError("Missing bt_outputs/daily_returns_all.csv. Run backtest file first.")
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError("Missing bt_outputs/summary_all.csv. Run backtest file first.")

    returns_df = pd.read_csv(RETURNS_PATH, parse_dates=["date"]).set_index("date")
    summary_df = pd.read_csv(SUMMARY_PATH)

    if "buyhold_equal" not in returns_df.columns:
        raise ValueError("buyhold_equal benchmark not found in daily_returns_all.csv")

    bh = returns_df["buyhold_equal"]
    rows = []

    for strategy in returns_df.columns:
        if strategy == "buyhold_equal":
            continue

        row = {"strategy": strategy}
        row.update(one_sample_ttest(returns_df[strategy]))
        row.update(paired_ttest(returns_df[strategy], bh))
        rows.append(row)

    inference_df = pd.DataFrame(rows)

    inference_df = inference_df.merge(
        summary_df[[
            "strategy",
            "final_value",
            "total_return",
            "annual_return",
            "volatility",
            "max_drawdown",
            "sharpe",
            "total_trades"
        ]],
        on="strategy",
        how="left"
    )

    inference_df = inference_df.sort_values(["sharpe", "mean_daily_return"], ascending=False)
    inference_df.to_csv(os.path.join(OUT_DIR, "inference_all.csv"), index=False)

    best_strategy = (
        summary_df[summary_df["strategy"] != "buyhold_equal"]
        .sort_values("sharpe", ascending=False)
        .iloc[0]["strategy"]
    )

    best_row = inference_df[inference_df["strategy"] == best_strategy].copy()
    best_row.to_csv(os.path.join(OUT_DIR, "inference_best_strategy.csv"), index=False)

    print("Saved:")
    print("- inference_outputs/inference_all.csv")
    print("- inference_outputs/inference_best_strategy.csv")
