# biolab/reporting/aggregator.py

from collections import defaultdict
import numpy as np
import pandas as pd

def concat_lists(series_of_lists: pd.Series) -> list:
    """
    Flatten a series of lists into a single list.
    """
    return [x for sub in series_of_lists for x in sub]

def combine_scores_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repeated runs by combining score lists and computing mean and standard deviation.

    Expects that the DataFrame contains the following fields generated by the parser:
      - model_id
      - display_name
      - output_dir
      - task_name, metric_name, is_higher_better,
      - train_scores, test_scores

    Returns a DataFrame with additional columns:
        - train_mean, train_std, test_mean, test_std.
    """
    group_cols = ["model_id", "display_name", "output_dir", "task_name", "metric_name", "is_higher_better"]

    aggregated = df.groupby(group_cols, dropna=False, as_index=False).agg({
        "train_scores": concat_lists,
        "test_scores": concat_lists,
    })

    def mean_std(scores: list) -> tuple[float | None, float | None]:
        if not scores:
            return (None, None)
        arr = np.array(scores, dtype=float)
        return arr.mean(), arr.std(ddof=1)

    train_means, train_stds, test_means, test_stds = [], [], [], []
    for _, row in aggregated.iterrows():
        tr_mean, tr_std = mean_std(row["train_scores"])
        te_mean, te_std = mean_std(row["test_scores"])
        train_means.append(tr_mean)
        train_stds.append(tr_std)
        test_means.append(te_mean)
        test_stds.append(te_std)

    aggregated["train_mean"] = train_means
    aggregated["train_std"] = train_stds
    aggregated["test_mean"] = test_means
    aggregated["test_std"] = test_stds

    return aggregated

def make_pivot_for_metric(df_agg: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Create a pivot table for a specific metric with tasks as rows and model display names as columns.
    The cell entries are formatted as "mean ± std", and the columns are sorted so that the leftmost column 
    corresponds to the on-average best-performing model.

    Parameters
    ----------
    df_agg : pd.DataFrame
        Aggregated DataFrame with metric results.
    metric_name : str
        The metric name to filter on.

    Returns
    -------
    pd.DataFrame
        Pivot table with sorted columns.
    """
    metric_df = df_agg[df_agg["metric_name"] == metric_name].copy()

    def fmt(mean, std):
        if pd.isna(mean):
            return "–"
        if pd.isna(std) or std == 0:
            return f"{mean:.3f}"
        return f"{mean:.3f} ± {std:.3f}"

    metric_df["test_str"] = metric_df.apply(
        lambda row: fmt(row["test_mean"], row["test_std"]), axis=1
    )

    # Use 'display_name' for labels if available, otherwise fallback to 'model_id'
    metric_df["model_label"] = metric_df.get("display_name", metric_df["model_id"])

    pivot = metric_df.pivot_table(
        index="task_name",
        columns="model_label",
        values="test_str",
        aggfunc="first"  # Expecting one value per (task, model)
    )
    pivot.sort_index(axis="index", inplace=True)

    # Sort columns based on average test_mean per model for this metric.
    if not metric_df.empty:
        is_hb = metric_df["is_higher_better"].iloc[0]
        avg_scores = metric_df.groupby("model_label")["test_mean"].mean()
        # For higher-is-better metrics, sort descending; for lower-is-better, ascending.
        if is_hb:
            sorted_models = avg_scores.sort_values(ascending=False).index.tolist()
        else:
            sorted_models = avg_scores.sort_values(ascending=True).index.tolist()
        pivot = pivot.reindex(columns=sorted_models)
    return pivot

def compute_win_rates(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model win rates across tasks and metrics.

    Uses 'display_name' (or 'model_id' if display is absent) as the unique identifier.
    Returns a DataFrame with win rate statistics.
    """
    df = df_agg.copy()
    # Use display_name if available, otherwise fallback to model_id.
    if "display_name" in df.columns:
        df["model_label"] = df["display_name"]
    else:
        df["model_label"] = df["model_id"]

    group_map = defaultdict(list)
    ihb_map = {}

    for row in df.itertuples(index=False):
        key = (row.task_name, row.metric_name)
        group_map[key].append((row.model_label, row.test_mean))
        ihb_map[key] = row.is_higher_better

    all_models = sorted(df["model_label"].unique().tolist())
    presence_map = {key: {m for (m, _) in model_list} for key, model_list in group_map.items()}
    intersection_tasks = [key for key, present_ids in presence_map.items() if set(all_models).issubset(present_ids)]

    winners_union = {m: 0 for m in all_models}
    winners_intersection = {m: 0 for m in all_models}
    possible_union = {m: 0 for m in all_models}
    possible_intersection = {m: 0 for m in all_models}

    for key, model_list in group_map.items():
        is_hb = ihb_map[key]
        valid_list = [(m, x) for (m, x) in model_list if pd.notna(x)]
        if not valid_list:
            continue

        best_val = max(x for (_, x) in valid_list) if is_hb else min(x for (_, x) in valid_list)
        winning_models = [m for (m, x) in valid_list if x == best_val]
        present_ids = {m for (m, _) in valid_list}

        for m in present_ids:
            possible_union[m] += 1
        for m in winning_models:
            winners_union[m] += 1

        if key in intersection_tasks:
            for m in all_models:
                possible_intersection[m] += 1
            for m in winning_models:
                winners_intersection[m] += 1

    results = []
    for m in all_models:
        pi = possible_intersection[m]
        pu = possible_union[m]
        wi = winners_intersection[m]
        wu = winners_union[m]
        results.append({
            "model_label": m,
            "wins_intersection": wi,
            "wins_union": wu,
            "possible_intersection": pi,
            "possible_union": pu,
            "win_rate_intersection": wi / pi if pi > 0 else 0.0,
            "win_rate_union": wu / pu if pu > 0 else 0.0,
        })

    return pd.DataFrame(results)
