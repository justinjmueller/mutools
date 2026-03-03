import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def load_logs(
    log_dir: Path,
    pattern: str = "*.csv",
    method: str = "concat",
    bpe: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the training/validation log data from CSV files. The method of
    aggregation can be specified via the `method` parameter.

    Parameters
    ----------
    log_dir : Path
        Directory containing log CSV files.
    pattern : str
        Pattern to match CSV files.
    method : str
        Method to aggregate logs. Options are:
        - 'concat': Concatenate all logs (default).
        - 'mean': Compute the mean
    bpe : Optional[int]
        Number of iterations per epoch. This is used to compute the
        correspondence between checkpoints and epochs.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame containing all log data.
    """
    # Load all matching CSV files into DataFrames. We need to permute
    # the order of files to ensure chronological order based on the
    # number in the file name.
    files = log_dir.glob(pattern)
    files = sorted(files, key=lambda x: int(str(x).split("-")[-1][:-4]))
    dfs = [pd.read_csv(x) for x in files]
    files = [x.name for x in files]

    if not dfs:
        return pd.DataFrame()

    if method == "concat":
        # Determine the size of each log segment based on the starting
        # iteration of each log.
        first = [x["iter"].iloc[0] for x in dfs]
        size = np.diff(first, append=dfs[-1]["iter"].iloc[-1])

        result = pd.concat([x.iloc[: size[i]] for i, x in enumerate(dfs)], ignore_index=True)

        # If iteration is not-unique across logs, average each column
        # over duplicate iterations.
        if not result["iter"].is_unique:
            result = result.groupby("iter").mean().reset_index()

        return result.sort_values("epoch")

    elif method == "mean":
        # Determine the checkpoints based on the first entry of each
        # log. We require that `bpe` is provided to convert iterations
        # to epochs.
        if bpe is None:
            raise ValueError("bpe (iterations per epoch) must be provided for 'mean' method.")
        chk = [int(x.split("-")[-1][:-4]) / bpe for x in files]

        # For each field and each log, calculate the mean and standard
        # deviation over all iterations.
        result = {"checkpoint": chk}
        for field in dfs[0].columns:
            if field in ["iter", "epoch", "first_entry"]:
                continue

            means = []
            stds = []
            for i, df in enumerate(dfs):
                means.append(df[field].mean())
                stds.append(df[field].std())

            result[field + "_mean"] = means
            result[field + "_std"] = stds

        return pd.DataFrame(result)
