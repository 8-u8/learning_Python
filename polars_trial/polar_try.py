# %% import libratries.
import polars as pl
import pandas as pd

# %% read data
url = "https://theunitedstates.io/congress-legislators/legislators-historical.csv"

dtypes = {
    "first_name": pl.Categorical,
    "gender": pl.Categorical,
    "type": pl.Categorical,
    "state": pl.Categorical,
    "party": pl.Categorical,
}

dataset = pl.read_csv(url, dtype=dtypes)\
    .with_column(pl.col("birthday").str.strptime(pl.Date, strict=False))

# %% read data
q = (
    dataset.lazy()
    .groupby("first_name")
    .agg(
    [
        pl.count(),
        pl.col('gender').list(),
        pl.first("last_name"),
    ]
    )
    .sort("count", reverse=True)
    .limit(5)
)
df = q.collect()
# %%
