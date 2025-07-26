# %% import libratries.
import polars as pl
import pandas as pd

# %% read data
url = "https://unitedstates.github.io/congress-legislators/legislators-current.csv"

dtypes = {
    "last_name": pl.Categorical,
    "first_name": pl.Categorical,
    "gender": pl.Categorical,
    "type": pl.Categorical,
    "state": pl.Categorical,
    "party": pl.Categorical,
}

# %%
dataset = pl.read_csv(url,dtypes=dtypes, null_values=["", "NA", "N/A"])

# %%
dataset = dataset.with_columns(
    pl.col("birthday").str.strptime(pl.Date, strict=False)
)


# %% read data
q = (
    dataset
    .lazy()
    .group_by("first_name")
    .agg(
    [
        pl.count(),
        pl.col('gender'),
        pl.first("last_name"),
    ]
    )
    .sort("count", descending=True)
    .limit(5)
)
df = q.collect()
# %%
df