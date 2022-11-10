# %%
import warnings
import os
from collections import defaultdict

import plotly
import pandas as pd

from greykite.common.constants import TIME_COL, VALUE_COL
from greykite.framework.benchmark.data_loader_ts import DataLoader
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries

from greykite.framework.templates.autogen.forecast_config import \
    EvaluationPeriodParam, ForecastConfig, \
    MetadataParam, ModelComponentsParam

from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.utils.result_summary import summarize_grid_search_results

warnings.filterwarnings('ignore')


# %%
data_loader = DataLoader()
agg_func = {
    'count': 'sum'
}

df = data_loader.load_bikesharing(
    agg_freq='weekly',
    agg_func=agg_func
)

df.drop(df.head(1).index, inplace=True)
df.drop(df.tail(1).index, inplace=True)
df.reset_index(inplace=True)

# %% 
ts = UnivariateTimeSeries()
ts.load_data(
    df=df,
    time_col='ts',
    value_col='count',
    freq='W-MON'
)

print(ts.df.head())
# %%
fig = ts.plot()
plotly.io.show(fig)

# %%
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature='month',
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_time_feature='year',
    overlay_style={
        'line': {
            'width': 1},
        'opacity': 0.5
            },
    xlabel='month',
    ylabel=ts.original_value_col,
    title='Yearly seasonality by year'
)

plotly.io.show(fig)
# %%
fig = ts.plot_quantiles_and_overlays(
    groupby_time_feature='woy',
    show_mean=True,
    show_quantiles=False,
    show_overlays=True,
    center_values=True,
    overlay_label_time_feature='year',
    overlay_style={
        'line': {
            'width': 1},
        'opacity': 0.5
            },
    xlabel='week of year',
    ylabel=ts.original_value_col,
    title='Yearly seasonality by year(centered)'
)

plotly.io.show(fig)
# %% fit greykite model
forecast_horizon = 4
time_col, value_col = TIME_COL, VALUE_COL

metadata = MetadataParam(
    time_col=time_col,
    value_col=value_col,
    freq='W-MON'
)

# %%
cv_min_train_periods = 52 * 2
# Let CV use most recent splits for cross validation.
cv_use_most_recent_splits = True
cv_max_splits = 6

evaluation_period = EvaluationPeriodParam(
    test_horizon=forecast_horizon,
    cv_horizon=forecast_horizon,
    periods_between_train_test=0,
    cv_min_train_periods=cv_min_train_periods,
    cv_expanding_window=True,
    cv_use_most_recent_splits=cv_use_most_recent_splits,
    cv_periods_between_splits=None,
    cv_periods_between_train_test=0,
    cv_max_splits=cv_max_splits
)

# %%


def get_model_result_summary(result):
    model = result.model[-1]
    backtest = result.backtest
    grid_search = result.grid_search

    # print(model.summary())

    cv_results = summarize_grid_search_results(
        grid_search=grid_search,
        decimals=2,
        cv_report_metrics=None,
        column_order=[
            'rank', 'mean_test', 'split_test', 'mean_train', 'split_train',
            'mean_fit_time', 'mean_score_time', 'params'
        ]
    )

    backtest_eval = defaultdict(list)
    for metric, value in backtest.train_evaluation.items():
        backtest_eval[metric].append(value)
        backtest_eval[metric].append(backtest.test_evaluation[metric])
    metrics = pd.DataFrame(
        backtest_eval,
        index=['train', 'test']
    ).T
    print(f'CV Results:\n {cv_results.transpose()}')
    print(f'Train/Test evaluation: \n {metrics}')

    return cv_results, metrics


# %%
autoregression = None
extra_pred_cols = [
    'ct1', 'ct_sqrt', 'ct1:C(month, levels=list(range(1, 13)))'
]

seasonality = {
    'yearly_seasonality': 25,
    'quarterly_seasonality': 0,
    'monthly_seasonality': 0,
    'weekly_seasonality': 0,
    'daily_seasonality': 0
}
changepoints = {
    'changepoints_dict': {
        'method': 'auto',
        'resample_freq': '7D',
        'regularization_strength': 0.5,
        'potential_changepoint_distance': '14D',
        'no_changepoint_distance_from_end': '60D',
        'yearly_seasonality_order': 25,
        'yearly_seasonality_change_freq': None
    },
    'seasonality_changepoints_dict': None
}

events = {
    'holiday_lookup_countries': []
}

growth = {
    'growth_term': None
}

custom = {
    'feature_sets_enabled': False,
    'fit_algorithm_dict': {
        'fit_algorithm': 'ridge'
    },
    'extra_pred_cols': extra_pred_cols
}

model_components = ModelComponentsParam(
    seasonality=seasonality,
    changepoints=changepoints,
    events=events,
    growth=growth,
    custom=custom
)

# %%
forecast_config = ForecastConfig(
    metadata_param = metadata,
    forecast_horizon=forecast_horizon,
    coverage=0.95,
    evaluation_period_param=evaluation_period,
    model_components_param=model_components
)

forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=ts.df,
    config=forecast_config
)
# %%
get_model_result_summary(result)

# %%
fig = result.backtest.plot()
plotly.io.show(fig)

# %% 
fig = result.forecast.plot()
plotly.io.show(fig)

# %%
fig = result.forecast.plot_components()
plotly.io.show(fig)
# %%
