# %% import libraries
import numpy as np
import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt

# %% load data
# using cigar data from statsmodels
cigar = sm.datasets.get_rdataset("Cigar", "Ecdat").data
# %% preprocessing

def preprocess_data(df: pd.DataFrame,
                    intercept: bool=True) -> pd.DataFrame:
    # 古い時系列を削除、州を絞る
    target_states = [3,5]
    condition = ( 
        (df.loc[:, 'year'] >= 70) &
        (df.loc[:, 'state'].isin(target_states))
    )
    df_out = df.loc[condition, :].copy().reset_index(drop=True)
    # df_out = df_out.reset_index(drop=True) 
    
    # yearの最小値を1として経過年数を作成する
    min_year = df_out.loc[:, 'year'].min()
    df_out.loc[:, 'time'] = df_out.loc[:, 'year'] - min_year + 1

    # 1980年を介入点として、介入点Tとダミー変数Dを作成
    df_out.loc[:, 'T'] = (df_out.loc[:, 'year'] == 78).astype(int)
    df_out.loc[:, 'D'] = (df_out.loc[:, 'year'] >= 78).astype(int)

    # 介入後の経過年数を作成
    # 1980年以前は0、1980年は1、1981年は2、...
    groupby_cols = ['state', 'D']
    df_out.loc[:, 'time_after'] = (
        df_out.loc[:, groupby_cols]
        .groupby('state')
        .cumsum()['D']
    )

    if intercept:
        df_out['const'] = int(1)

    return df_out

cigar_pp = preprocess_data(cigar, intercept=True)

# %% template chunk: check data
cigar_pp


# %%
# Python (statsmodels: NW標準誤差調整)
# OLSモデルを使用し、標準誤差計算時に調整を行う
def OLS_wrapper(
    df: pd.DataFrame,
    y: str,
    time: str,
    D: str,
    time_after: str,
    cooperate: list[str],
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    OLSモデルを使用し、標準誤差計算時に調整を行う

    Parameters
    ----------
    df : pd.DataFrame
    データフレーム
    y : str
    目的変数のカラム名
    time : str
    時間変数のカラム名
    D : str
    介入変数のカラム名 (0/1)
    cooperate : list[str]
    介入と相互作用する説明変数のカラム名リスト

    Returns
    -------
    sm.regression.linear_model.RegressionResultsWrapper
    statsmodelsのOLSモデルフィット結果オブジェクト
    """
    # 説明変数の設定
    if cooperate is None:
        cooperate = []
    exog_vars = ['const', time, D, time_after] + cooperate
    X = df[exog_vars]

    # 目的変数の設定
    Y = df[y]

    # OLSモデルの適用
    model = sm.OLS(Y, X)
    
    # 系列相関を考慮した標準誤差の計算によるフィッティング
    out = model.fit(cov_type='HAC', cov_kwds={'maxlags':3})
    # out = model.fit()
    return out

# %% model 
reg_res = OLS_wrapper(
    df=cigar_pp,
    y='sales',
    time='time',
    D='D',
    time_after='time_after',
    cooperate=['state']
)
reg_res.summary()


# %% visualization
def plot_itas_by_state(df, reg_results, feature_cols, intervention_year=78):
    """
    Create ITAS plots for each state separately
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data containing all states
    reg_results : statsmodels results object
        Fitted regression results
    feature_cols : list
        List of feature column names for prediction
    intervention_year : int
        Year of intervention (default: 78)
    """
    states = df['state'].unique()
    n_states = len(states)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_states, figsize=(6*n_states, 6))
    if n_states == 1:
        axes = [axes]
    
    for i, state in enumerate(states):
        ax = axes[i]
        
        # Filter data for current state
        state_data = df[df['state'] == state].copy()
        
        # Create counterfactual data (intervention effects set to 0)
        state_counterfactual = state_data.copy()
        state_counterfactual.loc[:, 'D'] = 0
        state_counterfactual.loc[:, 'time_after'] = 0
        
        # Calculate predictions
        # Counterfactual predictions
        X_counterfactual = state_counterfactual[feature_cols]
        pred_counterfactual = reg_results.get_prediction(X_counterfactual).summary_frame()
        pred_counterfactual = pred_counterfactual[['mean']]
        pred_counterfactual.columns = ['pred_counterfactual']
        
        # Actual predictions with confidence intervals
        X_actual = state_data[feature_cols]
        pred_actual = reg_results.get_prediction(X_actual).summary_frame()
        pred_actual = pred_actual[['mean', 'mean_ci_lower', 'mean_ci_upper']]
        pred_actual.columns = ['pred_actual', 'pred_actual_ci_lower', 'pred_actual_ci_upper']
        
        # Add predictions to state data
        state_plot = pd.concat([
            state_data,
            pred_counterfactual['pred_counterfactual'],
            pred_actual[['pred_actual', 'pred_actual_ci_lower', 'pred_actual_ci_upper']]
        ], axis=1)
        
        # Plot actual values
        ax.scatter(state_plot['year'], state_plot['sales'], 
                  alpha=0.6, color='blue', label='Actual Sales', s=30)
        
        # Pre-intervention fit
        pre_data = state_plot[state_plot['year'] <= intervention_year]
        ax.plot(pre_data['year'], pre_data['pred_actual'], 
               color='green', linewidth=2, linestyle='-', label='Pre-intervention fit')
        
        # Counterfactual (what would have happened without intervention)
        post_data = state_plot[state_plot['year'] >= intervention_year]
        ax.plot(post_data['year'], post_data['pred_counterfactual'], 
               color='red', linewidth=2, linestyle='--', label='Counterfactual')
        
        # Post-intervention fit with confidence intervals
        ax.plot(post_data['year'], post_data['pred_actual'], 
               color='green', linewidth=2, linestyle='-', label='Post-intervention fit')
        ax.fill_between(post_data['year'],
                       post_data['pred_actual_ci_lower'],
                       post_data['pred_actual_ci_upper'],
                       alpha=0.3, color='green', label='95% Confidence Interval')
        
        # Add intervention line
        ax.axvline(x=intervention_year, color='black', linestyle='--', 
                  alpha=0.7, label='Intervention')
        
        # Customize plot
        ax.set_xlabel('Year')
        ax.set_ylabel('Sales')
        ax.set_title(f'ITAS: State {state}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Create plots for each state
feature_cols = ['const', 'time', 'D', 'time_after', 'state']
plot_itas_by_state(cigar_pp, reg_res, feature_cols)

# %%
