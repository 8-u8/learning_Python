# %% import libraries
import numpy as np
import pandas as pd

import statsmodels.api as sm

# %% load data
# using cigar data from statsmodels
cigar = sm.datasets.get_rdataset("Cigar", "Ecdat").data
# %% preprocessing

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # 古い時系列を削除、州を絞る
    condition = (df.loc[:,'year'] >= 70) & (df.loc[:,'state'].isin([3,9,10,22,21,23,31,33,48]))
    df_out = df.loc[condition,:].copy()

    # stateをカテゴリに変換
    df_out.loc[:,'area'] = df_out.loc[:,'state'].apply(lambda x: 'CA' if x == 5 else 'Rest of US')
    # yearの最小値を1として経過年数を作成する
    df_out.loc[:,'time'] = df_out.loc[:,'year'] - df_out.loc[:,'year'].min() + 1

    # 1980年を介入点として、  介入点Tとダミー変数Dを作成
    df_out.loc[:,'T'] = (df_out.loc[:,'year'] == 80).astype(int)
    df_out.loc[:,'D'] = (df_out.loc[:,'year'] >= 80).astype(int)

    # 介入後の経過年数を作成
    # 1980年以前は0、1980年は1、1981年は2、...
    df_out.loc[:,'time_after'] = df_out.loc[:,['state','D']].groupby('state').cumsum()['D']
    return df_out

cigar_pp = preprocess_data(cigar)
# %%
# Python (statsmodels: NW標準誤差調整)
# OLSモデルを使用し、標準誤差計算時に調整を行う
def OLS_wrapper(
        df: pd.DataFrame,
        y: str,
        time: str,
        D: str,
        cooperate: list[str],
) -> RegressionModel:
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
    RegressionModel
        statsmodelsの回帰モデルオブジェクト
    """
    # 切片項の追加
    df = sm.add_constant(df)
    

    # 説明変数の設定
    exog_vars = [time, D, 'time_after'] + cooperate
    X = df[exog_vars]
    X = sm.add_constant(X)

    # 目的変数の設定
    Y = df[y]

    # OLSモデルの適用
    model = sm.OLS(Y, X)
    
    return model
# データの準備 (切片項の追加)
# Y: 目的変数

# exog_vars: 説明変数 (time, D, time_after, X)
exog_vars = cigar_pp[]
Y_data = cigar_pp['Y']
X_with_const = sm.add_constant(exog_vars)

# OLSによる推定
ols_model = sm.OLS(Y_data, X_with_const)

# Newey-West (HAC: Heteroskedasticity and Autocorrelation Consistent) 標準誤差を適用 [11]
# 'maxlags'は考慮するラグの最大値を指定 (例: 季節性に応じて設定)
# NWはOLSの推定パラメータを維持しつつ、調整されたSEを出力する [10]
nw_results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': 6}) 
print(nw_results.summary())