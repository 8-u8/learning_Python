# %%
import pandas as pd
import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

## copied from: https://note.com/smith_217/n/n46e5c43af10b?sub_rt=share_h
# %% loading data
iris_df = sns.load_dataset("iris")
iris_df = iris_df[(iris_df['species']=='versicolor') | (iris_df['species']=='virginica')].reset_index(drop=True)

# %% model with default parameters
# 説明変数・目的変数の設定
X = iris_df.drop("species",axis=1)# 説明変数
y = iris_df['species'].map({'versicolor': 0, 'virginica': 1}) # versicolorをクラス0, virginicaをクラス1とする

# ロジスティック回帰モデルへのフィッティング
clf = LogisticRegression(random_state=0).fit(X, y)

# %% model via statsmodels
logit_model = sm.Logit(y, X)
logit_model.fit().params

# Optimization terminated successfully.
#          Current function value: 0.108399
#          Iterations 10
# sepal_length    -6.327719
# sepal_width     -6.618187
# petal_length     8.433801
# petal_width     10.282544
# dtype: float64

# %% model via sklearn with small penalty weights
clf02 = LogisticRegression(random_state=0, fit_intercept=False, C = 1e9).fit(X, y)
clf02.coef_
# almost equal to estimate via statsmodels
# > array([[-6.32771779, -6.61818115,  8.43379737, 10.28254174]])

# %% model via sklearn with none penalty 
clf03 = LogisticRegression(random_state=0, fit_intercept=False, penalty="none").fit(X, y)
clf03.coef_
# almost equal to estimate via statsmodels
# > array([[-6.32771795, -6.61818131,  8.4337976 , 10.28254195]])
