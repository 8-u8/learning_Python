# %%
import pandas as pd
import numpy as np

# import sample dataset
import seaborn as sns

# import modeling modules
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

## copied from: https://note.com/smith_217/n/n46e5c43af10b?sub_rt=share_h
# %% loading data
iris_df = sns.load_dataset("iris")
iris_df = iris_df[(iris_df['species']=='versicolor') 
                  | (iris_df['species']=='virginica')].reset_index(drop=True)

# %% utility function to comparison.
def result_to_df(features, params):
    out_dict = {
        'features': features,
        'params':params
    }

    out = pd.DataFrame(out_dict)
    
    return out

# %% settings target and features
# features
X = iris_df.drop("species",axis=1)
# versicolor convert from category to binary values.
y = iris_df['species'].map({'versicolor': 0, 'virginica': 1}) 

# logistic regression
# assumption: no intercept.
# %% baseline: logistic regression using statsmodels
clf_statsmodels = sm.Logit(y, X)
clf_statsmodels = clf_statsmodels.fit()
statsmodels_params = result_to_df(features=clf_statsmodels.params.index.values,
                                  params=clf_statsmodels.params.values)
statsmodels_params

#      features     params
# 0	sepal_length	-6.327719
# 1	sepal_width	    -6.618187
# 2	petal_length     8.433801
# 3	petal_width 	10.282544

#%% logistic regression using sklearn
# Logistic regression from sklearn is L2 regularization by default.
clf_sklearn_default = LogisticRegression(random_state=0, fit_intercept=False)
clf_sklearn_default.fit(X, y)
sklearn_default_params = result_to_df(features=X.columns.values,
                                     params=clf_sklearn_default.coef_[0])
sklearn_default_params
# There are difference from statsmodels result.
#   features	     params
# 0	sepal_length	-1.863342
# 1	sepal_width 	-1.646021
# 2	petal_length     2.477247
# 3	petal_width	     2.593144

# %% modification 1: using small penalty weights
clf_sklearn_low_penalty = LogisticRegression(
                            random_state=0,
                            fit_intercept=False,
                            C = 1e9)
clf_sklearn_low_penalty.fit(X, y)
sklearn_low_penalty_params = result_to_df(features=X.columns.values,
                                          params=clf_sklearn_low_penalty.coef_[0])
sklearn_low_penalty_params
# almost equal to estimate via statsmodels

#    features   	params
# 0	sepal_length	-6.327718
# 1	sepal_width 	-6.618181
# 2	petal_length     8.433797
# 3	petal_width 	10.282542

# %% modification 2: set penalty as None.
clf_sklearn_None_penalty = LogisticRegression(
                              random_state=0,
                              fit_intercept=False,
                              penalty=None)

clf_sklearn_None_penalty.fit(X, y)
sklearn_None_penalty_params = result_to_df(features=X.columns.values,
                                          params=clf_sklearn_None_penalty.coef_[0])
sklearn_None_penalty_params
# almost equal to estimate via statsmodels
# 	features	    params
# 0	sepal_length	-6.327718
# 1	sepal_width 	-6.618181
# 2	petal_length     8.433798
# 3	petal_width	    10.282542


# %%
