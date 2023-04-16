# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.tree import SurvivalTree
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sklearn.tree import DecisionTreeClassifier
from sksurv.tree import SurvivalTree
from sksurv.tree._criterion import LogrankCriterion

from sksurv_optional import plot_tree

set_config(display="text")

# %%
X, y = load_gbsg2()

print(X) # features
print(y) # tuple of (True/False, time)

# %%
grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

X_no_grade = X.drop("tgrade", axis=1)
# not run for pandas 2.0.0
# https://github.com/YosefLab/Compass/issues/92
Xt = OneHotEncoder().fit_transform(X_no_grade)
Xt.loc[:, "tgrade"] = grade_num
# %%
random_state = 20
X_train, X_test, y_train, y_test = train_test_split(
    Xt, y, test_size=0.25, random_state=random_state
)

# %%
rsf = RandomSurvivalForest(
    n_estimators=1000,
    min_samples_split=10,
    min_samples_leaf=15,
    n_jobs=-1,
    random_state=random_state,
    
)

rsf.fit(X_train, y_train)
# %%
rsf.score(X_test, y_test)

# %%
X_test_sorted = X_test.sort_values(by=["pnodes", "age"])
X_test_sel = pd.concat((X_test_sorted.head(3), X_test_sorted.tail(3)))

X_test_sel

# %%
surv = rsf.predict_survival_function(X_test_sel, return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)

# %% 
# how to plot the path of random survival forest?
plot_tree(rsf, feature_names=X_train.columns.values, impurity=False, label="none")
# %%
