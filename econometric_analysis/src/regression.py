import numpy as np 
import pandas as pd
import statsmodels.api as sm

'''
This module provides functions for regression analysis based on econometric approaches.

We provide functions for estimate coefficients, calculate marginal effects of estimated regressors.

Also, we provide simulation functions for finding threshold of covariates like marginal utility, ROAS, etc.

- Estimate coefficients using several methods
  - Ordinary Least Squares (OLS)
  - Weighted Least Squares (WLS)
  - Generalized Least Squares (GLS)
  - Maximum Likelihood Estimation (MLE)
  
- Calculate marginal effects of estimated regressors
  - Partial derivatives of the estimated coefficients with respect to the regressors
  
- Simulation functions for finding threshold of covariates
  - marginal utility
  - ROAS
'''