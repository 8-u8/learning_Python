"""
This module provides functions for regression analysis based on econometric approaches.

Functions provided:
- Estimate coefficients using OLS and other methods
- Calculate marginal effects of estimated regressors
- Find saturation points for covariates

Features:
- General Linear Model with OLS estimation
- Marginal effects calculation (partial derivatives)
- Saturation point detection and analysis
- Elasticity estimation for economic interpretation
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pygam import GAM, s
from scipy import optimize


class LinearRegressionModel:
    """
    A linear regression model using OLS estimation.

    This class provides functionality for fitting linear regression models,
    making predictions, and calculating basic statistics.

    Attributes:
        coef_ (np.ndarray): Estimated coefficients (excluding intercept).
        intercept_ (float): Estimated intercept.
        results_ (statsmodels.regression.linear_model.RegressionResults):
            Full regression results from statsmodels.
        X_train_ (pd.DataFrame): Training feature data.
        y_train_ (pd.Series): Training target data.
    """

    def __init__(self) -> None:
        """Initialize the linear regression model."""
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.results_: Optional[sm.regression.linear_model.RegressionResults] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self._feature_names: Optional[list] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "LinearRegressionModel":
        """
        Fit the linear regression model using OLS.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features (n_samples, n_features).
            y (Union[pd.Series, np.ndarray]): Target values (n_samples,).

        Returns:
            LinearRegressionModel: Fitted model instance.
        """
        # Convert to pandas if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self._feature_names = list(X.columns)

        # Add constant for OLS
        X_with_const = sm.add_constant(X)

        # Fit OLS model
        self.results_ = sm.OLS(y, X_with_const).fit()

        # Extract coefficients
        self.intercept_ = self.results_.params.iloc[0]
        self.coef_ = self.results_.params.iloc[1:].values

        return self

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]):
                Features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted before making predictions.")

        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_with_const = sm.add_constant(X, has_constant="add")
        return self.results_.predict(X_with_const).values

    def r_squared(self) -> float:
        """
        Get the R-squared value.

        Returns:
            float: R-squared value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return self.results_.rsquared

    def adjusted_r_squared(self) -> float:
        """
        Get the adjusted R-squared value.

        Returns:
            float: Adjusted R-squared value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return self.results_.rsquared_adj

    def summary(self) -> str:
        """
        Get a summary of the regression results.

        Returns:
            str: Summary statistics as a string.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return str(self.results_.summary())

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get coefficient estimates with standard errors and p-values.

        Returns:
            pd.DataFrame: Dataframe with coefficient, std error, t-stat, p-value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")

        coef_df = pd.DataFrame(
            {
                "Coefficient": self.results_.params,
                "Std Error": self.results_.bse,
                "t-statistic": self.results_.tvalues,
                "p-value": self.results_.pvalues,
            }
        )
        return coef_df


class RegressionAnalyzer:
    """
    High-level analyzer combining regression modeling and effect calculations.

    This class provides a convenient interface for complete regression analysis
    including fitting, diagnostics, and effect calculations.
    """

    def __init__(self) -> None:
        """Initialize the regression analyzer."""
        self.model: Optional[LinearRegressionModel] = None
        self.calculator: Optional[MarginalEffectsCalculator] = None

    def fit_and_analyze(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "RegressionAnalyzer":
        """
        Fit the model and initialize the calculator.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features.
            y (Union[pd.Series, np.ndarray]): Target values.

        Returns:
            RegressionAnalyzer: Self for method chaining.
        """
        self.model = LinearRegressionModel()
        self.model.fit(X, y)
        self.calculator = MarginalEffectsCalculator(self.model)
        return self

    def get_summary(self) -> str:
        """
        Get a comprehensive summary of the analysis.

        Returns:
            str: Summary statistics and interpretation.
        """
        if self.model is None:
            raise ValueError("Model must be fitted first.")
        return self.model.summary()

    def analyze_variable(self, variable_name: str) -> Dict:
        """
        Perform comprehensive analysis for a specific variable.

        Args:
            variable_name (str): Name of the variable to analyze.

        Returns:
            Dict: Comprehensive analysis results.
        """
        if self.calculator is None:
            raise ValueError("Model must be fitted first.")

        if self.model.X_train_ is None:
            raise ValueError("Training data not available.")

        x_range = np.linspace(
            self.model.X_train_[variable_name].min(),
            self.model.X_train_[variable_name].max(),
            50,
        )

        return {
            "marginal_effect": self.calculator.marginal_effect(variable_name),
            "saturation": self.calculator.detect_saturation_point(
                variable_name, x_range
            ),
            "responsiveness": self.calculator.responsiveness_analysis(variable_name),
        }


class SemiLogRegressionModel:
    """
    Semi-log regression model with log-transformed explanatory variables.

    This model implements regression where explanatory variables are
    log-transformed (ln). The typical form is:

        y = β₀ + β₁*ln(x₁) + β₂*ln(x₂) + ... + ε

    For this specification, the coefficient β_j represents the change in y
    for a 1% change in x_j (semi-elasticity).

    Attributes:
        coef_ (np.ndarray): Estimated coefficients (excluding intercept).
        intercept_ (float): Estimated intercept.
        results_ (statsmodels.regression.linear_model.RegressionResults):
            Full regression results from statsmodels.
        X_train_ (pd.DataFrame): Training feature data (original scale, not logged).
        y_train_ (pd.Series): Training target data.
    """

    def __init__(self) -> None:
        """Initialize the semi-log regression model."""
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.results_: Optional[sm.regression.linear_model.RegressionResults] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self._feature_names: Optional[list] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "SemiLogRegressionModel":
        """
        Fit the semi-log regression model using OLS on log-transformed variables.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features (n_samples, n_features).
                Values must be positive for log transformation.
            y (Union[pd.Series, np.ndarray]): Target values (n_samples,).

        Returns:
            SemiLogRegressionModel: Fitted model instance.

        Raises:
            ValueError: If any X values are non-positive.
        """
        # Convert to pandas if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Store original data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self._feature_names = list(X.columns)

        # Check for non-positive values
        if (X <= 0).any().any():
            raise ValueError(
                "All X values must be positive for log transformation.")

        # Log-transform the features
        X_logged = np.log(X)

        # Add constant for OLS
        X_with_const = sm.add_constant(X_logged)

        # Fit OLS model
        self.results_ = sm.OLS(y, X_with_const).fit()

        # Extract coefficients
        self.intercept_ = self.results_.params.iloc[0]
        self.coef_ = self.results_.params.iloc[1:].values

        return self

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted semi-log model.

        Args:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]):
                Features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted before making predictions.")

        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Log-transform features
        X_logged = np.log(X)
        X_with_const = sm.add_constant(X_logged, has_constant="add")
        return self.results_.predict(X_with_const).values

    def r_squared(self) -> float:
        """
        Get the R-squared value.

        Returns:
            float: R-squared value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return self.results_.rsquared

    def adjusted_r_squared(self) -> float:
        """
        Get the adjusted R-squared value.

        Returns:
            float: Adjusted R-squared value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return self.results_.rsquared_adj

    def summary(self) -> str:
        """
        Get a summary of the regression results.

        Returns:
            str: Summary statistics as a string.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return str(self.results_.summary())

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get coefficient estimates with standard errors and p-values.

        Returns:
            pd.DataFrame: Dataframe with coefficient, std error, t-stat, p-value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")

        coef_df = pd.DataFrame(
            {
                "Coefficient": self.results_.params,
                "Std Error": self.results_.bse,
                "t-statistic": self.results_.tvalues,
                "p-value": self.results_.pvalues,
            }
        )
        return coef_df


class DoubLogRegressionModel:
    """
    Double-log (power law) regression model.

    This model implements regression where both the target and explanatory
    variables are log-transformed. The typical form is:

        ln(y) = β₀ + β₁*ln(x₁) + β₂*ln(x₂) + ... + ε

    Which expands to the power law form:

        y = exp(β₀) * x₁^β₁ * x₂^β₂ * ...

    For a single variable: y = a * x^β

    Key properties:
    - β is the elasticity (constant across the range)
    - The marginal effect dy/dx = β * a * x^(β-1) is non-linear
    - When 0 < β < 1, diminishing returns occur
    - Saturation point can be detected when ME approaches zero

    Attributes:
        coef_ (np.ndarray): Estimated log-coefficients (excluding intercept).
        intercept_ (float): Estimated log-intercept.
        results_ (statsmodels.regression.linear_model.RegressionResults):
            Full regression results from statsmodels.
        X_train_ (pd.DataFrame): Training feature data (original scale).
        y_train_ (pd.Series): Training target data (original scale).
    """

    def __init__(self) -> None:
        """Initialize the double-log regression model."""
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.results_: Optional[sm.regression.linear_model.RegressionResults] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self._feature_names: Optional[list] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "DoubLogRegressionModel":
        """
        Fit the double-log regression model using OLS on log-transformed variables.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features (n_samples, n_features).
                Values must be positive for log transformation.
            y (Union[pd.Series, np.ndarray]): Target values (n_samples,).
                Values must be positive for log transformation.

        Returns:
            DoubLogRegressionModel: Fitted model instance.

        Raises:
            ValueError: If any X or y values are non-positive.
        """
        # Convert to pandas if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Store original data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self._feature_names = list(X.columns)

        # Check for non-positive values
        if (X <= 0).any().any():
            raise ValueError(
                "All X values must be positive for log transformation.")
        if (y <= 0).any():
            raise ValueError(
                "All y values must be positive for log transformation.")

        # Log-transform both X and y
        X_logged = np.log1p(X)
        y_logged = np.log1p(y)

        # Add constant for OLS
        X_with_const = sm.add_constant(X_logged)

        # Fit OLS model
        self.results_ = sm.OLS(y_logged, X_with_const).fit()

        # Extract coefficients
        self.intercept_ = self.results_.params.iloc[0]
        self.coef_ = self.results_.params.iloc[1:].values

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted double-log model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted before making predictions.")

        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Log-transform features
        X_logged = np.log(X)
        # Convert to ndarray for statsmodels compatibility
        X_logged_array = np.asarray(X_logged, dtype=float)

        # Ensure 2D shape (single column DataFrames become 1D arrays)
        if X_logged_array.ndim == 1:
            X_logged_array = X_logged_array.reshape(-1, 1)

        X_with_const = sm.add_constant(X_logged_array, has_constant="add")

        # Predict log(y), then exponentiate to get y
        log_predictions = np.asarray(
            self.results_.predict(X_with_const), dtype=float)
        return np.exp(log_predictions)

    def r_squared(self) -> float:
        """
        Get the R-squared value for the log-log model.

        Returns:
            float: R-squared value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return self.results_.rsquared

    def adjusted_r_squared(self) -> float:
        """
        Get the adjusted R-squared value.

        Returns:
            float: Adjusted R-squared value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return self.results_.rsquared_adj

    def summary(self) -> str:
        """
        Get a summary of the regression results.

        Returns:
            str: Summary statistics as a string.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")
        return str(self.results_.summary())

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get coefficient estimates with standard errors and p-values.

        Returns:
            pd.DataFrame: Dataframe with coefficient, std error, t-stat, p-value.
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first.")

        coef_df = pd.DataFrame(
            {
                "Coefficient": self.results_.params,
                "Std Error": self.results_.bse,
                "t-statistic": self.results_.tvalues,
                "p-value": self.results_.pvalues,
            }
        )
        return coef_df


class GAMRegressionModel:
    """
    Generalized Additive Model (GAM) for flexible nonlinear regression.

    This model extends linear regression by allowing each feature to have
    a smooth, nonlinear relationship with the target variable:

        y = β₀ + f₁(x₁) + f₂(x₂) + ... + ε

    where f_j are smooth functions estimated using splines.

    Attributes:
        model_ (GAM): Fitted pygam model instance.
        X_train_ (pd.DataFrame): Training feature data.
        y_train_ (pd.Series): Training target data.
        _feature_names (list): Names of features.
    """

    def __init__(self, lam: float = 0.6) -> None:
        """
        Initialize the GAM model.

        Args:
            lam (float): Regularization parameter (0 to 1). Lower values lead to
                more wiggly fits. Default is 0.6.
        """
        self.model_: Optional[GAM] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self._feature_names: Optional[list] = None
        self.lam = lam

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "GAMRegressionModel":
        """
        Fit the GAM model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features (n_samples, n_features).
            y (Union[pd.Series, np.ndarray]): Target values (n_samples,).

        Returns:
            GAMRegressionModel: Fitted model instance.
        """
        # Convert to pandas if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self._feature_names = list(X.columns)

        # Create GAM formula: apply smooth splines to each feature
        # Start with the first spline and add the rest to avoid sum(0, ...) issue
        n_features = X.shape[1]
        if n_features == 1:
            gam_formula = s(0, lam=self.lam)
        else:
            gam_formula = s(0, lam=self.lam)
            for i in range(1, n_features):
                gam_formula = gam_formula + s(i, lam=self.lam)

        # Fit GAM
        self.model_ = GAM(gam_formula)
        self.model_.fit(X.values, y.values)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted GAM model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions.")

        if isinstance(X, np.ndarray):
            X_arr = X
        else:
            X_arr = X.values

        return self.model_.predict(X_arr)

    def r_squared(self) -> float:
        """
        Get the pseudo R-squared value (explained deviance).

        Returns:
            float: R-squared value.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first.")

        # Calculate R² = 1 - (SS_res / SS_tot)
        y_pred = self.predict(self.X_train_)
        ss_res = np.sum((self.y_train_.values - y_pred) ** 2)
        ss_tot = np.sum(
            (self.y_train_.values - self.y_train_.values.mean()) ** 2)

        if ss_tot == 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)

    def summary(self) -> str:
        """
        Get a summary of the GAM model.

        Returns:
            str: Summary information as a string.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first.")

        summary_text = f"GAM Model Summary\n"
        summary_text += f"================\n"
        summary_text += f"R² (pseudo): {self.r_squared():.6f}\n"
        summary_text += f"Number of features: {len(self._feature_names)}\n"
        summary_text += f"Features: {', '.join(self._feature_names)}\n"
        summary_text += f"Regularization parameter (λ): {self.lam}\n"

        return summary_text

    def get_partial_dependence(
        self, feature_idx: int, x_range: np.ndarray
    ) -> np.ndarray:
        """
        Get partial dependence values for a specific feature.

        Args:
            feature_idx (int): Index of the feature.
            x_range (np.ndarray): Values at which to evaluate the partial dependence.

        Returns:
            np.ndarray: Partial dependence values.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first.")

        return self.model_.partial_dependence(
            term=feature_idx, X=self.X_train_.values, width=x_range
        )


class MarginalEffectsCalculator:
    """
    Calculator for marginal effects and elasticity from fitted regression models.

    This class provides methods to calculate marginal effects (partial derivatives),
    saturation points, and elasticity measures for econometric interpretation.

    Supports multiple model types:
    - LinearRegressionModel: Linear OLS regression
    - SemiLogRegressionModel: Semi-log regression with log-transformed features
    - GAMRegressionModel: Generalized Additive Models with nonlinear relationships

    Attributes:
        model: Fitted regression model instance (flexible type).
    """

    def __init__(
        self,
        model: Union[
            "LinearRegressionModel",
            "SemiLogRegressionModel",
            "DoubLogRegressionModel",
            "GAMRegressionModel",
        ],
    ) -> None:
        """
        Initialize the marginal effects calculator.

        Args:
            model: A fitted regression model (LinearRegressionModel,
                SemiLogRegressionModel, DoubLogRegressionModel, or GAMRegressionModel).

        Raises:
            ValueError: If model is not properly fitted.
        """
        # Check if model is fitted
        if isinstance(
            model,
            (LinearRegressionModel, SemiLogRegressionModel, DoubLogRegressionModel),
        ):
            if model.results_ is None:
                raise ValueError(
                    "Model must be fitted before calculating marginal effects."
                )
        elif isinstance(model, GAMRegressionModel):
            if model.model_ is None:
                raise ValueError("GAM model must be fitted first.")
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        self.model = model
        self._model_type = type(model).__name__

    def marginal_effect(
        self,
        variable_name: str,
        mean_value: Optional[float] = None,
    ) -> float:
        """
        Calculate the marginal effect of a variable.

        Behavior depends on model type:
        - LinearRegressionModel: Returns constant coefficient
        - SemiLogRegressionModel: Returns semi-elasticity (effect for 1% change)
        - GAMRegressionModel: Returns marginal effect at specified mean_value

        Args:
            variable_name (str): Name of the variable.
            mean_value (Optional[float]): Value at which to evaluate marginal effect
                (required for GAM, optional for linear/semi-log).

        Returns:
            float: Marginal effect (coefficient value for linear/semi-log,
                or numerical derivative for GAM).

        Raises:
            ValueError: If variable name not found or mean_value missing for GAM.
        """
        if variable_name not in self.model._feature_names:
            raise ValueError(
                f"Variable '{variable_name}' not found in model features.")

        if self._model_type == "DoubLogRegressionModel":
            return self._marginal_effect_doublog(variable_name, mean_value)
        elif self._model_type == "GAMRegressionModel":
            return self._marginal_effect_gam(variable_name, mean_value)
        else:
            # LinearRegressionModel and SemiLogRegressionModel
            idx = self.model._feature_names.index(variable_name)
            return float(self.model.coef_[idx])

    def _marginal_effect_doublog(
        self,
        variable_name: str,
        mean_value: Optional[float] = None,
    ) -> float:
        """
        Calculate marginal effect for double-log (power law) model.

        For double-log: ln(y) = β₀ + β₁*ln(x)
        Which is: y = a * x^β₁

        Marginal effect: dy/dx = β₁ * a * x^(β₁-1) = β₁ * y / x

        At mean_value x₀:
            dy/dx = β₁ * y(x₀) / x₀

        Args:
            variable_name (str): Name of the variable.
            mean_value (Optional[float]): Point at which to evaluate ME.

        Returns:
            float: Marginal effect at mean_value.
        """
        if mean_value is None:
            mean_value = self.model.X_train_[variable_name].mean()

        idx = self.model._feature_names.index(variable_name)
        beta = self.model.coef_[idx]

        # For double-log model: ln(y) = β₀ + β₁*ln(x₁) + ... + β_k*ln(x_k)
        # Direct calculation: ME = β * y / x
        # We need to compute y at mean_value

        # y_pred = exp(β₀ + β₁*ln(mean_value) + other_terms_at_their_mean)
        # Use the mean values of training data for other variables
        log_prediction = self.model.intercept_

        for i, fname in enumerate(self.model._feature_names):
            if fname == variable_name:
                x_val = mean_value
            else:
                x_val = self.model.X_train_[fname].mean()

            log_prediction += self.model.coef_[i] * np.log(x_val)

        y_pred = np.exp(log_prediction)

        # ME = β * (y / x)
        me = beta * (y_pred / mean_value)
        return float(me)

    def _marginal_effect_gam(
        self,
        variable_name: str,
        mean_value: Optional[float] = None,
    ) -> float:
        """
        Calculate numerical marginal effect for GAM model.

        Computed as the finite difference derivative at mean_value.

        Args:
            variable_name (str): Name of the variable.
            mean_value (Optional[float]): Point at which to evaluate derivative.

        Returns:
            float: Estimated marginal effect.
        """
        if mean_value is None:
            mean_value = self.model.X_train_[variable_name].mean()

        feature_idx = self.model._feature_names.index(variable_name)
        epsilon = 1e-6

        # Create data points for numerical differentiation
        X_base = self.model.X_train_.iloc[:1].copy()
        X_plus = X_base.copy()
        X_minus = X_base.copy()

        X_plus.iloc[0, feature_idx] = mean_value + epsilon
        X_minus.iloc[0, feature_idx] = mean_value - epsilon

        # Evaluate predictions
        y_plus = self.model.predict(X_plus)[0]
        y_minus = self.model.predict(X_minus)[0]

        # Finite difference approximation
        me = (y_plus - y_minus) / (2 * epsilon)
        return float(me)

    def elasticity(
        self,
        variable_name: str,
        x_value: float,
        y_value: float,
    ) -> float:
        """
        Calculate the elasticity of a variable.

        Elasticity = (dY/dX) * (X/Y), measuring the percentage change in Y
        resulting from a 1% change in X.

        Args:
            variable_name (str): Name of the variable.
            x_value (float): Value of the independent variable.
            y_value (float): Value of the dependent variable (prediction or actual).

        Returns:
            float: Elasticity value.

        Raises:
            ValueError: If variable or y_value is invalid.
        """
        if y_value == 0:
            raise ValueError(
                "y_value cannot be zero for elasticity calculation.")

        me = self.marginal_effect(variable_name)
        return (me * x_value) / y_value

    def detect_saturation_point(
        self,
        variable_name: str,
        x_range: np.ndarray,
    ) -> Dict[str, Union[float, int, str, bool]]:
        """
        Detect the saturation point of a variable's effect.

        For linear/semi-log models: constant effect, no saturation.
        For double-log models: detects when ME becomes negligibly small.
        For GAM models: detects local maxima and diminishing returns.

        Args:
            variable_name (str): Name of the variable.
            x_range (np.ndarray): Range of values to evaluate.

        Returns:
            Dict[str, Union[float, int, str, bool]]: Dictionary containing saturation analysis.

        Raises:
            ValueError: If variable name not found.
        """
        if variable_name not in self.model._feature_names:
            raise ValueError(
                f"Variable '{variable_name}' not found in model features.")

        if self._model_type == "DoubLogRegressionModel":
            return self._detect_saturation_doublog(variable_name, x_range)
        else:
            # For Linear and SemiLog: constant effect
            me = self.marginal_effect(variable_name)

            return {
                "variable": variable_name,
                "marginal_effect": me,
                "x_min": float(x_range.min()),
                "x_max": float(x_range.max()),
                "saturation_type": "linear",
                "is_decreasing": me < 0,
                "interpretation": (
                    f"The model shows a constant marginal effect of {me:.4f}. "
                    "No saturation is detected."
                ),
            }

    def _detect_saturation_doublog(
        self,
        variable_name: str,
        x_range: np.ndarray,
    ) -> Dict[str, Union[float, int, str, bool]]:
        """
        Detect saturation point for double-log (power law) model.

        Saturation point is defined as where ME becomes negligibly small
        (< 1% of max ME in the range), indicating diminishing returns.

        Args:
            variable_name (str): Name of the variable.
            x_range (np.ndarray): Range of x values to evaluate.

        Returns:
            Dict with saturation analysis.
        """
        idx = self.model._feature_names.index(variable_name)
        beta = self.model.coef_[idx]

        # Calculate ME at each point in x_range
        # For double-log: ME(x) = β * y(x) / x
        me_values = []
        for x_val in x_range:
            me = self._marginal_effect_doublog(variable_name, mean_value=x_val)
            me_values.append(me)

        me_values = np.array(me_values)

        # Find saturation threshold where |ME| falls below 1% of max |ME|.
        # This makes the threshold scale to the estimated model instead of
        # defaulting to the range endpoint.
        abs_me_values = np.abs(me_values)
        max_abs_me = float(abs_me_values.max()) if len(
            abs_me_values) > 0 else 0.0

        if max_abs_me > 0:
            threshold = 0.01 * max_abs_me
            below_threshold = abs_me_values <= threshold

            saturation_threshold_x = None
            if below_threshold.any():
                saturation_idx = int(np.argmax(below_threshold))
                saturation_threshold_x = float(x_range[saturation_idx])
        else:
            threshold = 0.0
            saturation_threshold_x = None

        # Diminishing returns: check if β < 1 (in log-log space)
        diminishing_returns = beta < 1 and beta > 0

        return {
            "variable": variable_name,
            "beta_coefficient": float(beta),
            "x_min": float(x_range.min()),
            "x_max": float(x_range.max()),
            "saturation_type": "power_law",
            "diminishing_returns": diminishing_returns,
            "saturation_threshold_x": saturation_threshold_x,
            "max_abs_marginal_effect": max_abs_me,
            "threshold_definition": "1% of max |ME|",
            "interpretation": (
                f"Power law model with β={beta:.4f}. "
                f"{'Diminishing returns detected (β < 1). ' if diminishing_returns else ''}"
                f"Saturation point "
                f"{'at x≈' + f'{saturation_threshold_x:.2f}' if saturation_threshold_x else 'not reached in range'}."
            ),
        }

    def responsiveness_analysis(
        self,
        variable_name: str,
        percentile_change: float = 1.0,
    ) -> Dict[str, float]:
        """
        Analyze the responsiveness of the output to a change in input.

        Args:
            variable_name (str): Name of the variable.
            percentile_change (float): Percentage change in the variable.

        Returns:
            Dict[str, float]: Dictionary with responsiveness metrics.
                - 'percentage_change_x': The percentage change applied
                - 'absolute_effect': Absolute change in Y per unit change in X
                - 'percentage_effect': Percentage change in Y per unit change in X
        """
        me = self.marginal_effect(variable_name)

        # Get mean value from training data
        if self.model.X_train_ is None:
            raise ValueError("Training data not available in model.")

        x_mean = self.model.X_train_[variable_name].mean()
        y_mean = self.model.y_train_.mean()

        # Estimate impact of percentile change
        abs_change = (percentile_change / 100) * x_mean
        y_change = me * abs_change

        return {
            "variable": variable_name,
            "base_x_mean": x_mean,
            "base_y_mean": y_mean,
            "percentage_change_x": percentile_change,
            "absolute_x_change": abs_change,
            "absolute_y_effect": y_change,
            "percentage_y_effect": (y_change / y_mean) * 100,
        }
