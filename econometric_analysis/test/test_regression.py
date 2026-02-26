import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regression import (
    GAMRegressionModel,
    LinearRegressionModel,
    MarginalEffectsCalculator,
    SemiLogRegressionModel,
)


class TestLinearRegressionModel:
    """Test linear regression model estimation and basic properties."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple linear data for testing."""
        np.random.seed(42)
        n = 100
        X = np.column_stack(
            [
                np.ones(n),  # intercept
                np.linspace(1, 10, n),  # x1
                np.linspace(2, 20, n),  # x2
            ]
        )
        # True coefficients: [5, 2, 1.5]
        true_coef = np.array([5, 2, 1.5])
        y = X @ true_coef + np.random.normal(0, 1, n)

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "x2": X[:, 2],
            }
        )
        return df

    def test_model_initialization(self, simple_data):
        """Test model initialization."""
        model = LinearRegressionModel()
        assert model is not None
        assert not hasattr(model, "results_")

    def test_model_fit(self, simple_data):
        """Test model fitting."""
        model = LinearRegressionModel()
        model.fit(simple_data[["x1", "x2"]], simple_data["y"])

        assert hasattr(model, "results_")
        assert model.results_ is not None
        assert len(model.coef_) == 2

    def test_model_predict(self, simple_data):
        """Test model prediction."""
        model = LinearRegressionModel()
        model.fit(simple_data[["x1", "x2"]], simple_data["y"])

        predictions = model.predict(simple_data[["x1", "x2"]])
        assert len(predictions) == len(simple_data)
        assert predictions.dtype == np.float64

    def test_model_r_squared(self, simple_data):
        """Test R-squared calculation."""
        model = LinearRegressionModel()
        model.fit(simple_data[["x1", "x2"]], simple_data["y"])

        r_squared = model.r_squared()
        assert 0 <= r_squared <= 1

    def test_model_summary(self, simple_data):
        """Test summary statistics."""
        model = LinearRegressionModel()
        model.fit(simple_data[["x1", "x2"]], simple_data["y"])

        summary = model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestMarginalEffectsCalculator:
    """Test marginal effects calculations."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        np.random.seed(42)
        n = 100
        X = np.linspace(1, 10, n)
        y = 5 + 2 * X + 1.5 * (X**2) + np.random.normal(0, 1, n)

        df = pd.DataFrame(
            {
                "y": y,
                "x": X,
            }
        )

        model = LinearRegressionModel()
        model.fit(df[["x"]], df["y"])
        return model, df

    def test_marginal_effect_linear(self):
        """Test marginal effect for linear term."""
        np.random.seed(42)
        n = 100
        X = np.linspace(1, 10, n)
        y = 5 + 2 * X + np.random.normal(0, 0.5, n)

        df = pd.DataFrame({"y": y, "x": X})

        model = LinearRegressionModel()
        model.fit(df[["x"]], df["y"])

        calculator = MarginalEffectsCalculator(model)

        # For linear model, marginal effect should be constant (the coefficient)
        me = calculator.marginal_effect("x", mean_value=5)
        assert abs(me - model.coef_[0]) < 0.1

    def test_saturation_point_detection(self, fitted_model):
        """Test saturation point detection."""
        model, df = fitted_model
        calculator = MarginalEffectsCalculator(model)

        # Test if saturation point detection returns valid range
        x_range = np.linspace(df["x"].min(), df["x"].max(), 50)
        saturation_info = calculator.detect_saturation_point("x", x_range)

        assert saturation_info is not None
        assert isinstance(saturation_info, dict)

    def test_elasticity_calculation(self, fitted_model):
        """Test elasticity calculation."""
        model, df = fitted_model
        calculator = MarginalEffectsCalculator(model)

        # Elasticity = (dY/dX) * (X/Y)
        x_val = 5.0
        y_val = df.iloc[(df["x"] - x_val).abs().argmin()]["y"]

        elasticity = calculator.elasticity("x", x_value=x_val, y_value=y_val)
        assert isinstance(elasticity, (int, float, np.number))


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_complete_workflow(self):
        """Test complete workflow from data to insights."""
        # Generate realistic data
        np.random.seed(42)
        n = 200
        X = np.linspace(0.1, 10, n)
        # Quadratic relationship
        y = 10 + 3 * X - 0.15 * (X**2) + np.random.normal(0, 2, n)

        df = pd.DataFrame(
            {
                "y": y,
                "advertising": X,
            }
        )

        # Fit model
        model = LinearRegressionModel()
        model.fit(df[["advertising"]], df["y"])

        # Calculate marginal effects
        calculator = MarginalEffectsCalculator(model)

        # Get summary
        summary = model.summary()
        assert len(summary) > 0

        # Check model properties
        assert model.r_squared() > 0


class TestSemiLogRegressionModel:
    """Test semi-log regression model (log-transformed explanatory variables)."""

    @pytest.fixture
    def log_data(self):
        """Generate data suitable for semi-log regression."""
        np.random.seed(42)
        n = 150

        # Explanatory variables (positive values for log transformation)
        x1 = np.linspace(1, 50, n)
        x2 = np.linspace(2, 100, n)

        # Target: y = 100 + 5*ln(x1) + 2*ln(x2) + noise
        y = 100 + 5 * np.log(x1) + 2 * np.log(x2) + np.random.normal(0, 10, n)

        df = pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        )
        return df

    def test_semilog_initialization(self):
        """Test semi-log model initialization."""
        model = SemiLogRegressionModel()
        assert model is not None

    def test_semilog_fit(self, log_data):
        """Test semi-log model fitting."""
        model = SemiLogRegressionModel()
        model.fit(log_data[["x1", "x2"]], log_data["y"])

        assert model.results_ is not None
        assert len(model.coef_) == 2

    def test_semilog_predict(self, log_data):
        """Test semi-log model prediction."""
        model = SemiLogRegressionModel()
        model.fit(log_data[["x1", "x2"]], log_data["y"])

        predictions = model.predict(log_data[["x1", "x2"]])
        assert len(predictions) == len(log_data)
        assert predictions.dtype == np.float64

    def test_semilog_elasticity(self, log_data):
        """Test elasticity calculation for semi-log model."""
        model = SemiLogRegressionModel()
        model.fit(log_data[["x1", "x2"]], log_data["y"])

        calculator = MarginalEffectsCalculator(model)

        # For semi-log model, elasticity = coefficient / variable value
        x1_val = log_data["x1"].mean()
        y_val = log_data["y"].mean()

        elasticity = calculator.elasticity("x1", x1_val, y_val)
        assert isinstance(elasticity, (int, float, np.number))


class TestGAMRegressionModel:
    """Test Generalized Additive Model (GAM)."""

    @pytest.fixture
    def nonlinear_data(self):
        """Generate nonlinear data suitable for GAM."""
        np.random.seed(42)
        n = 200

        # Nonlinear relationship
        x1 = np.linspace(0, 10, n)
        x2 = np.linspace(0, 10, n)

        # y = 50 + sin(x1) + exp(-0.1*x2) + noise
        y = 50 + np.sin(x1) + np.exp(-0.1 * x2) + np.random.normal(0, 0.5, n)

        df = pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        )
        return df

    def test_gam_initialization(self):
        """Test GAM initialization."""
        model = GAMRegressionModel()
        assert model is not None

    def test_gam_fit(self, nonlinear_data):
        """Test GAM fitting."""
        model = GAMRegressionModel()
        model.fit(nonlinear_data[["x1", "x2"]], nonlinear_data["y"])

        assert model.model_ is not None
        assert len(model.X_train_) == len(nonlinear_data)

    def test_gam_predict(self, nonlinear_data):
        """Test GAM prediction."""
        model = GAMRegressionModel()
        model.fit(nonlinear_data[["x1", "x2"]], nonlinear_data["y"])

        predictions = model.predict(nonlinear_data[["x1", "x2"]])
        assert len(predictions) == len(nonlinear_data)
        assert predictions.dtype == np.float64

    def test_gam_r_squared(self, nonlinear_data):
        """Test GAM R-squared calculation."""
        model = GAMRegressionModel()
        model.fit(nonlinear_data[["x1", "x2"]], nonlinear_data["y"])

        r_squared = model.r_squared()
        assert 0 <= r_squared <= 1

    def test_gam_marginal_effects(self, nonlinear_data):
        """Test marginal effects calculation for GAM."""
        model = GAMRegressionModel()
        model.fit(nonlinear_data[["x1", "x2"]], nonlinear_data["y"])

        calculator = MarginalEffectsCalculator(model)

        # For GAM, marginal effects are evaluated at specific points
        x1_val = nonlinear_data["x1"].mean()
        me = calculator.marginal_effect("x1", mean_value=x1_val)

        assert isinstance(me, (int, float, np.number))

    def test_gam_elasticity(self, nonlinear_data):
        """Test elasticity calculation for GAM at mean values."""
        model = GAMRegressionModel()
        model.fit(nonlinear_data[["x1", "x2"]], nonlinear_data["y"])

        calculator = MarginalEffectsCalculator(model)

        x1_val = nonlinear_data["x1"].mean()
        y_val = nonlinear_data["y"].mean()

        elasticity = calculator.elasticity("x1", x1_val, y_val)
        assert isinstance(elasticity, (int, float, np.number))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
