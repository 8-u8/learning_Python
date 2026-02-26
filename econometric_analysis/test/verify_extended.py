#!/usr/bin/env python
"""
Verification script for extended regression models.

This script tests semi-log and GAM model functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

try:
    from regression import (
        GAMRegressionModel,
        LinearRegressionModel,
        MarginalEffectsCalculator,
        SemiLogRegressionModel,
    )

    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_semilog_model():
    """Test semi-log regression model."""
    print("\n" + "=" * 60)
    print("Testing Semi-Log Regression Model")
    print("=" * 60)

    np.random.seed(42)
    n = 100

    X = np.column_stack(
        [
            np.linspace(1, 50, n),
            np.linspace(2, 100, n),
        ]
    )
    y = 50 + 5 * np.log(X[:, 0]) + 2 * np.log(X[:, 1]) + np.random.normal(0, 5, n)

    df = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})

    try:
        model = SemiLogRegressionModel()
        model.fit(df[["x1", "x2"]], df["y"])
        print("✓ Semi-log model fitting successful")

        preds = model.predict(df[["x1", "x2"]])
        print(f"✓ Predictions generated ({len(preds)} samples)")

        r2 = model.r_squared()
        print(f"✓ R²: {r2:.4f}")

        # Marginal effects
        calc = MarginalEffectsCalculator(model)
        me_x1 = calc.marginal_effect("x1")
        me_x2 = calc.marginal_effect("x2")
        print(f"✓ Marginal effects calculated:")
        print(f"  - x1: {me_x1:.4f} (semi-elasticity)")
        print(f"  - x2: {me_x2:.4f} (semi-elasticity)")

        # Elasticity
        elasticity = calc.elasticity("x1", df["x1"].mean(), df["y"].mean())
        print(f"✓ Elasticity calculated: {elasticity:.6f}")

        return True

    except Exception as e:
        print(f"✗ Semi-log test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gam_model():
    """Test GAM model."""
    print("\n" + "=" * 60)
    print("Testing GAM (Generalized Additive Model)")
    print("=" * 60)

    np.random.seed(123)
    n = 200

    X = np.column_stack(
        [
            np.linspace(0, 10, n),
            np.linspace(0, 10, n),
        ]
    )
    # Nonlinear relationship
    y = 50 + np.sin(X[:, 0]) + np.exp(-0.1 * X[:, 1]) + np.random.normal(0, 1, n)

    df = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})

    try:
        model = GAMRegressionModel()
        model.fit(df[["x1", "x2"]], df["y"])
        print("✓ GAM model fitting successful")

        preds = model.predict(df[["x1", "x2"]])
        print(f"✓ Predictions generated ({len(preds)} samples)")

        r2 = model.r_squared()
        print(f"✓ R²: {r2:.4f}")

        # Marginal effects at mean values
        calc = MarginalEffectsCalculator(model)
        me_x1 = calc.marginal_effect("x1")
        me_x2 = calc.marginal_effect("x2")
        print(f"✓ Marginal effects calculated (at mean):")
        print(f"  - x1: {me_x1:.6f}")
        print(f"  - x2: {me_x2:.6f}")

        # Marginal effects at specific points
        me_x1_at_5 = calc.marginal_effect("x1", mean_value=5.0)
        print(f"✓ Marginal effect at specific point:")
        print(f"  - x1 at x=5: {me_x1_at_5:.6f}")

        # Elasticity
        elasticity = calc.elasticity("x1", df["x1"].mean(), df["y"].mean())
        print(f"✓ Elasticity calculated: {elasticity:.6f}")

        return True

    except Exception as e:
        print(f"✗ GAM test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_comparison():
    """Test comparing different model types."""
    print("\n" + "=" * 60)
    print("Testing Model Comparison")
    print("=" * 60)

    np.random.seed(789)
    n = 150

    x = np.linspace(1, 100, n)
    y = 100 + 20 * np.log(x) + np.random.normal(0, 20, n)

    df = pd.DataFrame({"y": y, "x": x})

    try:
        # Linear model
        linear = LinearRegressionModel()
        linear.fit(df[["x"]], df["y"])
        print(f"✓ Linear Model R²: {linear.r_squared():.6f}")

        # Semi-log model
        semilog = SemiLogRegressionModel()
        semilog.fit(df[["x"]], df["y"])
        print(f"✓ Semi-Log Model R²: {semilog.r_squared():.6f}")

        # GAM model
        gam = GAMRegressionModel()
        gam.fit(df[["x"]], df["y"])
        print(f"✓ GAM Model R²: {gam.r_squared():.6f}")

        print(f"\n✓ Model Comparison:")
        print(f"  Semi-Log model should fit best (true relationship is log)")

        return True

    except Exception as e:
        print(f"✗ Model comparison test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Extended Regression Models - Verification")
    print("=" * 60)

    results = [
        test_semilog_model(),
        test_gam_model(),
        test_model_comparison(),
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All extended model tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)
