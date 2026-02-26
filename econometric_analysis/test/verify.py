#!/usr/bin/env python
"""
Quick verification script to test the regression module.

This script runs basic checks to ensure the module works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

try:
    from regression import LinearRegressionModel, MarginalEffectsCalculator

    print("✓ Successfully imported modules")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic functionality of the regression module."""
    print("\nTesting basic functionality...")

    # Generate simple test data
    np.random.seed(42)
    n = 100

    X = np.column_stack(
        [
            np.linspace(1, 10, n),
            np.linspace(2, 20, n),
        ]
    )
    y = 5 + 2 * X[:, 0] + 1.5 * X[:, 1] + np.random.normal(0, 2, n)

    df = pd.DataFrame(
        {
            "y": y,
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )

    # Test fitting
    try:
        model = LinearRegressionModel()
        model.fit(df[["x1", "x2"]], df["y"])
        print("✓ Model fitting successful")
    except Exception as e:
        print(f"✗ Model fitting failed: {e}")
        return False

    # Test prediction
    try:
        preds = model.predict(df[["x1", "x2"]])
        assert len(preds) == len(df)
        print(f"✓ Predictions successful ({len(preds)} predictions)")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

    # Test R-squared
    try:
        r2 = model.r_squared()
        print(f"✓ R-squared calculation: {r2:.4f}")
    except Exception as e:
        print(f"✗ R-squared calculation failed: {e}")
        return False

    # Test marginal effects
    try:
        calc = MarginalEffectsCalculator(model)
        me_x1 = calc.marginal_effect("x1")
        me_x2 = calc.marginal_effect("x2")
        print(f"✓ Marginal effects calculated:")
        print(f"  - x1 marginal effect: {me_x1:.4f}")
        print(f"  - x2 marginal effect: {me_x2:.4f}")
    except Exception as e:
        print(f"✗ Marginal effects calculation failed: {e}")
        return False

    # Test elasticity
    try:
        x1_val = df["x1"].mean()
        y_val = df["y"].mean()
        elasticity = calc.elasticity("x1", x1_val, y_val)
        print(f"✓ Elasticity calculation: {elasticity:.6f}")
    except Exception as e:
        print(f"✗ Elasticity calculation failed: {e}")
        return False

    # Test saturation detection
    try:
        x_range = np.linspace(df["x1"].min(), df["x1"].max(), 50)
        sat_info = calc.detect_saturation_point("x1", x_range)
        print(f"✓ Saturation detection: {sat_info['saturation_type']}")
    except Exception as e:
        print(f"✗ Saturation detection failed: {e}")
        return False

    # Test responsiveness analysis
    try:
        resp = calc.responsiveness_analysis("x1", percentile_change=1.0)
        print(
            f"✓ Responsiveness analysis: 1% change leads to "
            f"{resp['percentage_y_effect']:.4f}% effect"
        )
    except Exception as e:
        print(f"✗ Responsiveness analysis failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Econometric Regression Module - Verification")
    print("=" * 60)

    success = test_basic_functionality()

    print("\n" + "=" * 60)
    if success:
        print("✓ All basic tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)
