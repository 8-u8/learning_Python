"""
Example usage of the econometric regression analysis module.

This script demonstrates how to use the linear regression model,
calculate marginal effects, and detect saturation points.
"""

# Add src to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from regression import (
    LinearRegressionModel,
    MarginalEffectsCalculator,
    RegressionAnalyzer,
)


def example_1_simple_linear_regression() -> None:
    """
    Example 1: Simple linear regression with marginal effects.

    This example demonstrates a basic linear regression with
    two explanatory variables and calculates marginal effects.
    """
    print("=" * 70)
    print("Example 1: Simple Linear Regression with Marginal Effects")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n = 150

    # Create features
    advertising = np.linspace(1, 50, n)
    price = np.linspace(10, 100, n)

    # True relationship: Sales = 100 + 3*Advertising - 0.5*Price + noise
    sales = 100 + 3 * advertising - 0.5 * price + np.random.normal(0, 10, n)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Sales": sales,
            "Advertising": advertising,
            "Price": price,
        }
    )

    print("\nDataset Summary:")
    print(df.describe())

    # Fit model
    model = LinearRegressionModel()
    model.fit(df[["Advertising", "Price"]], df["Sales"])

    print("\nModel Summary:")
    print(model.summary())

    # Calculate marginal effects
    calculator = MarginalEffectsCalculator(model)

    print("\n" + "-" * 70)
    print("Marginal Effects Analysis:")
    print("-" * 70)

    for var in ["Advertising", "Price"]:
        me = calculator.marginal_effect(var)
        print(f"\n{var}:")
        print(f"  Marginal Effect: {me:.4f}")
        print(
            f"  Interpretation: A 1 unit increase in {var} "
            f"leads to a {me:.4f} unit change in Sales"
        )

    # Elasticity at mean values
    print("\n" + "-" * 70)
    print("Elasticity Analysis (at mean values):")
    print("-" * 70)

    x_mean_adv = df["Advertising"].mean()
    y_pred = model.predict(df[["Advertising", "Price"]]).mean()

    elasticity_adv = calculator.elasticity("Advertising", x_mean_adv, y_pred)
    print(f"\nAdvertising Elasticity: {elasticity_adv:.4f}")
    print(
        f"Interpretation: A 1% increase in Advertising leads to a "
        f"{elasticity_adv:.4f}% change in Sales"
    )


def example_2_comprehensive_analysis() -> None:
    """
    Example 2: Comprehensive analysis with saturation detection.

    This example demonstrates a more complex analysis including
    saturation point detection and responsiveness analysis.
    """
    print("\n\n" + "=" * 70)
    print("Example 2: Comprehensive Analysis with Saturation Detection")
    print("=" * 70)

    # Generate data with potential saturation-like pattern
    np.random.seed(123)
    n = 200

    # Marketing spend (feature)
    marketing = np.linspace(0.5, 100, n)

    # Revenue (target) - quadratic-like with noise
    revenue = 500 + 4 * marketing - 0.01 * (marketing**2) + np.random.normal(0, 50, n)

    df = pd.DataFrame(
        {
            "Revenue": revenue,
            "Marketing": marketing,
        }
    )

    # Fit model
    analyzer = RegressionAnalyzer()
    analyzer.fit_and_analyze(df[["Marketing"]], df["Revenue"])

    # Analyze Marketing variable
    analysis = analyzer.analyze_variable("Marketing")

    print("\nVariable Analysis for 'Marketing':")
    print("-" * 70)
    print(f"\nMarginal Effect: {analysis['marginal_effect']:.6f}")

    print("\nSaturation Analysis:")
    saturation = analysis["saturation"]
    for key, value in saturation.items():
        print(f"  {key}: {value}")

    print("\nResponsiveness Analysis:")
    responsiveness = analysis["responsiveness"]
    for key, value in responsiveness.items():
        print(
            f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}"
        )


def example_3_coefficient_comparison() -> None:
    """
    Example 3: Coefficient comparison across different feature scales.

    This example demonstrates how marginal effects help interpret
    coefficients across variables with different scales.
    """
    print("\n\n" + "=" * 70)
    print("Example 3: Coefficient Interpretation across Different Scales")
    print("=" * 70)

    # Generate data with different scales
    np.random.seed(456)
    n = 180

    # Feature 1: Small scale (0-10)
    x1 = np.linspace(0, 10, n)

    # Feature 2: Large scale (0-1000)
    x2 = np.linspace(0, 1000, n)

    # Target: y = 50 + 5*x1 + 0.01*x2
    y = 50 + 5 * x1 + 0.01 * x2 + np.random.normal(0, 20, n)

    df = pd.DataFrame(
        {
            "y": y,
            "feature_small": x1,
            "feature_large": x2,
        }
    )

    print("\nDataset Summary:")
    print(df.describe())

    # Fit model
    model = LinearRegressionModel()
    model.fit(df[["feature_small", "feature_large"]], df["y"])

    print("\nCoefficients:")
    coef_df = model.get_coefficients()
    print(coef_df)

    # Marginal effects (equivalent to coefficients for linear model)
    calculator = MarginalEffectsCalculator(model)

    print("\n" + "-" * 70)
    print("Marginal Effects (Standardized Interpretation):")
    print("-" * 70)

    for var in ["feature_small", "feature_large"]:
        me = calculator.marginal_effect(var)
        x_mean = df[var].mean()
        y_mean = df["y"].mean()

        # Percentage effect: percentage change in Y per 1% change in X
        pct_change_effect = (me * x_mean / y_mean) * 100

        print(f"\n{var}:")
        print(f"  Coefficient (Marginal Effect): {me:.6f}")
        print(f"  % Change in Y per 1% change in X: {pct_change_effect:.6f}%")


if __name__ == "__main__":
    example_1_simple_linear_regression()
    example_2_comprehensive_analysis()
    example_3_coefficient_comparison()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
