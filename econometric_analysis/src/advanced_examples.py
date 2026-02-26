"""
Advanced examples demonstrating semi-log and GAM regression models.

This script demonstrates:
1. Semi-log regression (log-transformed explanatory variables)
2. Generalized Additive Models (GAM) for nonlinear relationships
3. Marginal effects and elasticity calculations for each model type
"""

# Add src to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from regression import (
    DoubLogRegressionModel,
    GAMRegressionModel,
    LinearRegressionModel,
    MarginalEffectsCalculator,
    SemiLogRegressionModel,
)


def example_1_semilog_regression() -> None:
    """
    Example 1: Semi-log regression (log-transformed explanatory variables).

    In this model: y = β₀ + β₁*ln(x₁) + β₂*ln(x₂) + ε
    The coefficient β_j represents the change in y for a 1% change in x_j.
    """
    print("=" * 80)
    print("Example 1: Semi-Log Regression Analysis")
    print("=" * 80)

    # Generate synthetic data with log relationship
    np.random.seed(42)
    n = 200

    # Explanatory variables (must be positive for log)
    advertising = np.linspace(10, 500, n)
    price = np.linspace(20, 200, n)

    # True relationship: Sales = 1000 + 300*ln(Advertising) - 100*ln(Price) + noise
    sales = (
        1000
        + 300 * np.log(advertising)
        - 100 * np.log(price)
        + np.random.normal(0, 100, n)
    )

    df = pd.DataFrame(
        {
            "Sales": sales,
            "Advertising": advertising,
            "Price": price,
        }
    )

    print("\nDataset Summary:")
    print(df.describe())

    # Fit semi-log model
    model = SemiLogRegressionModel()
    model.fit(df[["Advertising", "Price"]], df["Sales"])

    print("\n" + "-" * 80)
    print("Semi-Log Model Results (explanatory variables log-transformed):")
    print("-" * 80)
    print(model.summary())

    # Marginal effects and elasticity
    calculator = MarginalEffectsCalculator(model)

    print("\n" + "-" * 80)
    print("Interpretation of Coefficients:")
    print("-" * 80)

    for var in ["Advertising", "Price"]:
        coef = calculator.marginal_effect(var)
        print(f"\n{var} coefficient: {coef:.2f}")
        print(f"  → A 1% increase in {var} leads to a {coef:.2f} unit change in Sales")

    # Elasticity at mean values
    print("\n" + "-" * 80)
    print("Elasticity Analysis (at mean values):")
    print("-" * 80)

    x_adv_mean = df["Advertising"].mean()
    y_pred_mean = model.predict(df[["Advertising", "Price"]]).mean()

    elasticity_adv = calculator.elasticity("Advertising", x_adv_mean, y_pred_mean)
    elasticity_price = calculator.elasticity("Price", df["Price"].mean(), y_pred_mean)

    print(f"\nAdvertising Elasticity: {elasticity_adv:.4f}")
    print(
        f"  → A 1% increase in Advertising leads to {elasticity_adv:.4f}% change in Sales"
    )

    print(f"\nPrice Elasticity: {elasticity_price:.4f}")
    print(
        f"  → A 1% increase in Price leads to {elasticity_price:.4f}% change in Sales"
    )

    print(f"\nR²: {model.r_squared():.4f}")


def example_2_gam_nonlinear() -> None:
    """
    Example 2: Generalized Additive Model (GAM) for nonlinear relationships.

    GAM allows nonlinear relationships: y = β₀ + f₁(x₁) + f₂(x₂) + ... + ε
    where f_j are smooth spline functions fitted to the data.
    """
    print("\n\n" + "=" * 80)
    print("Example 2: GAM (Generalized Additive Model) - Nonlinear Regression")
    print("=" * 80)

    # Generate nonlinear data
    np.random.seed(123)
    n = 300

    # Explanatory variables
    marketing = np.linspace(1, 100, n)
    competition = np.linspace(0, 50, n)

    # Nonlinear relationship
    # Revenue exhibits diminishing returns to marketing and negative effect from competition
    revenue = (
        1000
        + 50 * np.sqrt(marketing)
        - 20 * competition
        - 0.1 * competition**2
        + np.random.normal(0, 100, n)
    )

    df = pd.DataFrame(
        {
            "Revenue": revenue,
            "Marketing": marketing,
            "Competition": competition,
        }
    )

    print("\nDataset Summary:")
    print(df.describe())

    # Fit GAM
    print("\nFitting GAM model...")
    model = GAMRegressionModel(lam=0.6)
    model.fit(df[["Marketing", "Competition"]], df["Revenue"])

    print(model.summary())

    # Marginal effects at different points
    calculator = MarginalEffectsCalculator(model)

    print("\n" + "-" * 80)
    print("Marginal Effects Analysis (Nonlinear Model):")
    print("-" * 80)

    # Marginal effects at mean values
    for var in ["Marketing", "Competition"]:
        me_mean = calculator.marginal_effect(var)
        print(f"\n{var}:")
        print(f"  Marginal effect (at mean): {me_mean:.6f}")

        # Marginal effects at min, mean, max
        x_min = df[var].min()
        x_mean = df[var].mean()
        x_max = df[var].max()

        me_min = calculator.marginal_effect(var, mean_value=x_min)
        me_max = calculator.marginal_effect(var, mean_value=x_max)

        print(f"  Marginal effect (at min):  {me_min:.6f}")
        print(f"  Marginal effect (at max):  {me_max:.6f}")
        print(
            f"  Effect changes with {var}? "
            f"{'Yes (nonlinear)' if abs(me_min - me_max) > 0.001 else 'No'}"
        )

    # Elasticity analysis
    print("\n" + "-" * 80)
    print("Elasticity Analysis for GAM Model:")
    print("-" * 80)

    y_mean = df["Revenue"].mean()

    for var in ["Marketing", "Competition"]:
        x_mean = df[var].mean()
        elasticity = calculator.elasticity(var, x_mean, y_mean)
        print(f"\n{var} Elasticity (at mean): {elasticity:.6f}")


def example_3_model_comparison() -> None:
    """
    Example 3: Compare linear, semi-log, and nonlinear (GAM) models.

    This example shows how different model specifications lead to different
    interpretations of the same relationship.
    """
    print("\n\n" + "=" * 80)
    print("Example 3: Model Comparison - Linear vs Semi-Log vs GAM")
    print("=" * 80)

    # Generate data
    np.random.seed(456)
    n = 250

    x = np.linspace(5, 500, n)
    y = 100 + 50 * np.log(x) + np.random.normal(0, 50, n)

    df = pd.DataFrame({"y": y, "x": x})

    print("\nDataset: y = 100 + 50*ln(x) + noise\n")

    # 1. Linear Model
    print("-" * 80)
    print("1. Linear Model: y = β₀ + β₁*x + ε")
    print("-" * 80)

    linear_model = LinearRegressionModel()
    linear_model.fit(df[["x"]], df["y"])
    print(f"Coefficient: {linear_model.coef_[0]:.6f}")
    print(f"R²: {linear_model.r_squared():.6f}")

    linear_calc = MarginalEffectsCalculator(linear_model)
    me_linear = linear_calc.marginal_effect("x")
    print(f"Marginal Effect: {me_linear:.6f}")

    # 2. Semi-Log Model
    print("\n" + "-" * 80)
    print("2. Semi-Log Model: y = β₀ + β₁*ln(x) + ε")
    print("-" * 80)

    semilog_model = SemiLogRegressionModel()
    semilog_model.fit(df[["x"]], df["y"])
    print(f"Coefficient: {semilog_model.coef_[0]:.6f}")
    print(f"R²: {semilog_model.r_squared():.6f}")

    semilog_calc = MarginalEffectsCalculator(semilog_model)
    me_semilog = semilog_calc.marginal_effect("x")
    print(f"Semi-Elasticity (1% change effect): {me_semilog:.6f}")

    # 3. GAM Model
    print("\n" + "-" * 80)
    print("3. GAM Model: y = β₀ + f(x) + ε")
    print("-" * 80)

    gam_model = GAMRegressionModel(lam=0.7)
    gam_model.fit(df[["x"]], df["y"])
    print(f"R²: {gam_model.r_squared():.6f}")

    gam_calc = MarginalEffectsCalculator(gam_model)

    # Marginal effects at different points
    x_vals = [50, 250, 450]
    print("\nMarginal Effects at Different Points:")
    for x_val in x_vals:
        me = gam_calc.marginal_effect("x", mean_value=x_val)
        print(f"  At x={x_val:3d}: {me:.6f}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary Comparison:")
    print("=" * 80)
    print(f"Linear Model R²:     {linear_model.r_squared():.6f}")
    print(f"Semi-Log Model R²:   {semilog_model.r_squared():.6f}")
    print(f"GAM Model R²:        {gam_model.r_squared():.6f}")
    print(f"\n→ Semi-Log model fits best (matches the true log relationship)")


def example_4_doublog_saturation() -> None:
    """
    Example 4: Double-log model (power law) with saturation point detection.

    This example demonstrates how double-log regression captures
    diminishing returns effects and where saturation occurs.

    Model: ln(y) = β₀ + β₁*ln(x)
    Expands to: y = a * x^β₁

    When 0 < β₁ < 1, diminishing returns are observed.
    The saturation point is where marginal effect becomes negligibly small.
    """
    print("\n\n" + "=" * 80)
    print("Example 4: Double-Log Model - Diminishing Returns & Saturation")
    print("=" * 80)

    # Generate power law data
    np.random.seed(789)
    n = 200

    # Marketing budget (feature)
    marketing = np.linspace(10, 1000, n)

    # Revenue (target) - power law relationship: y = 50000 * x^0.65
    revenue = 50000 * (marketing**0.65) + np.random.normal(0, 50000, n)

    df = pd.DataFrame(
        {
            "Revenue": revenue,
            "Marketing": marketing,
        }
    )

    print("\nDataset Summary:")
    print(df.describe())

    # Fit double-log model
    model = DoubLogRegressionModel()
    model.fit(df[["Marketing"]], df["Revenue"])

    print("\nDouble-Log Model Results:")
    print(model.summary())

    # Analyze marginal effects and saturation
    calculator = MarginalEffectsCalculator(model)

    print("\n" + "-" * 80)
    print("Marginal Effects Analysis:")
    print("-" * 80)

    # Marginal effects at different marketing levels
    marketing_levels = [50, 200, 500, 1000]
    print("\nMarginal Effect at Different Marketing Spend Levels:")
    print("(Shows diminishing returns as marketing increases)")

    for m_level in marketing_levels:
        me = calculator.marginal_effect("Marketing", mean_value=m_level)
        print(f"  Marketing=${m_level:4d}: ME=${me:10,.2f} per dollar")

    # Saturation point detection
    print("\n" + "-" * 80)
    print("Saturation Point Detection:")
    print("-" * 80)

    x_range = np.linspace(df["Marketing"].min(), df["Marketing"].max(), 100)
    saturation_info = calculator.detect_saturation_point("Marketing", x_range)

    print(f"\nBeta Coefficient (elasticity): {saturation_info['beta_coefficient']:.4f}")
    print(f"Diminishing Returns: {saturation_info['diminishing_returns']}")

    if saturation_info["saturation_threshold_x"] is not None:
        print(f"Saturation Point: ${saturation_info['saturation_threshold_x']:,.2f}")
    else:
        print(f"Saturation Point: Not reached in range")

    print(f"\nInterpretation:")
    print(f"  {saturation_info['interpretation']}")

    # Responsiveness analysis
    print("\n" + "-" * 80)
    print("Business Impact Analysis:")
    print("-" * 80)

    responsiveness = calculator.responsiveness_analysis(
        "Marketing", percentile_change=5.0
    )

    print(f"\n5% Increase in Marketing Spend:")
    print(f"  Base Marketing: ${responsiveness['base_x_mean']:,.2f}")
    print(f"  Absolute Change: ${responsiveness['absolute_x_change']:,.2f}")
    print(f"  Revenue Impact: ${responsiveness['absolute_y_effect']:,.2f}")
    print(f"  Revenue % Change: {responsiveness['percentage_y_effect']:.4f}%")

    # R-squared
    print(f"\nR²: {model.r_squared():.6f}")


if __name__ == "__main__":
    example_1_semilog_regression()
    example_2_gam_nonlinear()
    example_3_model_comparison()
    example_4_doublog_saturation()

    print("\n\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
