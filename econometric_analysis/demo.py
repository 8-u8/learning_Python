#!/usr/bin/env python
"""
Final implementation summary and quick demo.

This script provides a comprehensive overview of the implemented functionality.
"""

import sys

import numpy as np
import pandas as pd

# Import all models
try:
    from src.regression import (
        DoubLogRegressionModel,
        GAMRegressionModel,
        LinearRegressionModel,
        MarginalEffectsCalculator,
        SemiLogRegressionModel,
    )

    print("✓ All models imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def main():
    """Run a comprehensive demo of all models."""
    print("=" * 80)
    print("計量経済学的回帰分析 - 統合実装デモ")
    print("=" * 80)

    # Generate demo data
    np.random.seed(42)
    n = 150

    x = np.linspace(5, 500, n)
    y = 100 + 50 * np.log(x) + np.random.normal(0, 50, n)

    df = pd.DataFrame({"y": y, "x": x})

    print("\n" + "=" * 80)
    print("1. Linear Model (線形モデル)")
    print("=" * 80)

    linear = LinearRegressionModel()
    reg_x = df[["x"]]
    reg_y = df["y"]
    linear.fit(X=reg_x, y=reg_y)

    print(f"R² = {linear.r_squared():.6f}")

    linear_calc = MarginalEffectsCalculator(linear)
    me = linear_calc.marginal_effect("x")
    elasticity = linear_calc.elasticity("x", df["x"].mean(), df["y"].mean())

    print(f"Marginal Effect = {me:.6f}")
    print(f"Elasticity = {elasticity:.6f}")

    print("\n" + "=" * 80)
    print("2. Semi-Log Model (片対数モデル)")
    print("=" * 80)

    semilog = SemiLogRegressionModel()
    semilog.fit(df[["x"]], df["y"])

    print(f"R² = {semilog.r_squared():.6f}")

    semilog_calc = MarginalEffectsCalculator(semilog)
    me_semi = semilog_calc.marginal_effect("x")
    elasticity_semi = semilog_calc.elasticity("x", df["x"].mean(), df["y"].mean())

    print(f"Semi-Elasticity (1%の効果) = {me_semi:.6f}")
    print(f"Elasticity = {elasticity_semi:.6f}")
    print("→ 説明：真の関係が log なので、最も当てはまりが良いはず")

    print("\n" + "=" * 80)
    print("3. GAM Model (一般化加法モデル)")
    print("=" * 80)

    gam = GAMRegressionModel(lam=0.7)
    gam.fit(df[["x"]], df["y"])

    print(f"R² = {gam.r_squared():.6f}")

    gam_calc = MarginalEffectsCalculator(gam)

    # Marginal effects at different points
    x_vals = [50, 250, 450]
    print("\nMarginal Effects at Different X Values:")
    print("(Diminishing returns effect should be visible)")

    for x_val in x_vals:
        me_gam = gam_calc.marginal_effect("x", mean_value=x_val)
        print(f"  x = {x_val:3d}: ME = {me_gam:.6f}")

    elasticity_gam = gam_calc.elasticity("x", df["x"].mean(), df["y"].mean())
    print(f"\nElasticity (at mean) = {elasticity_gam:.6f}")

    print("\n" + "=" * 80)
    print("4. Double-Log Model (Power Law) - Saturation Point Detection")
    print("=" * 80)

    # Double-log data with diminishing returns
    np.random.seed(555)
    n = 150
    x_doublog = np.linspace(10, 500, n)
    y_doublog = 100 * (x_doublog**0.7) + np.random.normal(0, 100, n)

    df_doublog = pd.DataFrame({"y": y_doublog, "x": x_doublog})

    doublog = DoubLogRegressionModel()
    doublog.fit(df_doublog[["x"]], df_doublog["y"])

    print(f"R² = {doublog.r_squared():.6f}")

    doublog_calc = MarginalEffectsCalculator(doublog)

    # Get coefficients
    coef = doublog.get_coefficients()
    beta = coef.loc[0, "Coefficient"]

    print(f"\nElasticity (Beta): {beta:.4f}")
    print("Interpretation: Power law relationship y = a * x^β")

    # Marginal effects at different points
    print("\nMarginal Effects at Different X Values:")
    print("(Showing diminishing returns)")

    for x_val in [50, 150, 300]:
        me = doublog_calc.marginal_effect("x", mean_value=x_val)
        print(f"  x = {x_val:3d}: ME = {me:.6f} (declining effect)")

    # Saturation point detection
    x_range = np.linspace(df_doublog["x"].min(), df_doublog["x"].max(), 100)
    sat_info = doublog_calc.detect_saturation_point("x", x_range)

    print(f"\nSaturation Point Analysis:")
    print(f"  Diminishing Returns: {sat_info['diminishing_returns']}")

    if sat_info["saturation_threshold_x"] is not None:
        print(f"  Saturation at x ≈ {sat_info['saturation_threshold_x']:.2f}")
    else:
        print(f"  Saturation: Not reached in range")

    print(f"  Interpretation: {sat_info['interpretation']}")

    print("\n" + "=" * 80)
    print("5. Model Comparison Summary")
    print("=" * 80)

    print(f"{'Model':<20} {'R²':<15} {'Notes'}")
    print("-" * 80)
    print(f"{'Linear':<20} {linear.r_squared():<15.6f} {'Simplest model'}")
    print(
        f"{'Semi-Log':<20} {semilog.r_squared():<15.6f} {'Best fit (log relationship)'}"
    )
    print(
        f"{'Double-Log':<20} {doublog.r_squared():<15.6f} {'Power law, diminishing returns'}"
    )
    print(
        f"{'GAM':<20} {gam.r_squared():<15.6f} {'Most flexible, captures nonlinearity'}"
    )

    print("\n" + "=" * 80)
    print("6. Key Features Summary")
    print("=" * 80)

    features = {
        "LinearRegressionModel": [
            "✓ Constant marginal effects",
            "✓ Simple interpretation",
            "✓ High transparency",
        ],
        "SemiLogRegressionModel": [
            "✓ Log-transformed features",
            "✓ Direct elasticity interpretation",
            "✓ Good for marketing analytics",
        ],
        "GAMRegressionModel": [
            "✓ Nonlinear relationships",
            "✓ Varying marginal effects",
            "✓ Detects diminishing returns",
        ],
        "DoubLogRegressionModel": [
            "✓ Power law relationships",
            "✓ Elasticity interpretation via Beta",
            "✓ Saturation point detection",
        ],
    }

    for model_name, features_list in features.items():
        print(f"\n{model_name}:")
        for feature in features_list:
            print(f"  {feature}")

    print("\n" + "=" * 80)
    print("Implementation Completed Successfully! ✓")
    print("=" * 80)

    print("\nNext Steps:")
    print("1. Run: python verify.py (基本機能確認)")
    print("2. Run: python verify_extended.py (拡張機能確認)")
    print("3. Run: python src/example_usage.py (基本例)")
    print("4. Run: python src/advanced_examples.py (詳細例)")
    print("5. Run: pytest test/test_regression.py -v (全テスト)")

    print("\nDocumentation:")
    print("- README.md: 詳細なドキュメント")
    print("- IMPLEMENTATION.md: 実装内容の説明")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
