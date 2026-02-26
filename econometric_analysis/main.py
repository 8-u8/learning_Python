"""
Main entry point for the econometric analysis application.

This script demonstrates the basic workflow of fitting a linear regression
model and calculating marginal effects.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from regression import (
    DoubLogRegressionModel,
    LinearRegressionModel,
    MarginalEffectsCalculator,
)


def main() -> None:
    """Run a simple demonstration of the regression analysis."""
    print("=" * 70)
    print("計量経済学的線形回帰分析 - デモンストレーション")
    print("=" * 70)

    # サンプルデータの生成
    np.random.seed(42)
    n = 100

    # 説明変数
    advertising = np.linspace(10, 100, n)
    price = np.linspace(50, 200, n)

    # 目的変数: Sales = 500 + 3*Advertising - 1*Price + noise
    sales = 500 + 3 * advertising - 1 * price + np.random.normal(0, 50, n)

    # DataFrameの作成
    df = pd.DataFrame(
        {
            "Sales": sales,
            "Advertising": advertising,
            "Price": price,
        }
    )

    print("\n【データセット要約】")
    print(df.describe())

    # モデルの推定
    print("\n【モデルのフィッティング】")
    model = LinearRegressionModel()
    model.fit(df[["Advertising", "Price"]], df["Sales"])
    print("✓ モデルのフィッティング完了")

    # 統計サマリー
    print("\n【回帰分析の結果】")
    print(model.summary())

    # 係数の取得
    print("\n【推定係数】")
    coef_df = model.get_coefficients()
    print(coef_df)

    # 限界効果の計算
    print("\n【限界効果分析】")
    calculator = MarginalEffectsCalculator(model)

    for var in ["Advertising", "Price"]:
        me = calculator.marginal_effect(var)
        print(f"\n{var}:")
        print(f"  限界効果: {me:.4f}")
        print(f"  解釈: {var}が1単位増加すると、売上は{me:.4f}単位変化します")

    # 弾力性の計算
    print("\n【弾力性分析】")
    x_mean_adv = df["Advertising"].mean()
    y_pred_mean = model.predict(df[["Advertising", "Price"]]).mean()

    elasticity_adv = calculator.elasticity("Advertising", x_mean_adv, y_pred_mean)
    print(f"\n広告費の弾力性: {elasticity_adv:.6f}")
    print(f"解釈: 広告費が1%増加するとき、売上は{elasticity_adv:.4f}%変化します")

    # 応答性分析
    print("\n【応答性分析（広告費）】")
    responsiveness = calculator.responsiveness_analysis(
        "Advertising", percentile_change=1.0
    )
    for key, value in responsiveness.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # 両対数モデルのデモ
    print("\n" + "=" * 70)
    print("【両対数モデル - 飽和点検出デモ】")
    print("=" * 70)

    # 両対数データの生成（べき乗法則）
    np.random.seed(123)
    n = 150
    marketing_budget = np.linspace(50, 1000, n)
    revenue = 100000 * (marketing_budget**0.65) + np.random.normal(0, 50000, n)

    df_doublog = pd.DataFrame({"Revenue": revenue, "Marketing": marketing_budget})

    # 両対数モデルの推定
    doublog_model = DoubLogRegressionModel()
    doublog_model.fit(df_doublog[["Marketing"]], df_doublog["Revenue"])

    print("\n【推定結果】")
    print(f"R² = {doublog_model.r_squared():.6f}")

    # 限界効果と飽和点
    doublog_calc = MarginalEffectsCalculator(doublog_model)

    print("\n【異なるマーケティング予算での限界効果】")
    print("（効果が減少する逓減効果を確認できます）")

    for budget in [100, 300, 700]:
        me = doublog_calc.marginal_effect("Marketing", mean_value=budget)
        print(f"  予算${budget:3d}千円: 限界効果=${me:10,.2f}")

    # 飽和点検出
    x_range = np.linspace(
        df_doublog["Marketing"].min(), df_doublog["Marketing"].max(), 100
    )
    saturation = doublog_calc.detect_saturation_point("Marketing", x_range)

    print("\n【飽和点分析】")
    print(f"  Beta係数（弾力性）: {saturation['beta_coefficient']:.4f}")
    print(f"  逓減効果あり: {saturation['diminishing_returns']}")

    if saturation["saturation_threshold_x"] is not None:
        print(f"  飽和点（実務的なゼロ）: ${saturation['saturation_threshold_x']:,.0f}")
    else:
        print(f"  飽和点（実務的なゼロ）: 範囲内で未検出（{saturation['saturation_threshold_x']}）")

    print(f"\n  解釈: {saturation['interpretation']}")

    print("\n" + "=" * 70)
    print("デモンストレーション完了!")
    print("=" * 70)


if __name__ == "__main__":
    main()
