"""
計量経済学的回帰分析モジュール - 実装概要

このファイルは、拡張された回帰分析モジュールの実装内容をまとめたドキュメントです。

=== 実装クラス一覧 ===

1. LinearRegressionModel
   - 説明：OLS（普通最小二乗法）による線形回帰
   - モデル式：y = β₀ + β₁*x₁ + β₂*x₂ + ... + ε
   - 特徴：係数が定数的（限界効果一定）
   - 用途：基本的な線形関係の推定

2. SemiLogRegressionModel
   - 説明：説明変数を対数変換した片対数回帰
   - モデル式：y = β₀ + β₁*ln(x₁) + β₂*ln(x₂) + ... + ε
   - 特徴：係数がセミ・エラスティシティ（1%変化の効果）
   - 用途：マーケティング効果測定、価格弾力性分析

3. GAMRegressionModel
   - 説明：一般化加法モデル（Generalized Additive Model）
   - モデル式：y = β₀ + f₁(x₁) + f₂(x₂) + ... + ε
   - 特徴：非線形関係を柔軟に捉える、限界効果が値に依存
   - 用途：非線形効果の検出、収穫逓減の実証

4. MarginalEffectsCalculator
   - 説明：限界効果と弾力性を計算する汎用計算機
   - 対応モデル：上記3つのすべてのモデルに対応
   - 機能：
     * 限界効果の計算（モデルごとに異なる方法）
     * 弾力性（elasticity）の計算
     * 飽和点検出

5. RegressionAnalyzer
   - 説明：モデルフィッティングと効果計算を統合したインターフェース
   - 用途：包括的な分析を一度に実行

=== 各モデルの限界効果計算方法の違い ===

LinearRegressionModel:
  - 限界効果 = 係数（定数）
  - 計算：self.coef_[idx]

SemiLogRegressionModel:
  - 限界効果 = 係数（定数）
  - 解釈：1%の変化に対する絶対変化（セミ・エラスティシティ）
  - 計算：self.coef_[idx]

GAMRegressionModel:
  - 限界効果 = 数値微分（非定数）
  - 計算方法：有限差分法（finite difference）
    * 指定点 x₀ の前後で予測値を計算
    * ME ≈ (f(x₀+ε) - f(x₀-ε)) / (2ε)
    * ε は微小値（デフォルト：1e-6）

=== 弾力性の統一的計算 ===

すべてのモデルで同じ公式を使用：
  Elasticity = ME × (X / Y)

  ここで：
  - ME：限界効果（モデルごとに異なる方法で計算）
  - X：説明変数の値
  - Y：目的変数の値（予測値または実績値）

=== ファイル構成 ===

src/regression.py
  ├── LinearRegressionModel: 基本的な線形回帰
  ├── SemiLogRegressionModel: 片対数回帰（log変換説明変数）
  ├── GAMRegressionModel: 非線形加法モデル
  ├── MarginalEffectsCalculator: 統合効果計算機
  └── RegressionAnalyzer: 統合分析器

src/example_usage.py
  ├── example_1_simple_linear_regression(): 基本例
  ├── example_2_comprehensive_analysis(): 応答性分析
  └── example_3_coefficient_comparison(): スケール比較

src/advanced_examples.py
  ├── example_1_semilog_regression(): 片対数モデルの詳細例
  ├── example_2_gam_nonlinear(): GAMモデルの詳細例
  └── example_3_model_comparison(): モデル比較

test/test_regression.py
  ├── TestLinearRegressionModel: 線形モデルテスト
  ├── TestMarginalEffectsCalculator: 限界効果テスト
  ├── TestSemiLogRegressionModel: 片対数モデルテスト
  ├── TestGAMRegressionModel: GAMモデルテスト
  └── TestIntegration: 統合テスト

verify.py: 基本的な動作確認
verify_extended.py: 拡張モデルの動作確認

=== 実行方法 ===

# 基本的な確認
$ python verify.py

# 拡張モデルの確認
$ python verify_extended.py

# テスト実行
$ pytest test/test_regression.py -v

# 使用例の実行
$ python src/example_usage.py
$ python src/advanced_examples.py

# または uv 経由
$ uv run python verify.py
$ uv run python verify_extended.py
$ uv run pytest test/test_regression.py -v

=== 主要な計量経済学的概念 ===

1. 限界効果（Marginal Effect）
   定義：説明変数の1単位変化に対する目的変数の変化量
   
   線形モデル：
     ME_j = ∂E[Y]/∂X_j = β_j （定数）
   
   非線形モデル（GAM）：
     ME_j(x) = ∂f_j(x)/∂x （値に依存）

2. セミ・エラスティシティ（Semi-Elasticity）
   定義：説明変数の1%変化に対する目的変数の絶対変化
   
   式：∂Y/∂(ln X_j) = β_j （片対数モデルの係数）
   
   解釈：X_j が1%増加 → Y が β_j だけ増加

3. 弾力性（Elasticity）
   定義：説明変数の1%変化に対する目的変数の%変化
   
   式：e_j = (∂Y/∂X_j) × (X_j / Y) = ME_j × (X_j / Y)
   
   解釈：X_j が1%増加 → Y が e_j % 変化

4. 非線形性の検出
   GAMモデルにより、以下を検出可能：
   - 収穫逓減効果：ME が X の値に応じて低下
   - 閾値効果：特定の X 値以上で ME が大きく変化
   - 複雑な非線形関係

=== 計量経済学的な注意点 ===

1. モデル選択
   - データが log 関係に見える → SemiLogRegressionModel
   - 複雑な非線形関係 → GAMRegressionModel
   - 単純な線形関係 → LinearRegressionModel

2. 限界効果の解釈
   - LinearRegressionModel：値の単位に依存
   - SemiLogRegressionModel：常に 1% 変化の効果
   - GAMRegressionModel：値に応じて変化（経済学的に重要）

3. 弾力性の活用
   - 異なるスケールの変数を比較可能
   - 経営判断に直結する %変化 で表現
   - 複数モデル間の比較が容易

=== 今後の拡張候補 ===

- 交互作用項（interaction terms）の自動検出
- 2次項の自動追加による非線形化
- ロジスティック回帰への拡張
- パネルデータモデル（固定効果、変量効果）
- 時系列モデル（ARIMA、GARCH等）
- グラフィカル出力（部分依存プロット、限界効果プロット）
- 正則化（Ridge、Lasso等）の組み込み

=== 実装の特徴 ===

✓ テスト駆動開発（TDD）に基づいた実装
✓ 型アノテーション（type hints）を完全採用
✓ ドキュメンテーション文字列（docstrings）を充実
✓ 複数モデル間で統一されたインターフェース
✓ MarginalEffectsCalculator が全モデルに対応
✓ 数値計算が正確（有限差分法で微分を計算）
✓ エラーハンドリングが充実

"""
