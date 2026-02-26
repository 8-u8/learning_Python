# 計量経済学的線形回帰分析モジュール

## 概要

このプロジェクトは、計量経済学的アプローチに基づいた線形回帰分析を実装しています。**限界効果（Marginal Effects）** と **飽和点検出** を中心とした、経済学的解釈に重点を置いた機能を提供します。

## 主要機能

### 1. 線形回帰モデル (`LinearRegressionModel`)

OLS（普通最小二乗法）を使用した線形回帰の推定

- **係数推定**: 説明変数の係数を推定
- **予測**: フィッティング後のデータに対する予測値の計算
- **診断**: R², 調整R², 統計サマリーの提供

```python
from regression import LinearRegressionModel
import pandas as pd

# モデルの作成とフィッティング
model = LinearRegressionModel()
model.fit(X, y)

# 予測
predictions = model.predict(X_test)

# 統計サマリーの表示
print(model.summary())
```

### 2. 限界効果計算器 (`MarginalEffectsCalculator`)

説明変数から目的変数への影響度を多角的に分析

#### 限界効果 (Marginal Effects)
- **定義**: 説明変数の1単位変化に対する目的変数の変化量
- **線形モデル**: 係数と一致（定数的な効果）

```python
from regression import MarginalEffectsCalculator

calculator = MarginalEffectsCalculator(model)

# 限界効果を計算
me = calculator.marginal_effect('advertising')
# 解釈: 広告費が1単位増加すると、売上が me だけ増加する
```

#### 弾力性（Elasticity）
- **定義**: 説明変数の1%変化に対する目的変数の%変化
- **計算式**: $e = \frac{dY}{dX} \times \frac{X}{Y}$
- **経済学的解釈**: より直感的な効果の大きさを表現

```python
elasticity = calculator.elasticity('advertising', x_value=50, y_value=1000)
# 解釈: 広告費が1%増加するとき、売上は elasticity% 変化する
```

#### 飽和点検出 (Saturation Point Detection)
- **線形モデルの場合**: 定数的な限界効果のため、飽和点は存在しない
- **効果の方向性**: 限界効果が正か負かを判定

```python
x_range = np.linspace(df['advertising'].min(), df['advertising'].max(), 50)
saturation_info = calculator.detect_saturation_point('advertising', x_range)

# 返り値の例:
# {
#     'variable': 'advertising',
#     'marginal_effect': 2.5,
#     'x_min': 10.0,
#     'x_max': 100.0,
#     'saturation_type': 'linear',
#     'is_decreasing': False,
#     'interpretation': '...'
# }
```

#### 応答性分析 (Responsiveness Analysis)
- **変数の応答性**: 説明変数の百分比変化に対する目的変数の絶対的・相対的変化

```python
responsiveness = calculator.responsiveness_analysis('advertising', percentile_change=1.0)

# 返り値の例:
# {
#     'variable': 'advertising',
#     'base_x_mean': 50.5,
#     'base_y_mean': 1000.0,
#     'percentage_change_x': 1.0,
#     'absolute_x_change': 0.505,
#     'absolute_y_effect': 1.2625,
#     'percentage_y_effect': 0.1263  # %
# }
```

### 3. 統合分析器 (`RegressionAnalyzer`)

モデルフィッティングと効果計算を統合したインターフェース

```python
from regression import RegressionAnalyzer

analyzer = RegressionAnalyzer()
analyzer.fit_and_analyze(X, y)

# 特定の変数の包括的な分析
analysis = analyzer.analyze_variable('advertising')

# 以下を含む:
# - 限界効果
# - 飽和点情報
# - 応答性分析
```

### 4. 片対数回帰モデル (`SemiLogRegressionModel`)

説明変数が対数変換された回帰モデル。

**モデル形式**:
$$y = \beta_0 + \beta_1 \ln(x_1) + \beta_2 \ln(x_2) + \cdots + \epsilon$$

**係数の解釈**:
- 係数 $\beta_j$ は、説明変数 $x_j$ が1%変化するときの目的変数の**絶対変化**（セミ・エラスティシティ）を表します
- 例：$\beta_1 = 50$ なら、広告費が1%増加すると売上は50単位増加します

```python
from regression import SemiLogRegressionModel, MarginalEffectsCalculator

# 片対数モデルの推定
model = SemiLogRegressionModel()
model.fit(df[['advertising', 'price']], df['sales'])

# 限界効果（セミ・エラスティシティ）の計算
calculator = MarginalEffectsCalculator(model)
semi_elasticity = calculator.marginal_effect('advertising')

# 弾力性の計算
elasticity = calculator.elasticity('advertising', x_value=100, y_value=1000)
# 解釈: 広告費が1%増加するとき、売上は elasticity% 変化する
```

**使用場面**:
- マーケティング効果測定（ROI分析）
- 価格弾力性分析
- 所得弾力性の推定

### 5. 一般化加法モデル (`GAMRegressionModel`)

非線形な関係性を柔軟に捉える加法モデル。

**モデル形式**:
$$y = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + \epsilon$$

ここで $f_j$ はスプライン平滑化で推定される平滑関数です。

**特徴**:
- 線形性の仮定を緩和
- 各説明変数の非線形効果を個別に推定
- 限界効果が変数の値に依存する（数値微分で計算）

```python
from regression import GAMRegressionModel, MarginalEffectsCalculator

# GAMモデルの推定
model = GAMRegressionModel(lam=0.6)  # λは平滑化パラメータ
model.fit(df[['marketing', 'competition']], df['revenue'])

# 異なる値での限界効果を計算
calculator = MarginalEffectsCalculator(model)

# 平均値での限界効果
me_at_mean = calculator.marginal_effect('marketing')

# 特定の値での限界効果
me_at_low = calculator.marginal_effect('marketing', mean_value=50)
me_at_high = calculator.marginal_effect('marketing', mean_value=200)

print(f"Diminishing returns? {abs(me_at_low - me_at_high) > 0.001}")

# 弾力性の計算（非線形モデルでも可）
elasticity = calculator.elasticity('marketing', x_value=100, y_value=5000)
```

**特別な機能**:
- `get_partial_dependence()`: 特定の説明変数の部分依存関係をプロット用に抽出

```python
# 部分依存プロット用データの取得
x_range = np.linspace(50, 200, 100)
partial_dep = model.get_partial_dependence(0, x_range)
# これをプロットすることで非線形な関係を視覚化できます
```

**使用場面**:
- マーケティングの収穫逓減効果の実証
- 非線形な価格反応関数の推定
- 環境要因の複雑な効果の分析

### 6. 拡張された限界効果計算器

`MarginalEffectsCalculator` はすべてのモデルタイプに対応：

```python
from regression import (
    LinearRegressionModel,
    SemiLogRegressionModel,
    GAMRegressionModel,
    MarginalEffectsCalculator,
)

# どのモデルにも対応
models = [
    LinearRegressionModel(),
    SemiLogRegressionModel(),
    GAMRegressionModel(),
]

for model in models:
    model.fit(X, y)
    calculator = MarginalEffectsCalculator(model)
    
    # 同じインターフェースで限界効果と弾力性を計算
    me = calculator.marginal_effect('x')
    elasticity = calculator.elasticity('x', x_value=100, y_value=1000)
```

**モデル別の計算方法**:

| 機能 | 線形モデル | 片対数モデル | GAMモデル |
|-----|---------|----------|---------|
| 限界効果 | 定数（係数） | 定数（係数） | 数値微分 |
| 弾力性 | ME × (X/Y) | ME × (X/Y) | ME × (X/Y) |
| 飽和点検出 | 線形(飽和なし) | 線形(飽和なし) | 実装可能 |

## インストール・セットアップ

### 依存パッケージ

```
numpy >= 2.4.2
pandas >= 3.0.1
scipy >= 1.17.1
statsmodels >= 0.14.6
scikit-learn >= 1.8.0
matplotlib >= 3.10.8
seaborn >= 0.13.2
pygam >= 0.10.1
```

### 環境構築（`uv`を使用）

```bash
# 仮想環境の作成と依存パッケージのインストール
uv sync

# 仮想環境の有効化
source .venv/bin/activate

# または直接実行
uv run python src/example_usage.py

# 拡張モデルの例を実行
uv run python src/advanced_examples.py
```

## 使用例

### 例1: 基本的な線形回帰と限界効果分析

```python
import numpy as np
import pandas as pd
from regression import LinearRegressionModel, MarginalEffectsCalculator

# データの準備
np.random.seed(42)
n = 150
data = pd.DataFrame({
    'Sales': np.random.normal(1000, 100, n),
    'Advertising': np.linspace(10, 100, n),
    'Price': np.linspace(50, 200, n),
})

# モデルのフィッティング
model = LinearRegressionModel()
model.fit(data[['Advertising', 'Price']], data['Sales'])

# 統計サマリー
print(model.summary())
print(f"R-squared: {model.r_squared():.4f}")

# 限界効果の計算
calculator = MarginalEffectsCalculator(model)
for var in ['Advertising', 'Price']:
    me = calculator.marginal_effect(var)
    print(f"{var} の限界効果: {me:.4f}")
```

### 例2: 飽和点検出と応答性分析

```python
from regression import RegressionAnalyzer
import numpy as np

analyzer = RegressionAnalyzer()
analyzer.fit_and_analyze(X, y)

# マーケティング変数の包括的な分析
analysis = analyzer.analyze_variable('marketing')

print("限界効果:", analysis['marginal_effect'])
print("飽和点情報:", analysis['saturation'])
print("応答性:", analysis['responsiveness'])
```

### 例3: 片対数モデルによる弾力性分析

```python
from regression import SemiLogRegressionModel, MarginalEffectsCalculator
import numpy as np
import pandas as pd

# 対数関係のデータを生成
np.random.seed(42)
n = 150
x1 = np.linspace(10, 500, n)
x2 = np.linspace(20, 200, n)
y = 1000 + 300 * np.log(x1) - 100 * np.log(x2) + np.random.normal(0, 100, n)

df = pd.DataFrame({
    'Sales': y,
    'Advertising': x1,
    'Price': x2,
})

# 片対数モデルの推定
model = SemiLogRegressionModel()
model.fit(df[['Advertising', 'Price']], df['Sales'])

print(model.summary())

# セミ・エラスティシティの計算
calculator = MarginalEffectsCalculator(model)

# 広告費の1%増加の効果
adv_semi_elast = calculator.marginal_effect('Advertising')
print(f"広告費が1%増加 → 売上が {adv_semi_elast:.2f} 単位増加")

# 弾力性（パーセンテージ変化）
elasticity = calculator.elasticity(
    'Advertising',
    x_value=df['Advertising'].mean(),
    y_value=df['Sales'].mean()
)
print(f"弾力性: {elasticity:.4f}")
```

### 例4: 非線形モデル（GAM）による限界効果の非線形性の検出

```python
from regression import GAMRegressionModel, MarginalEffectsCalculator
import numpy as np
import pandas as pd

# 非線形関係のデータを生成
np.random.seed(123)
n = 300
marketing = np.linspace(1, 100, n)
competition = np.linspace(0, 50, n)

# 収穫逓減効果: sqrt(marketing) で表現
revenue = (
    1000 + 50 * np.sqrt(marketing) - 20 * competition
    + np.random.normal(0, 100, n)
)

df = pd.DataFrame({
    'Revenue': revenue,
    'Marketing': marketing,
    'Competition': competition,
})

# GAMモデルの推定
model = GAMRegressionModel(lam=0.6)
model.fit(df[['Marketing', 'Competition']], df['Revenue'])

print(model.summary())

# 異なるレベルでの限界効果を計算
calculator = MarginalEffectsCalculator(model)

marketing_values = [10, 50, 100]
print("\nマーケティング支出レベル別の限界効果:")
print("(収穫逓減効果が見られるはず)")

for val in marketing_values:
    me = calculator.marginal_effect('Marketing', mean_value=val)
    print(f"Marketing={val:3d}: ME={me:.6f}")

# 弾力性の計算
elasticity = calculator.elasticity(
    'Marketing',
    x_value=df['Marketing'].mean(),
    y_value=df['Revenue'].mean()
)
print(f"\n平均値での弾力性: {elasticity:.4f}")
```

## テスト

### ユニットテストの実行

```bash
# 仮想環境が有効な状態で
pytest test/test_regression.py -v

# または
uv run pytest test/test_regression.py -v
```

### 検証スクリプトの実行

```bash
# 基本的な機能チェック
python verify.py

# または
uv run python verify.py

# 拡張モデル（片対数、GAM）の検証
python verify_extended.py

# または
uv run python verify_extended.py
```

### 詳細な使用例の実行

```bash
# 基本例
python src/example_usage.py

# 拡張例（片対数、GAM）
python src/advanced_examples.py

# または uv 経由
uv run python src/advanced_examples.py
```

## プロジェクト構成

```
econometric_analysis/
├── README.md                      # このファイル
├── pyproject.toml                 # プロジェクト設定（pygam追加）
├── main.py                        # デモンストレーション
├── verify.py                      # 基本的な機能検証
├── verify_extended.py             # 拡張モデルの検証
├── src/
│   ├── regression.py              # メインモジュール
│   │                              #  - LinearRegressionModel
│   │                              #  - SemiLogRegressionModel
│   │                              #  - GAMRegressionModel
│   │                              #  - MarginalEffectsCalculator
│   │                              #  - RegressionAnalyzer
│   ├── example_usage.py           # 基本的な使用例
│   └── advanced_examples.py       # 拡張モデルの詳細な例
└── test/
    └── test_regression.py         # ユニットテスト
                                   #  - 線形モデルテスト
                                   #  - 片対数モデルテスト
                                   #  - GAMモデルテスト
```

## 計量経済学的背景

### 限界効果について

限界効果（Marginal Effect）は、ある説明変数を1単位変化させたときの、目的変数の期待値の変化を表します。

$$ME_j = \frac{\partial E[Y]}{\partial X_j}$$

線形回帰モデルの場合、限界効果は係数と同じになります：

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \epsilon$$

$$ME_j = \beta_j$$

### 弾力性について

弾力性は、説明変数の1%変化に対する目的変数の%変化を表します。より直感的な経済解釈を可能にします。

$$e_j = \frac{\partial Y}{\partial X_j} \times \frac{X_j}{Y} = ME_j \times \frac{X_j}{Y}$$

例えば、広告費の弾力性が0.5であれば、広告費が1%増加するとき、売上は0.5%増加することを意味します。

### 片対数モデル（Semi-Log Model）

片対数モデルは説明変数が対数変換されたモデルです：

$$Y = \beta_0 + \beta_1 \ln(X_1) + \beta_2 \ln(X_2) + \cdots + \epsilon$$

このモデルでは、係数 $\beta_j$ は**セミ・エラスティシティ** を表します：

$$\frac{\Delta Y}{\Delta \ln(X_j)} = \beta_j$$

つまり、$X_j$ が1%変化するとき、$Y$ は $\beta_j$ 単位変化します。

**計量経済学的な意義**:
- マーケティング効果測定で頻用される
- 係数が直接的に「1%変化の効果」を表すため、経済学的に解釈しやすい
- 収穫逓減効果（log関数の性質）を自然に組み込める

### 非線形モデル（一般化加法モデル：GAM）

一般化加法モデルは非線形な関係を柔軟に捉えます：

$$Y = \beta_0 + f_1(X_1) + f_2(X_2) + \cdots + \epsilon$$

ここで $f_j(\cdot)$ は平滑スプライン関数です。

**特徴**:
- 線形性の仮定を緩和
- 各説明変数の限界効果が値に依存（非定数）
- 限界効果の非線形性を直接検出可能

**限界効果の計算**:
$$ME_j(X_j) = \frac{\partial f_j(X_j)}{\partial X_j}$$

数値微分で推定されます。

**計量経済学的な意義**:
- 実務的に見られる「逓減効果」「閾値効果」などを自然に推定
- 複数の説明変数の非線形効果を同時に評価可能
- モデル選択の自由度が高い（過剰適合のリスクあり）

### 飽和点について

飽和点は、説明変数の効果が最大値に達し、それ以上増加しない地点を指します。

- **線形モデル**: 飽和点は存在しません（効果は常に一定）
- **GAMモデル**: 非線形性により飽和点が存在する可能性があります

GAMで検出された非線形性は、以下の効果を示唆します：
- $ME_j$ が $X_j$ の値に応じて変化 → 実務的には重要な情報

## 今後の拡張予定

- [ ] 非線形モデル（2次項、交互作用項）への対応
- [ ] ロジスティック回帰への対応
- [ ] 固定効果・変量効果モデルの実装
- [ ] 時系列分析への対応
- [ ] 視覚化機能の充実（部分依存プロット、限界効果プロット）
- [ ] GAMの正則化パラメータの自動チューニング

## 参考文献

1. Wooldridge, J. M. (2019). *Introductory Econometrics: A Modern Approach* (7th ed.). Cengage Learning.
2. Greene, W. H. (2012). *Econometric Analysis* (7th ed.). Pearson.
3. Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
   - 特に Chapter 9: Additive Models and Trees
3. Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.

## ライセンス

MIT License

## 作成者

計量経済学分析プロジェクト
