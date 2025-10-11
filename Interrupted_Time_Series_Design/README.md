# Interrupted Time Series Analysis (ITS) Package

このパッケージは、Interrupted Time Series Designを使った分析を抽象化し、任意のデータに対して適用可能なPythonクラス群を提供します。元のITAS_mock.pyファイルをもとに、より汎用的で使いやすい形に設計されています。

## 📋 目次

- [特徴](#特徴)
- [インストール・セットアップ](#インストールセットアップ)
- [クラス構成](#クラス構成)
- [使用方法](#使用方法)
- [サンプルコード](#サンプルコード)
- [テスト](#テスト)
- [貢献](#貢献)

## ✨ 特徴

### 1. ITSDataPreprocessor
Interrupted Time Series Designに必要な以下の変数を自動生成：
- **t**: 時系列の起点から終点までのカウントアップ
- **T**: 介入のあった単位時間のときに1、それ以外は0を示すダミー変数
- **D**: 介入のあった単位時間より前は0、介入のあった単位時間以降に1を示すダミー変数
- **time_after**: 介入後の経過時間を表すカウントアップ
- **const**: 切片項、すなわち全ての期間を通して1となるような変数

### 2. ITSModel
- `ITSDataPreprocessor`を継承し、`statsmodels.api.OLS`を実行
- 共変量(covariates)の有無を自動確認し、適切にモデル実行
- カテゴリカル変数の自動ダミー変数変換
- モデルの保存・読み込み機能

### 3. ITSVisualizer
- グループ別描画に対応した柔軟な可視化
- 反実仮想（counterfactual）ラインの表示
- 信頼区間の表示
- 画像保存機能

## 🚀 インストール・セットアップ

### 必要な依存パッケージ

```bash
# 仮想環境をアクティブ化
source .venv/bin/activate

# 必要なパッケージ（既にインストール済みの場合）
pip install numpy pandas statsmodels matplotlib
```

### ファイル構成

```
Interrupted_Time_Series_Design/
├── src/
│   ├── module/
│   │   ├── __init__.py          # モジュールの初期化
│   │   ├── its_analysis.py      # ITS分析の主要クラス群
│   │   └── generate_report.py   # Markdownレポート生成
│   ├── example_usage.py         # Pythonスクリプト版の使用例
│   └── examples.ipynb           # Jupyter Notebook版の使用例
├── test/
│   ├── test_its_analysis.py           # メインテストコード
│   └── test_import_generate_report.py # レポート生成のインポートテスト
├── models/                      # モデル保存用フォルダ
├── output/                      # 図表・レポート出力用フォルダ
├── README.md                    # このファイル
└── pyproject.toml               # プロジェクト設定
```

## 📚 クラス構成

### ITSDataPreprocessor

```python
from src.module.its_analysis import ITSDataPreprocessor

# 基本的な初期化
preprocessor = ITSDataPreprocessor(
    time_column='year',           # 時間を表すカラム名
    intervention_point=2010,      # 介入が起こった時点
    group_column='state'          # グループを表すカラム名（オプション）
)

# データの変換
processed_data = preprocessor.fit_transform(df)
```

### ITSModel

```python
from src.module.its_analysis import ITSModel

# モデルの初期化とフィッティング
model = ITSModel(
    time_column='year',
    intervention_point=2010,
    group_column='state'
)

# モデルのフィッティング
results = model.fit(
    df, 
    target_column='sales',
    covariates=['state', 'gdp']    # 共変量（オプション）
)

# 予測
predictions = model.predict(return_ci=True)  # 信頼区間付き

# モデルの保存・読み込み
model.save_model('my_model.pkl')
new_model = ITSModel('dummy', 999)
new_model.load_model('my_model.pkl')
```

### ITSVisualizer

```python
from src.module.its_analysis import ITSVisualizer

# 可視化
visualizer = ITSVisualizer(model)

# プロットの作成
fig = visualizer.plot(
    group_column='state',                    # グループ別プロット
    figsize=(12, 8),
    save_path='its_analysis_result.png',    # 画像保存
    show_counterfactual=True,               # 反実仮想ライン表示
    show_confidence_interval=True           # 信頼区間表示
)
```

## 💡 使用方法

### 基本的なワークフロー

```python
import pandas as pd
from src.module.its_analysis import ITSModel, ITSVisualizer

# 1. データの準備（time, outcome, 共変量列を含む）
data = pd.read_csv('your_data.csv')

# 2. モデルの作成と実行
model = ITSModel(
    time_column='time_col',
    intervention_point=intervention_time
)

results = model.fit(data, target_column='outcome')

# 3. 結果の確認
model.summary()

# 4. 可視化
visualizer = ITSVisualizer(model)
fig = visualizer.plot(save_path='result.png')
```

### グループ別分析の場合

```python
# グループ列を指定
model = ITSModel(
    time_column='year',
    intervention_point=2010,
    group_column='region'  # 地域別分析
)

results = model.fit(
    data, 
    target_column='outcome',
    covariates=['region', 'population']  # 地域ダミーと人口を共変量に
)

# グループ別可視化
visualizer = ITSVisualizer(model)
fig = visualizer.plot(group_column='region')
```

## 🎯 サンプルコード

詳細な使用例は `examples.ipynb` (Jupyter Notebook) または `src/examples.py` (Pythonスクリプト) をご覧ください：

```bash
# Jupyter Notebookで実行（推奨）
jupyter notebook examples.ipynb

# またはPythonスクリプトで実行
python src/examples.py
```

このファイルには以下の例が含まれています：
1. **Cigarデータを使った例**: 元のITAS_mock.pyと同じデータでの分析
2. **合成データを使った例**: 複数グループ、複数共変量での分析  
3. **シンプルな例**: 最小限のコードでの分析

### モデル・図表の保存先
- **モデル保存**: `models/` フォルダ (デフォルト: `models/its_model.pkl`)
- **図表保存**: `output/` フォルダ

### クイックスタート例

```python
import numpy as np
import pandas as pd
from src.module.its_analysis import ITSModel, ITSVisualizer

# サンプルデータ作成
np.random.seed(42)
data = pd.DataFrame({
    'time': range(1, 31),
    'outcome': np.concatenate([
        np.random.normal(10, 2, 15),  # 介入前
        np.random.normal(15, 2, 15)   # 介入後
    ])
})

# 分析実行
model = ITSModel('time', intervention_point=16)
results = model.fit(data, target_column='outcome')

# 結果表示
print(f"介入効果: {results.params['D']:.3f}")

# 可視化
visualizer = ITSVisualizer(model)
fig = visualizer.plot()
```

## 🧪 テスト

パッケージには包括的なテストスイートが含まれています：

```bash
# テスト実行
python test/test_its_analysis.py

# 特定のテストクラスのみ実行
python test/test_its_analysis.py TestITSModel

# 特定のテストメソッドのみ実行  
python test/test_its_analysis.py TestIntegration.test_full_workflow
```

### テスト内容

- **ユニットテスト**: 各クラスの個別機能をテスト
- **統合テスト**: 完全なワークフローをテスト
- **エラーハンドリング**: 不正な入力に対する適切な例外処理をテスト

## � Markdownレポート生成

ITS分析結果を包括的なMarkdownレポートとして出力できます。

### 基本的な使い方

```python
from module import generate_markdown_report

# レポート生成
report_path = generate_markdown_report(output_path='output/analysis_report.md')
print(f"レポート生成完了: {report_path}")
```

### モジュールとして実行

```bash
# srcディレクトリをPYTHONPATHに追加して実行
cd /path/to/Interrupted_Time_Series_Design
PYTHONPATH=src python -m module.generate_report
```

### レポートに含まれる内容

1. **データ概要**: 使用したデータセット、変数、介入ポイントの情報
2. **OLSモデル分析**: 複数グループ・複数介入に対応した分析結果
3. **SARIMAXモデル分析**: 時系列の自己相関を考慮した分析結果
4. **Prophetモデル分析**: トレンドと季節性を柔軟にモデル化した分析結果
5. **モデル比較**: 各モデルの介入効果を期間別に比較
6. **可視化**: 各モデルの予測値、反実仮想、信頼区間を含むグラフ

生成されたレポートは`output/analysis_report.md`に保存され、Markdownビューアーで確認できます。

## �📊 分析結果の解釈

ITSモデルの主要な係数：

- **const**: ベースライン水準
- **t**: 介入前の時間トレンド
- **D**: 介入による水準の変化（レベルシフト）
- **time_after**: 介入後の時間トレンドの変化（スロープ変化）

### 例：係数の解釈
```
const = 100        → 介入前の初期値は100
t = 2.0           → 介入前は毎期2ずつ増加
D = 15.0          → 介入により水準が15上昇
time_after = -1.0 → 介入後のトレンドは毎期1減少（合計で毎期1増加に変化）
```

### 介入効果DataFrameの解釈

`calculate_intervention_effect()`メソッドは、期間別の集約結果を返します：

```python
effect_df = model.calculate_intervention_effect()
```

| Period | Actual_mean | Predicted_mean | Counterfactual_mean | Effect_mean |
|--------|-------------|----------------|---------------------|-------------|
| Pre-intervention | 119.48 | 123.37 | 123.37 | -3.89 |
| Intervention_D_1 | 123.02 | 124.07 | 120.53 | 2.49 |
| Intervention_D_2 | 114.46 | 114.01 | 114.76 | -0.30 |
| Intervention_D_3 | 96.15 | 92.47 | 99.46 | -3.31 |

- **Period**: 介入前（Pre-intervention）または介入期間（Intervention_D_1, D_2, D_3...）
- **Actual_mean**: 実際の観測値の平均
- **Predicted_mean**: モデルによる予測値の平均
- **Counterfactual_mean**: 介入がなかった場合の推定値の平均
- **Effect_mean**: 介入効果（Actual - Counterfactual）の平均

## 🤝 貢献

バグ報告、機能要望、プルリクエストは歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 📞 サポート

質問やサポートが必要な場合は、GitHubのIssueを作成してください。

---

**Note**: このパッケージは元の`ITAS_mock.py`を抽象化・拡張したものです。元のファイルとの互換性を保ちつつ、より柔軟で再利用可能な設計になっています。