"""
ITS分析結果を学術的なMarkdownレポートに出力するプログラム

run.pyで保存されたモデルを読み込み、包括的な分析レポートを生成します。
"""


import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# モジュールのインポートパス設定
sys.path.append(str(Path(__file__).parent))
from module.its_analysis import (
    ITSDataPreprocessor,
    ITSModelOLS,
    ITSModelProphet,
    ITSModelSARIMAX,
    ITSVisualizer
)

warnings.filterwarnings('ignore')


def load_model(filename: str):
    """
    保存されたモデルをpklファイルから読み込み

    Args:
        filename: 読み込むファイル名

    Returns:
        読み込まれたモデルオブジェクト、またはNone（ファイルが存在しない場合）
    """
    # プロジェクトルートディレクトリを取得（src/moduleから2階層上）
    project_root = Path(__file__).parent.parent
    print(f"プロジェクトルート: {project_root}")
    filepath = project_root / 'models' / filename
    print(f"読み込みファイルパス: {filepath}")
    if not filepath.exists():
        print(f"警告: {filepath} が見つかりません。")
        return None

    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print(f"モデルを読み込みました: {filepath}")
    return model


def extract_coefficients_table(model, model_name: str) -> pd.DataFrame:
    """
    モデルから介入効果の回帰係数と標準誤差を抽出

    Args:
        model: ITSモデルオブジェクト
        model_name: モデル名（'OLS', 'SARIMAX', 'Prophet'）

    Returns:
        pandas.DataFrame: 回帰係数と標準誤差のテーブル
    """
    if model is None or model.model_results is None:
        return pd.DataFrame()

    # OLSとSARIMAXの場合
    if model_name in ['OLS', 'SARIMAX']:
        try:
            # model_resultsからパラメータと標準誤差を取得
            if hasattr(model, 'model_results'):
                params = model.model_results.params
                bse = model.model_results.bse  # 標準誤差
            else:
                # fallback: modelオブジェクト直下から取得を試行
                params = model.model.params if hasattr(
                    model.model, 'params') else {}
                bse = model.model.bse if hasattr(model.model, 'bse') else {}

            # D_1, D_2, D_3などの介入効果パラメータを抽出
            intervention_params = {k: v for k, v in params.items()
                                   if k.startswith('D_') or k.startswith('timedelta_')}
            intervention_bse = {k: v for k, v in bse.items()
                                if k.startswith('D_') or k.startswith('timedelta_')}

            # データフレームに整形
            df = pd.DataFrame({
                'Parameter': list(intervention_params.keys()),
                'Coefficient': list(intervention_params.values()),
                'Std Error': [intervention_bse[k] for k in intervention_params.keys()]
            })

            return df
        except Exception as e:
            print(f"警告: {model_name}モデルからパラメータ抽出に失敗: {e}")
            return pd.DataFrame()

    # Prophetの場合
    elif model_name == 'Prophet':
        try:
            # Prophetモデルの介入効果を抽出
            # modelオブジェクトから介入効果の情報を取得
            if hasattr(model, 'intervention_points'):
                # 介入効果の計算
                effect_df = model.calculate_intervention_effect()
                return effect_df
        except Exception as e:
            print(f"警告: Prophetモデルからパラメータ抽出に失敗: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


def generate_model_specification_section() -> str:
    """
    モデル特定化セクションを生成

    Returns:
        str: モデル特定化セクションのMarkdownテキスト
    """
    section = """
## 2. モデル特定化 (Model Specification)

本分析では、介入効果を推定するために3つの異なるモデルアプローチを使用しました。各モデルは異なる前提条件と特性を持ち、それぞれの視点から介入効果を評価します。

### 2.1 OLS (Ordinary Least Squares) モデル

**モデル仕様:**

OLSモデルは、時系列の自己相関を考慮しない最も基本的な回帰モデルです。以下のように定式化されます：

$$
Y_t = \\beta_0 + \\beta_1 t + \\sum_{i=1}^{k} \\delta_i D_{i,t} + \\sum_{i=1}^{k} \\gamma_i (t - T_i) \\cdot D_{i,t} + \\epsilon_t
$$

ここで、
- $Y_t$: 時点 $t$ における観測値
- $t$: 時間トレンド
- $D_{i,t}$: 介入 $i$ のダミー変数（介入後は1、それ以前は0）
- $T_i$: 介入 $i$ が発生した時点
- $\\delta_i$: 介入 $i$ の即時効果（レベルシフト）
- $\\gamma_i$: 介入 $i$ 後のトレンド変化
- $\\epsilon_t$: 誤差項

**前提条件:**
- 誤差項は独立同分布（i.i.d.）
- 自己相関が存在しない
- 分散均一性

**特徴:**
- 解釈が容易で、パラメータの意味が直感的
- 計算が高速
- 自己相関が存在する場合、標準誤差が過小評価される可能性

### 2.2 SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) モデル

**モデル仕様:**

SARIMAXモデルは、時系列の自己相関と季節性を明示的にモデル化します：

$$
\\Phi_P(B^s) \\phi_p(B) \\nabla^d \\nabla_s^D Y_t = \\Theta_Q(B^s) \\theta_q(B) \\epsilon_t + \\sum_{i=1}^{k} \\delta_i D_{i,t}
$$

ここで、
- $\\phi_p(B)$: 非季節的AR多項式（次数 $p$）
- $\\Phi_P(B^s)$: 季節的AR多項式（次数 $P$）
- $\\theta_q(B)$: 非季節的MA多項式（次数 $q$）
- $\\Theta_Q(B^s)$: 季節的MA多項式（次数 $Q$）
- $\\nabla^d$: 次数 $d$ の差分演算子
- $\\nabla_s^D$: 季節次数 $D$ の差分演算子
- $B$: バックシフト演算子
- $s$: 季節周期

**前提条件:**
- 時系列は（差分後に）定常
- 誤差項は正規分布に従う
- 自己相関構造がARMA過程で表現可能

**特徴:**
- 自己相関を明示的にモデル化するため、より正確な標準誤差が得られる
- 季節性パターンを捉えることができる
- パラメータの最適化にOptunaを使用し、AICを最小化

### 2.3 Prophet モデル

**モデル仕様:**

Prophetは、トレンド、季節性、イベント効果を加法的に分解する時系列モデルです：

$$
Y_t = g(t) + s(t) + h(t) + \\sum_{i=1}^{k} \\delta_i I(t \\geq T_i) + \\epsilon_t
$$

ここで、
- $g(t)$: トレンド成分（区分的線形または区分的ロジスティック）
- $s(t)$: 季節性成分（フーリエ級数で表現）
- $h(t)$: 休日・イベント効果
- $I(t \\geq T_i)$: 介入 $i$ の指示関数
- $\\epsilon_t$: 誤差項（正規分布）

**前提条件:**
- トレンドは区分的線形で表現可能
- 季節性はフーリエ級数で近似可能
- 変化点は自動的に検出可能

**特徴:**
- 外れ値や欠損値に頑健
- トレンドの変化点を柔軟に捉える
- ベイズ的アプローチにより不確実性を定量化
- ハイパーパラメータの最適化にOptunaを使用

### 2.4 モデル間の効果量の違いについて

3つのモデルで推定される介入効果は、以下の理由で異なる可能性があります：

1. **自己相関の扱い**: OLSは自己相関を無視するため、SARIMAXやProphetと比較して効果の推定値が異なる場合があります。

2. **季節性の調整**: SARIMAXとProphetは季節性を明示的にモデル化しますが、OLSは含めません（または外生変数として追加が必要）。

3. **トレンドの柔軟性**: Prophetは変化点を自動検出し、非線形トレンドを捉えることができますが、OLSとSARIMAXは線形トレンドを仮定します。

4. **不確実性の評価**: Prophetはベイズ的アプローチで不確実性を評価し、SARIMAXは最尤法、OLSは最小二乗法を使用します。

これらの違いにより、各モデルは異なる視点から介入効果を評価し、結果の頑健性を多角的に検証することができます。
"""
    return section


def generate_markdown_report(output_path='output/analysis_report.md'):
    """
    ITS分析結果のMarkdownレポートを生成

    Args:
        output_path: 出力ファイルパス（プロジェクトルートからの相対パス）

    Returns:
        str: 生成されたレポートファイルの絶対パス
    """
    # プロジェクトルートの取得
    project_root = Path(__file__).parent.parent
    output_path = project_root / output_path

    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # モデルの読み込み
    print("\n" + "="*80)
    print("モデルを読み込んでいます...")
    print("="*80)

    model_ols = load_model('its_model_ols.pkl')
    model_sarimax = load_model('its_model_sarimax_tuned.pkl')
    model_prophet = load_model('its_model_prophet_tuned.pkl')

    # レポート生成
    print("\nレポートを生成しています...")

    with open(output_path, 'w', encoding='utf-8') as f:
        # タイトルと日付
        f.write("# 断続的時系列分析レポート (Interrupted Time Series Analysis Report)\n\n")
        f.write(
            f"**生成日時 (Generated):** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")

        # 1. 概要
        f.write("## 1. 概要 (Executive Summary)\n\n")
        f.write("本レポートは、断続的時系列分析（Interrupted Time Series Analysis, ITS）を用いて、")
        f.write("複数の介入が観測値に与える効果を評価したものです。")
        f.write("3つの異なるモデリングアプローチ（OLS、SARIMAX、Prophet）を使用し、")
        f.write("介入効果の頑健性を多角的に検証しました。\n\n")

        # 2. モデル特定化
        f.write(generate_model_specification_section())
        f.write("\n")

        # 3. 回帰係数と標準誤差の表
        f.write("## 3. 介入効果の推定結果 (Intervention Effect Estimates)\n\n")
        f.write("以下の表は、各モデルで推定された介入効果（回帰係数）と標準誤差を示しています。\n\n")

        # OLS
        f.write("### 3.1 OLS Model\n\n")
        if model_ols:
            df_ols = extract_coefficients_table(model_ols, 'OLS')
            if not df_ols.empty:
                f.write(df_ols.to_markdown(index=False, floatfmt='.4f'))
                f.write("\n\n")
            else:
                f.write("データを抽出できませんでした。\n\n")
        else:
            f.write("モデルが読み込まれませんでした。\n\n")

        # SARIMAX
        f.write("### 3.2 SARIMAX Model (Optuna-tuned)\n\n")
        if model_sarimax:
            df_sarimax = extract_coefficients_table(model_sarimax, 'SARIMAX')
            if not df_sarimax.empty:
                f.write(df_sarimax.to_markdown(index=False, floatfmt='.4f'))
                f.write("\n\n")
            else:
                f.write("データを抽出できませんでした。\n\n")
        else:
            f.write("モデルが読み込まれませんでした。\n\n")

        # Prophet
        f.write("### 3.3 Prophet Model (Optuna-tuned)\n\n")
        f.write("Prophetは明示的に係数を抽出できないので、予測結果をもとに介入効果を評価します。\n\n")

        if model_prophet:
            df_prophet = extract_coefficients_table(model_prophet, 'Prophet')
            if not df_prophet.empty:
                f.write(df_prophet.to_markdown(index=False, floatfmt='.4f'))
                f.write("\n\n")
            else:
                f.write("データを抽出できませんでした。\n\n")
        else:
            f.write("モデルが読み込まれませんでした。\n\n")

        # 4. 可視化
        f.write("## 4. 可視化 (Visualization)\n\n")
        f.write("各モデルによる介入効果の可視化結果は以下のファイルに保存されています：\n\n")
        f.write("### 4.1 OLS Model Analysis\n")
        f.write("![OLS Analysis](./report_ols_multiple_interventions.png)\n\n")

        f.write("### 4.2 SARIMAX Model Analysis\n")
        f.write("![SARIMAX Analysis](./report_sarimax.png)\n\n")

        f.write("### 4.3 Prophet Model Analysis\n")
        f.write("![Prophet Analysis](./report_prophet.png)\n\n")

        # 5. 結論
        f.write("## 5. 結論 (Conclusion)\n\n")
        f.write("本分析では、3つの異なるモデリングアプローチを用いて介入効果を推定しました。")
        f.write("各モデルの結果を比較することで、推定値の頑健性と信頼性を評価することができます。\n\n")

        # 6. Discussion（空欄）
        f.write("## 6. 考察 (Discussion)\n\n")
        f.write("<!-- ここに考察を記入してください -->\n\n")
        f.write("<!-- 以下の観点から分析結果を考察することを推奨します：\n")
        f.write("- 各モデルの推定結果の一致度\n")
        f.write("- 介入効果の統計的有意性\n")
        f.write("- モデル間で結果が異なる場合の解釈\n")
        f.write("- 実務的な含意と推奨事項\n")
        f.write("-->\n\n")

        # 7. 参考文献
        f.write("## 7. 参考文献 (References)\n\n")
        f.write(
            "<!--レポート作成者により適宜追加してください。-->\n\n")

    print(f"\nレポートを生成しました: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    report_path = generate_markdown_report()
    if report_path:
        print(f"\nレポートファイル: {report_path}")
        print("レポートをMarkdownビューアーで確認してください。")
