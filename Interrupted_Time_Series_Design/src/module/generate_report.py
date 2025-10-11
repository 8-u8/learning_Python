"""
ITS分析結果をMarkdownレポートに出力するプログラム

example_usage.pyの実行結果をMarkdown形式でレポート化します。
"""

from .its_analysis import (
    ITSDataPreprocessor,
    ITSModelOLS,
    ITSModelSARIMAX,
    ITSModelProphet,
    ITSVisualizer
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime


def generate_markdown_report(output_path='output/analysis_report.md'):
    """
    ITS分析結果のMarkdownレポートを生成

    Args:
        output_path (str): レポートの保存先パス（プロジェクトルートからの相対パス）
    """
    # プロジェクトルートディレクトリを取得（src/moduleから2階層上）
    project_root = Path(__file__).parent.parent.parent

    # 絶対パスに変換
    output_path = project_root / output_path

    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cigarデータセットの読み込み
    cigar = sm.datasets.get_rdataset("Cigar", "Ecdat").data

    # 使用するデータの準備
    state = [3, 5]
    timestamp = [75, 80, 85]

    # サブセットを作成
    usecols = ['state', 'year', 'price', 'pop', 'sales']
    cigar_model = cigar.loc[(cigar['state'].isin(state)) & (
        cigar['year'] >= 65), usecols].copy()

    # Markdownレポートの生成
    with open(output_path, 'w', encoding='utf-8') as f:
        # ヘッダー
        f.write("# Interrupted Time Series Analysis Report\n\n")
        f.write(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # データ概要
        f.write("## 1. Data Overview\n\n")
        f.write("### Dataset\n")
        f.write("- **Source:** Cigar dataset from Ecdat package\n")
        f.write(f"- **States analyzed:** {state}\n")
        f.write(f"- **Intervention points:** {timestamp}\n")
        f.write(
            f"- **Time period:** {cigar_model['year'].min()} - {cigar_model['year'].max()}\n")
        f.write(f"- **Total observations:** {len(cigar_model)}\n\n")

        f.write("### Variables\n")
        f.write("- `year`: Year of observation\n")
        f.write("- `sales`: Cigarette sales (outcome variable)\n")
        f.write("- `price`: Cigarette price (covariate)\n")
        f.write("- `pop`: Population (covariate)\n")
        f.write("- `state`: State identifier\n\n")

        f.write("---\n\n")

        # OLSモデル
        f.write("## 2. OLS Model Analysis (Multiple Interventions)\n\n")

        model_ols = ITSModelOLS(
            time_column='year',
            intervention_points=timestamp,
            group_column='state'
        )

        model_ols.fit(cigar_model, target_column='sales',
                      covariates=['price', 'pop'])
        effect_df_ols = model_ols.calculate_intervention_effect()

        f.write("### Model Specification\n")
        f.write("- **Model type:** Ordinary Least Squares (OLS)\n")
        f.write("- **Covariates:** price, pop\n")
        f.write("- **Group variable:** state\n")
        f.write(f"- **Number of interventions:** {len(timestamp)}\n\n")

        f.write("### Intervention Effects Summary\n\n")
        f.write(effect_df_ols.to_markdown(index=False))
        f.write("\n\n")

        # 可視化の保存とリンク
        visualizer_ols = ITSVisualizer(model_ols)
        ols_plot_path = project_root / 'output/report_ols_multiple_interventions.png'
        fig = visualizer_ols.plot(save_path=str(ols_plot_path))
        plt.close()

        f.write("### Visualization\n\n")
        f.write(
            "![OLS Model Visualization](report_ols_multiple_interventions.png)\n\n")

        f.write("### Key Findings (OLS)\n")
        for _, row in effect_df_ols.iterrows():
            if 'state' in row:
                f.write(f"- **State {row['state']}, {row['Period']}:** ")
            else:
                f.write(f"- **{row['Period']}:** ")
            f.write(f"Actual mean = {row['Actual_mean']:.2f}, ")
            f.write(
                f"Counterfactual mean = {row['Counterfactual_mean']:.2f}, ")
            f.write(f"Effect = {row['Effect_mean']:.2f}\n")
        f.write("\n")

        f.write("---\n\n")

        # SARIMAXモデル
        f.write("## 3. SARIMAX Model Analysis (State 3 Only)\n\n")

        cigar_single = cigar_model[cigar_model['state'] == 3].copy()

        model_sarimax = ITSModelSARIMAX(
            time_column='year',
            intervention_points=timestamp
        )

        model_sarimax.fit(cigar_single, target_column='sales', order=(1, 0, 1))
        effect_df_sarimax = model_sarimax.calculate_intervention_effect()

        f.write("### Model Specification\n")
        f.write("- **Model type:** SARIMAX\n")
        f.write("- **ARIMA order:** (1, 0, 1)\n")
        f.write("- **Seasonal order:** (0, 0, 0, 0)\n")
        f.write(f"- **Number of interventions:** {len(timestamp)}\n\n")

        f.write("### Intervention Effects Summary\n\n")
        f.write(effect_df_sarimax.to_markdown(index=False))
        f.write("\n\n")

        # 可視化の保存とリンク
        visualizer_sarimax = ITSVisualizer(model_sarimax)
        sarimax_plot_path = project_root / 'output/report_sarimax.png'
        fig = visualizer_sarimax.plot(save_path=str(sarimax_plot_path))
        plt.close()

        f.write("### Visualization\n\n")
        f.write("![SARIMAX Model Visualization](report_sarimax.png)\n\n")

        f.write("### Key Findings (SARIMAX)\n")
        for _, row in effect_df_sarimax.iterrows():
            f.write(f"- **{row['Period']}:** ")
            f.write(f"Actual mean = {row['Actual_mean']:.2f}, ")
            f.write(
                f"Counterfactual mean = {row['Counterfactual_mean']:.2f}, ")
            f.write(f"Effect = {row['Effect_mean']:.2f}\n")
        f.write("\n")

        f.write("---\n\n")

        # Prophetモデル
        f.write("## 4. Prophet Model Analysis (State 3 Only)\n\n")

        model_prophet = ITSModelProphet(
            time_column='year',
            intervention_points=timestamp
        )

        model_prophet.fit(cigar_single, target_column='sales')
        effect_df_prophet = model_prophet.calculate_intervention_effect()

        f.write("### Model Specification\n")
        f.write("- **Model type:** Facebook Prophet\n")
        f.write("- **Changepoint prior scale:** 0.05 (default)\n")
        f.write("- **Seasonality prior scale:** 10.0 (default)\n")
        f.write(f"- **Number of interventions:** {len(timestamp)}\n\n")

        f.write("### Intervention Effects Summary\n\n")
        f.write(effect_df_prophet.to_markdown(index=False))
        f.write("\n\n")

        # 可視化の保存とリンク
        visualizer_prophet = ITSVisualizer(model_prophet)
        prophet_plot_path = project_root / 'output/report_prophet.png'
        fig = visualizer_prophet.plot(save_path=str(prophet_plot_path))
        plt.close()

        f.write("### Visualization\n\n")
        f.write("![Prophet Model Visualization](report_prophet.png)\n\n")

        f.write("### Key Findings (Prophet)\n")
        for _, row in effect_df_prophet.iterrows():
            f.write(f"- **{row['Period']}:** ")
            f.write(f"Actual mean = {row['Actual_mean']:.2f}, ")
            f.write(
                f"Counterfactual mean = {row['Counterfactual_mean']:.2f}, ")
            f.write(f"Effect = {row['Effect_mean']:.2f}\n")
        f.write("\n")

        f.write("---\n\n")

        # モデル比較
        f.write("## 5. Model Comparison\n\n")
        f.write("### Pre-intervention Period\n\n")
        f.write("| Model | Actual Mean | Counterfactual Mean | Effect Mean |\n")
        f.write("|-------|-------------|---------------------|-------------|\n")

        # OLS (State 3のみ抽出)
        ols_pre = effect_df_ols[(effect_df_ols['state'] == 3) & (
            effect_df_ols['Period'] == 'Pre-intervention')]
        if len(ols_pre) > 0:
            f.write(
                f"| OLS | {ols_pre['Actual_mean'].values[0]:.2f} | {ols_pre['Counterfactual_mean'].values[0]:.2f} | {ols_pre['Effect_mean'].values[0]:.2f} |\n")

        # SARIMAX
        sarimax_pre = effect_df_sarimax[effect_df_sarimax['Period']
                                        == 'Pre-intervention']
        if len(sarimax_pre) > 0:
            f.write(
                f"| SARIMAX | {sarimax_pre['Actual_mean'].values[0]:.2f} | {sarimax_pre['Counterfactual_mean'].values[0]:.2f} | {sarimax_pre['Effect_mean'].values[0]:.2f} |\n")

        # Prophet
        prophet_pre = effect_df_prophet[effect_df_prophet['Period']
                                        == 'Pre-intervention']
        if len(prophet_pre) > 0:
            f.write(
                f"| Prophet | {prophet_pre['Actual_mean'].values[0]:.2f} | {prophet_pre['Counterfactual_mean'].values[0]:.2f} | {prophet_pre['Effect_mean'].values[0]:.2f} |\n")

        f.write("\n### Post-intervention Periods\n\n")

        # 各介入期間について比較
        for i in range(len(timestamp)):
            period_name = f"Intervention_D_{i+1}"
            f.write(f"\n#### {period_name}\n\n")
            f.write("| Model | Actual Mean | Counterfactual Mean | Effect Mean |\n")
            f.write("|-------|-------------|---------------------|-------------|\n")

            # OLS
            ols_post = effect_df_ols[(effect_df_ols['state'] == 3) & (
                effect_df_ols['Period'] == period_name)]
            if len(ols_post) > 0:
                f.write(
                    f"| OLS | {ols_post['Actual_mean'].values[0]:.2f} | {ols_post['Counterfactual_mean'].values[0]:.2f} | {ols_post['Effect_mean'].values[0]:.2f} |\n")

            # SARIMAX
            sarimax_post = effect_df_sarimax[effect_df_sarimax['Period']
                                             == period_name]
            if len(sarimax_post) > 0:
                f.write(
                    f"| SARIMAX | {sarimax_post['Actual_mean'].values[0]:.2f} | {sarimax_post['Counterfactual_mean'].values[0]:.2f} | {sarimax_post['Effect_mean'].values[0]:.2f} |\n")

            # Prophet
            prophet_post = effect_df_prophet[effect_df_prophet['Period']
                                             == period_name]
            if len(prophet_post) > 0:
                f.write(
                    f"| Prophet | {prophet_post['Actual_mean'].values[0]:.2f} | {prophet_post['Counterfactual_mean'].values[0]:.2f} | {prophet_post['Effect_mean'].values[0]:.2f} |\n")

        f.write("\n---\n\n")

        # 結論
        f.write("## 6. Conclusions\n\n")
        f.write("### Summary of Intervention Effects\n\n")
        f.write("This analysis examined the impact of multiple interventions on cigarette sales using three different statistical models:\n\n")
        f.write(
            "1. **OLS Model**: Provides a baseline linear regression approach with group effects\n")
        f.write(
            "2. **SARIMAX Model**: Accounts for time series autocorrelation and moving average components\n")
        f.write(
            "3. **Prophet Model**: Captures trend and seasonal patterns with flexible changepoints\n\n")

        f.write("### Key Insights\n\n")
        f.write(
            "- All models show the intervention effects across multiple time periods\n")
        f.write(
            "- The counterfactual estimates represent what would have occurred without the interventions\n")
        f.write("- Effect sizes vary by model specification and assumptions\n")
        f.write("- Model comparison helps assess robustness of findings\n\n")

        f.write("### Recommendations\n\n")
        f.write("- Consider ensemble approaches combining multiple models\n")
        f.write("- Perform sensitivity analysis on key model parameters\n")
        f.write("- Validate results with additional robustness checks\n")
        f.write("- Use Optuna hyperparameter tuning for optimal model performance\n\n")

        f.write("---\n\n")
        f.write("*Report generated by ITS Analysis Package*\n")

    print(f"\n✅ Markdownレポートを生成しました: {output_path}")
    return output_path


if __name__ == "__main__":
    report_path = generate_markdown_report()
    print(f"\n📊 レポートファイル: {report_path}")
    print("🔍 レポートをMarkdownビューアーで確認してください！")
