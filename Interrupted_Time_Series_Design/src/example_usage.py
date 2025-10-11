"""
新しいits_analysis.pyの使用例

複数介入、複数モデル（OLS, SARIMAX, Prophet）の使用方法を示します。
"""

from module.its_analysis import (
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
import sys
from pathlib import Path

# モジュールのインポート
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Cigarデータセットの読み込み
cigar = sm.datasets.get_rdataset("Cigar", "Ecdat").data

# 使用するデータの準備
state = [3, 5]
timestamp = [75, 80, 85]

# サブセットを作成
usecols = ['state', 'year', 'price', 'pop', 'sales']
cigar_model = cigar.loc[(cigar['state'].isin(state)) &
                        (cigar['year'] >= 65), usecols].copy()

print("=" * 80)
print("新しいits_analysis.pyの使用例")
print("=" * 80)

# OLSモデルの例
print("\n1. OLSモデルによる複数介入分析")
print("-" * 80)

model_ols = ITSModelOLS(
    time_column='year',
    intervention_points=timestamp,
    group_column='state'
)

# モデルのフィッティング
model_ols.fit(cigar_model, target_column='sales', covariates=['price', 'pop'])

# 介入効果のDataFrame出力
effect_df = model_ols.calculate_intervention_effect()
print("\n介入効果DataFrame（最初の10行）:")
print(effect_df.head(10))

print("\n各期間の平均効果:")
for period in effect_df['Period'].unique():
    if period != 'Pre-intervention':
        period_data = effect_df[effect_df['Period'] == period]
        if 'state' in period_data.columns:
            # state別に集計
            for state in period_data['state'].unique():
                state_data = period_data[period_data['state'] == state]
                print(f"  state={state}, {period}: Actual_mean={state_data['Actual_mean'].values[0]:.2f}, "
                      f"Counterfactual_mean={state_data['Counterfactual_mean'].values[0]:.2f}, "
                      f"Effect_mean={state_data['Effect_mean'].values[0]:.2f}")
        else:
            print(f"  {period}: Actual_mean={period_data['Actual_mean'].values[0]:.2f}, "
                  f"Counterfactual_mean={period_data['Counterfactual_mean'].values[0]:.2f}, "
                  f"Effect_mean={period_data['Effect_mean'].values[0]:.2f}")

# 可視化
visualizer = ITSVisualizer(model_ols)
fig = visualizer.plot(
    save_path='output/example_ols_multiple_interventions.png')
print("\n可視化を保存しました: output/example_ols_multiple_interventions.png")
plt.close()

# SARIMAXモデルの例（単一グループ）
print("\n2. SARIMAXモデルによる分析（state=3のみ）")
print("-" * 80)

cigar_single = cigar_model[cigar_model['state'] == 3].copy()

model_sarimax = ITSModelSARIMAX(
    time_column='year',
    intervention_points=timestamp
)

# モデルのフィッティング（Optunaチューニングは時間がかかるのでスキップ）
model_sarimax.fit(cigar_single, target_column='sales', order=(1, 0, 1))

# 介入効果
effect_df_sarimax = model_sarimax.calculate_intervention_effect()
print("\n介入効果DataFrame（最初の5行）:")
print(effect_df_sarimax.head())

# 可視化
visualizer_sarimax = ITSVisualizer(model_sarimax)
fig = visualizer_sarimax.plot(save_path='output/example_sarimax.png')
print("\n可視化を保存しました: output/example_sarimax.png")
plt.close()

# Prophetモデルの例（単一グループ）
print("\n3. Prophetモデルによる分析（state=3のみ）")
print("-" * 80)

model_prophet = ITSModelProphet(
    time_column='year',
    intervention_points=timestamp
)

# モデルのフィッティング
model_prophet.fit(cigar_single, target_column='sales')

# 介入効果
effect_df_prophet = model_prophet.calculate_intervention_effect()
print("\n介入効果DataFrame（最初の5行）:")
print(effect_df_prophet.head())

# 可視化
visualizer_prophet = ITSVisualizer(model_prophet)
fig = visualizer_prophet.plot(save_path='output/example_prophet.png')
print("\n可視化を保存しました: output/example_prophet.png")
plt.close()

print("\n" + "=" * 80)
print("全ての例が完了しました！")
print("=" * 80)
