"""
ITS分析モデルの実行とモデル保存

Optunaによるハイパーパラメータチューニングを行い、
最適化されたモデルをpklファイルとして保存します。
"""


import pandas as pd
import statsmodels.api as sm
import pickle
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# モジュールのインポートパス設定
sys.path.insert(0, str(Path(__file__).parent))
from module.its_analysis import (
    ITSModelOLS,
    ITSModelSARIMAX,
    ITSModelProphet,
    ITSVisualizer
)

def save_model(model, filename):
    """
    モデルをpklファイルとして保存

    Args:
        model: 保存するモデルオブジェクト
        filename: 保存するファイル名
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)

    filepath = models_dir / filename

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"モデルを保存しました: {filepath}")
    return filepath


def run_analysis():
    """
    ITS分析を実行し、モデルを保存
    """
    print("=" * 80)
    print("ITS分析モデルの実行とモデル保存")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # データの準備
    print("\n[1/4] データの準備中...")
    cigar = sm.datasets.get_rdataset("Cigar", "Ecdat").data

    state = [3]
    timestamp = [75, 80, 85]

    usecols = ['state', 'year', 'price', 'pop', 'sales']
    cigar_model = cigar.loc[(cigar['state'].isin(state)) &
                            (cigar['year'] >= 65), usecols].copy()

    cigar_single = cigar_model[cigar_model['state'] == 3].copy()

    print(f"  - 全データ: {cigar_model.shape}")
    print(f"  - state=3のみ: {cigar_single.shape}")
    print(f"  - 介入ポイント: {timestamp}")

    # OLSモデルの実行
    print("\n[2/4] OLSモデルの実行中...")
    model_ols = ITSModelOLS(
        time_column='year',
        intervention_points=timestamp,
        group_column='state'
    )

    model_ols.fit(cigar_model, target_column='sales',
                  covariates=['price', 'pop'])
    print("  - OLSモデルのフィッティング完了")

    # OLSモデルの可視化
    visualizer = ITSVisualizer(model_ols)
    fig = visualizer.plot(save_path='output/report_ols_multiple_interventions.png')
    plt.close()

    # OLSモデルの保存
    save_model(model_ols, 'its_model_ols.pkl')

    # SARIMAXモデルの実行（Optunaチューニング付き）
    if Path('models/its_model_sarimax_tuned.pkl').exists():
        print("\n[3/4] SARIMAXモデルの保存済みファイルが存在します。スキップします。")
        pass
    else:
      print("\n[3/4] SARIMAXモデルの実行中（Optunaチューニング有効）...")
      print("  ※ チューニングには数分かかる場合があります")

      model_sarimax = ITSModelSARIMAX(
          time_column='year',
          intervention_points=timestamp
      )

      # Optunaチューニングを有効にして実行
      model_sarimax.fit(
          cigar_single,
          target_column='sales',
          tune_with_optuna=True,  # Optunaチューニングを有効化
          n_trials=50     # 試行回数

      )
      print("  - SARIMAXモデルのフィッティング完了")
      # SARIMAXモデルの可視化
      visualizer = ITSVisualizer(model_sarimax)
      fig = visualizer.plot(save_path='output/report_sarimax.png')
      plt.close()
      # SARIMAXモデルの保存
      save_model(model_sarimax, 'its_model_sarimax_tuned.pkl')

    # Prophetモデルの実行（Optunaチューニング付き）
    if Path('models/its_model_prophet_tuned.pkl').exists():
        print("\n[4/4] Prophetモデルの保存済みファイルが存在します。スキップします。")
        pass
    else:
      print("\n[4/4] Prophetモデルの実行中（Optunaチューニング有効）...")
      print("  ※ チューニングには数分かかる場合があります")

      model_prophet = ITSModelProphet(
          time_column='year',
          intervention_points=timestamp
      )

      # Optunaチューニングを有効にして実行
      model_prophet.fit(
          cigar_single,
          target_column='sales',
          tune_with_optuna=True,  # Optunaチューニングを有効化
          n_trials=50,      # 試行回数
      )
      print("  - Prophetモデルのフィッティング完了")
      # Prophetモデルの可視化
      visualizer = ITSVisualizer(model_prophet)
      fig = visualizer.plot(save_path='output/report_prophet.png')
      plt.close()
      # Prophetモデルの保存
      save_model(model_prophet, 'its_model_prophet_tuned.pkl')

    print("\n" + "=" * 80)
    print("すべてのモデルの実行と保存が完了しました")
    print("=" * 80)
    print("\n保存されたモデル:")
    print("  1. models/its_model_ols.pkl")
    print("  2. models/its_model_sarimax_tuned.pkl")
    print("  3. models/its_model_prophet_tuned.pkl")
    print("\nこれらのモデルは generate_report.py で使用されます。")


if __name__ == "__main__":
    run_analysis()
