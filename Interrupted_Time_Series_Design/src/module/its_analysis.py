"""
Interrupted Time Series Analysis (ITAS) Package

このパッケージは、Interrupted Time Series Designを使った分析を
抽象化し、任意のデータに対して適用可能なクラスを提供します。

Classes:
    ITSDataPreprocessor: ITSに必要な変数を生成するクラス（複数介入対応）
    ITSModelBase: モデルの基底クラス
    ITSModelOLS: OLSモデルを実行・管理するクラス
    ITSModelSARIMAX: SARIMAXモデルを実行・管理するクラス
    ITSModelProphet: Prophetモデルを実行・管理するクラス
    ITSVisualizer: 結果を可視化するクラス
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pickle
import optuna
from prophet import Prophet
from typing import Optional, List, Dict, Union, Any, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')


class ITSDataPreprocessor:
    """
    Interrupted Time Series Designに必要な変数を生成するクラス（複数介入対応）

    任意のデータから以下の変数を生成します：
    - t: 時系列の起点から終点までのカウントアップ
    - D_1, D_2, ..., D_P: 各介入期間のダミー変数
    - timedelta_1, timedelta_2, ..., timedelta_P: 各介入後の経過時間
    - const: 切片項、全期間を通して1となる変数
    """

    def __init__(self,
                 time_column: str,
                 intervention_points: List[Union[int, float]],
                 group_column: Optional[str] = None):
        """
        ITSDataPreprocessorの初期化

        Args:
            time_column (str): 時間を表すカラム名
            intervention_points (List[Union[int, float]]): 介入が起こった時点のリスト（昇順でソート済み）
            group_column (Optional[str]): グループを表すカラム名（州や地域など）
        """
        self.time_column = time_column
        self.intervention_points = sorted(intervention_points)
        self.group_column = group_column
        self.processed_data: Optional[pd.DataFrame] = None
        self.n_interventions = len(self.intervention_points)

    def fit_transform(self,
                      df: pd.DataFrame,
                      add_intercept: bool = True,
                      time_start_from_one: bool = True) -> pd.DataFrame:
        """
        データにITS用の変数を追加して変換（複数介入対応）

        Args:
            df (pd.DataFrame): 入力データフレーム
            add_intercept (bool): 切片項を追加するかどうか
            time_start_from_one (bool): 時間変数を1から開始するかどうか

        Returns:
            pd.DataFrame: 変換後のデータフレーム
        """
        # データのコピーを作成
        df_out = df.copy()

        # 時間変数の作成 (t)
        if time_start_from_one:
            if self.group_column is not None:
                # グループ別に時間変数を作成
                df_out['t'] = (df_out.groupby(self.group_column)[self.time_column]
                               .transform(lambda x: x - x.min() + 1))
            else:
                # 全体で時間変数を作成
                min_time = df_out[self.time_column].min()
                df_out['t'] = df_out[self.time_column] - min_time + 1
        else:
            df_out['t'] = df_out[self.time_column]

        # 複数介入に対応したダミー変数の生成
        for i in range(self.n_interventions):
            intervention_point = self.intervention_points[i]

            # D_i: 介入期間のダミー変数
            if i < self.n_interventions - 1:
                # 次の介入点まで
                next_intervention = self.intervention_points[i + 1]
                df_out[f'D_{i+1}'] = (
                    (df_out[self.time_column] >= intervention_point) &
                    (df_out[self.time_column] < next_intervention)
                ).astype(int)
            else:
                # 最後の介入以降すべて
                df_out[f'D_{i+1}'] = (
                    df_out[self.time_column] >= intervention_point
                ).astype(int)

            # timedelta_i: 介入後経過時間（介入期間中のみカウント）
            if self.group_column is not None:
                # グループ別に計算
                df_out[f'timedelta_{i+1}'] = (
                    df_out.groupby(self.group_column)[
                        f'D_{i+1}'].transform('cumsum')
                )
            else:
                # 全体で計算
                df_out[f'timedelta_{i+1}'] = df_out[f'D_{i+1}'].cumsum()

            # 介入期間外は0にリセット
            df_out.loc[df_out[f'D_{i+1}'] != 1, f'timedelta_{i+1}'] = 0

        # 切片項の追加
        if add_intercept:
            df_out['const'] = 1

        self.processed_data = df_out
        return df_out

    def get_its_variables(self) -> List[str]:
        """
        生成したITS変数のリストを返す

        Returns:
            List[str]: ITS変数名のリスト
        """
        base_vars = ['t']

        # 複数介入対応
        for i in range(self.n_interventions):
            base_vars.append(f'D_{i+1}')
            base_vars.append(f'timedelta_{i+1}')

        if self.processed_data is not None and 'const' in self.processed_data.columns:
            base_vars.insert(0, 'const')

        return base_vars

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        入力データの妥当性をチェック

        Args:
            df (pd.DataFrame): チェックするデータフレーム

        Returns:
            bool: データが妥当かどうか

        Raises:
            ValueError: データが妥当でない場合
        """
        # 必要なカラムの存在チェック
        if self.time_column not in df.columns:
            raise ValueError(f"時間カラム '{self.time_column}' がデータに存在しません")

        if self.group_column is not None and self.group_column not in df.columns:
            raise ValueError(f"グループカラム '{self.group_column}' がデータに存在しません")

        # 介入点の範囲チェック（厳密な一致ではなく、範囲内にあるかチェック）
        time_min = df[self.time_column].min()
        time_max = df[self.time_column].max()
        for intervention_point in self.intervention_points:
            if not (time_min <= intervention_point <= time_max):
                raise ValueError(
                    f"介入点 '{intervention_point}' がデータの時間軸範囲外です（範囲: {time_min} ~ {time_max}）")

        return True


class ITSModelBase(ITSDataPreprocessor, ABC):
    """
    ITS モデルの抽象基底クラス

    全てのモデル（OLS, SARIMAX, Prophet）で共通のインターフェースを提供します。
    """

    def __init__(self,
                 time_column: str,
                 intervention_points: List[Union[int, float]],
                 group_column: Optional[str] = None):
        """
        ITSModelBaseの初期化

        Args:
            time_column (str): 時間を表すカラム名
            intervention_points (List[Union[int, float]]): 介入が起こった時点のリスト
            group_column (Optional[str]): グループを表すカラム名
        """
        super().__init__(time_column, intervention_points, group_column)
        self.model_results: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None
        self.target_column: Optional[str] = None
        self.model_type: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_column: str, **kwargs) -> Any:
        """
        モデルをフィッティング（抽象メソッド）

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数のカラム名
            **kwargs: モデル固有のパラメータ

        Returns:
            Any: フィット結果
        """
        pass

    @abstractmethod
    def predict(self, df: Optional[pd.DataFrame] = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        予測値を計算（抽象メソッド）

        Args:
            df (Optional[pd.DataFrame]): 予測用データ
            **kwargs: 予測時のオプション

        Returns:
            Union[pd.Series, pd.DataFrame]: 予測結果
        """
        pass

    def predict_counterfactual(self,
                               df: Optional[pd.DataFrame] = None,
                               return_ci: bool = True) -> Union[pd.Series, pd.DataFrame]:
        """
        反実仮想（介入がなかった場合）の予測値を計算

        Args:
            df (Optional[pd.DataFrame]): 予測用データ（Noneの場合は学習データを使用）
            return_ci (bool): 信頼区間を返すかどうか

        Returns:
            Union[pd.Series, pd.DataFrame]: 反実仮想予測結果
        """
        if df is None:
            if self.processed_data is None:
                raise ValueError("学習データが保存されていません。")
            counterfactual_data = self.processed_data.copy()
        else:
            counterfactual_data = self.fit_transform(df)

        # すべての介入変数を0にセット
        for i in range(self.n_interventions):
            counterfactual_data[f'D_{i+1}'] = 0
            counterfactual_data[f'timedelta_{i+1}'] = 0

        return self.predict(counterfactual_data, return_ci=return_ci, use_preprocessed=True)

    def calculate_intervention_effect(self) -> pd.DataFrame:
        """
        介入効果をDataFrame形式で計算（集計値形式）

        Returns:
            pd.DataFrame: 介入効果の集計（Period, Actual_mean, Predicted_mean, Counterfactual_mean, Effect_mean）
        """
        if self.processed_data is None or self.model_results is None:
            raise ValueError("モデルがフィットされていません。まずfitメソッドを実行してください。")

        df = self.processed_data.copy()

        # 実予測値
        pred_actual = self.predict(return_ci=False)
        if isinstance(pred_actual, pd.DataFrame):
            pred_actual_values = pred_actual['mean'].values
        elif isinstance(pred_actual, pd.Series):
            pred_actual_values = pred_actual.values
        else:
            pred_actual_values = pred_actual

        # 反実仮想予測値
        pred_counterfactual = self.predict_counterfactual(return_ci=False)
        if isinstance(pred_counterfactual, pd.DataFrame):
            pred_counterfactual_values = pred_counterfactual['mean'].values
        elif isinstance(pred_counterfactual, pd.Series):
            pred_counterfactual_values = pred_counterfactual.values
        else:
            pred_counterfactual_values = pred_counterfactual

        # 長さを合わせる
        n = len(df)
        if len(pred_actual_values) != n:
            # 予測値の長さが異なる場合、最初のn個を使用
            pred_actual_values = pred_actual_values[:n] if len(pred_actual_values) > n else np.pad(
                pred_actual_values, (0, n - len(pred_actual_values)), constant_values=np.nan)
        if len(pred_counterfactual_values) != n:
            pred_counterfactual_values = pred_counterfactual_values[:n] if len(pred_counterfactual_values) > n else np.pad(
                pred_counterfactual_values, (0, n - len(pred_counterfactual_values)), constant_values=np.nan)

        # 介入期間の判定（複数介入対応）
        period_labels = []
        for idx, time_val in enumerate(df[self.time_column]):
            if time_val < self.intervention_points[0]:
                period_labels.append('Pre-intervention')
            else:
                # どの介入期間に属するか判定
                for i, intervention_point in enumerate(self.intervention_points):
                    if i == len(self.intervention_points) - 1:
                        # 最後の介入以降
                        if time_val >= intervention_point:
                            period_labels.append(f'Intervention_D_{i+1}')
                            break
                    else:
                        # i番目の介入期間
                        if intervention_point <= time_val < self.intervention_points[i + 1]:
                            period_labels.append(f'Intervention_D_{i+1}')
                            break

        # 詳細DataFrameの作成
        detail_df = pd.DataFrame({
            'Period': period_labels,
            self.time_column: df[self.time_column].values,
            'Actual': df[self.target_column].values,
            'Predicted': pred_actual_values,
            'Counterfactual': pred_counterfactual_values,
            'Effect': df[self.target_column].values - pred_counterfactual_values
        })

        if self.group_column is not None and self.group_column in df.columns:
            detail_df.insert(1, self.group_column,
                             df[self.group_column].values)

        # 集計DataFrameの作成
        group_cols = ['Period']
        if self.group_column is not None and self.group_column in df.columns:
            group_cols.insert(0, self.group_column)

        summary_df = detail_df.groupby(group_cols).agg({
            'Actual': 'mean',
            'Predicted': 'mean',
            'Counterfactual': 'mean',
            'Effect': 'mean'
        }).reset_index()

        # カラム名をわかりやすく変更
        summary_df.columns = group_cols + \
            ['Actual_mean', 'Predicted_mean', 'Counterfactual_mean', 'Effect_mean']

        return summary_df

    def save_model(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        モデルを保存

        Args:
            filepath (Optional[Union[str, Path]]): 保存先のパス
        """
        if self.model_results is None:
            raise ValueError("保存するモデルが存在しません。まずfitメソッドを実行してください。")

        if filepath is None:
            filepath = Path("models") / f"its_model_{self.model_type}.pkl"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model_results': self.model_results,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'time_column': self.time_column,
            'intervention_points': self.intervention_points,
            'group_column': self.group_column,
            'processed_data': self.processed_data,
            'model_type': self.model_type,
            'n_interventions': self.n_interventions
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        モデルを読み込み

        Args:
            filepath (Union[str, Path]): 読み込み元のパス
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model_results = model_data['model_results']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.time_column = model_data['time_column']
        self.intervention_points = model_data['intervention_points']
        self.group_column = model_data['group_column']
        self.processed_data = model_data['processed_data']
        self.model_type = model_data.get('model_type', 'base')
        self.n_interventions = model_data.get(
            'n_interventions', len(self.intervention_points))

    def _ttest_against_zero(self, effects: List[float]) -> float:
        """
        プラセボ効果がゼロと有意に異なるかをt検定で評価

        Args:
            effects (List[float]): プラセボ効果のリスト

        Returns:
            float: p値
        """
        from scipy import stats
        if len(effects) == 0:
            return np.nan
        t_stat, p_value = stats.ttest_1samp(effects, 0)
        return p_value

    def placebo_cross_validate(self,
                               df: pd.DataFrame,
                               target_column: str,
                               n_placebo_points: int = 3,
                               **fit_kwargs) -> Dict[str, Any]:
        """
        プラセボ検定によるクロスバリデーション

        介入点より前の時点を「偽の介入点」として設定し、
        介入効果が検出されないことを確認する。

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数のカラム名
            n_placebo_points (int): プラセボ介入点の数
            **fit_kwargs: fitメソッドに渡す追加パラメータ

        Returns:
            Dict[str, Any]: プラセボ効果の推定値とp値

        References:
            - Penfold & Zhang (2013). "Use of interrupted time series
              analysis in evaluating health care quality improvements"
        """
        # 実介入点の最小値
        real_intervention = min(self.intervention_points)

        # 介入前データのみ抽出
        pre_intervention_data = df[df[self.time_column]
                                   < real_intervention].copy()

        if len(pre_intervention_data) < 10:
            raise ValueError(
                f"介入前データが少なすぎます（{len(pre_intervention_data)}サンプル）。最低10サンプル必要です。")

        # 介入前期間を等分割してプラセボ点を設定
        time_range = pre_intervention_data[self.time_column].max() - \
            pre_intervention_data[self.time_column].min()
        placebo_points = [
            pre_intervention_data[self.time_column].min() +
            (i + 1) * time_range / (n_placebo_points + 1)
            for i in range(n_placebo_points)
        ]

        placebo_effects = []

        for placebo_point in placebo_points:
            # プラセボモデルを作成
            placebo_model = self.__class__(
                time_column=self.time_column,
                intervention_points=[placebo_point],
                group_column=self.group_column
            )

            # フィット（チューニングなしで実行）
            fit_kwargs_copy = fit_kwargs.copy()
            # SARIMAXまたはProphetの場合のみtune_with_optunaを設定
            if self.model_type in ['SARIMAX', 'Prophet']:
                fit_kwargs_copy['tune_with_optuna'] = False

            placebo_model.fit(pre_intervention_data,
                              target_column, **fit_kwargs_copy)

            # プラセボ効果を計算
            effect_df = placebo_model.calculate_intervention_effect()
            # 介入後の効果のみ抽出
            intervention_effects = effect_df[effect_df['Period'].str.contains(
                'Intervention', na=False)]
            if len(intervention_effects) > 0:
                placebo_effect = intervention_effects['Effect_mean'].mean()
                placebo_effects.append(placebo_effect)

        # 統計的検定
        p_value = self._ttest_against_zero(placebo_effects)

        return {
            'placebo_effects': placebo_effects,
            'placebo_points': placebo_points,
            'mean_placebo_effect': np.mean(placebo_effects) if placebo_effects else np.nan,
            'std_placebo_effect': np.std(placebo_effects) if placebo_effects else np.nan,
            'p_value': p_value,
            'is_valid': p_value > 0.05 if not np.isnan(p_value) else None,
            'n_placebo_tests': len(placebo_effects)
        }

    def placebo_cv_multiple_interventions(self,
                                          df: pd.DataFrame,
                                          target_column: str,
                                          n_placebo_per_intervention: int = 3,
                                          **fit_kwargs) -> pd.DataFrame:
        """
        複数介入それぞれに対してプラセボCVを実行

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数のカラム名
            n_placebo_per_intervention (int): 各介入あたりのプラセボ点数
            **fit_kwargs: fitメソッドに渡す追加パラメータ

        Returns:
            pd.DataFrame: 各介入のプラセボ効果の結果
        """
        results = []

        # 各実介入に対してプラセボCVを実行
        for i, real_intervention in enumerate(self.intervention_points):
            # この介入より前のデータのみ抽出
            if i == 0:
                # 最初の介入：全データの開始から介入直前まで
                pre_data = df[df[self.time_column] <
                              real_intervention].copy()
            else:
                # 2つ目以降の介入：前の介入から現在の介入直前まで
                prev_intervention = self.intervention_points[i-1]
                pre_data = df[
                    (df[self.time_column] >= prev_intervention) &
                    (df[self.time_column] < real_intervention)
                ].copy()

            if len(pre_data) < 10:  # 最低サンプルサイズチェック
                print(
                    f"警告: 介入{i+1}の前のデータが少なすぎます（{len(pre_data)}サンプル）。スキップします。")
                continue

            # プラセボ点を生成
            time_min = pre_data[self.time_column].min()
            time_max = pre_data[self.time_column].max()
            time_range = time_max - time_min

            if time_range == 0:
                print(f"警告: 介入{i+1}の前のデータ期間がゼロです。スキップします。")
                continue

            placebo_points = [
                time_min + (j + 1) * time_range /
                (n_placebo_per_intervention + 1)
                for j in range(n_placebo_per_intervention)
            ]

            # 各プラセボ点で効果を推定
            for placebo_point in placebo_points:
                # プラセボモデルを作成
                placebo_model = self.__class__(
                    time_column=self.time_column,
                    intervention_points=[placebo_point],
                    group_column=self.group_column
                )

                # フィット（チューニングなし）
                fit_kwargs_copy = fit_kwargs.copy()
                # SARIMAXまたはProphetの場合のみtune_with_optunaを設定
                if self.model_type in ['SARIMAX', 'Prophet']:
                    fit_kwargs_copy['tune_with_optuna'] = False

                try:
                    placebo_model.fit(
                        pre_data, target_column, **fit_kwargs_copy)
                    effect_df = placebo_model.calculate_intervention_effect()

                    # 介入後の効果を抽出
                    intervention_effects = effect_df[effect_df['Period'].str.contains(
                        'Intervention', na=False)]
                    if len(intervention_effects) > 0:
                        results.append({
                            'real_intervention_index': i,
                            'real_intervention_point': real_intervention,
                            'placebo_point': placebo_point,
                            'placebo_effect': intervention_effects['Effect_mean'].mean(),
                            'placebo_effect_std': intervention_effects['Effect_mean'].std()
                        })
                except Exception as e:
                    print(f"警告: プラセボ点{placebo_point}でのフィット失敗: {e}")
                    continue

        results_df = pd.DataFrame(results)

        # 統計的検定を追加
        if len(results_df) > 0:
            print("\n【複数介入プラセボCV結果】")
            for i in range(len(self.intervention_points)):
                intervention_results = results_df[
                    results_df['real_intervention_index'] == i
                ]

                if len(intervention_results) > 0:
                    effects = intervention_results['placebo_effect'].values
                    p_value = self._ttest_against_zero(effects)

                    print(f"\n介入 {i+1}: {self.intervention_points[i]}")
                    print(f"  プラセボ効果の平均: {np.mean(effects):.3f}")
                    print(f"  プラセボ効果の標準偏差: {np.std(effects):.3f}")
                    print(f"  t検定 p値: {p_value:.3f}")
                    print(
                        f"  → {'⚠️ モデルに問題の可能性' if p_value < 0.05 else '✅ モデルOK'}")

        return results_df

    def sensitivity_analysis_cv(self,
                                df: pd.DataFrame,
                                target_column: str,
                                time_window_variations: Optional[List[Tuple[int, int]]] = None,
                                **fit_kwargs) -> pd.DataFrame:
        """
        感度分析によるクロスバリデーション

        介入点の前後で時間窓を変えて、結果の頑健性を確認する。

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数のカラム名
            time_window_variations (Optional[List[Tuple[int, int]]]):
                (介入前の窓サイズ, 介入後の窓サイズ)のリスト
            **fit_kwargs: fitメソッドに渡す追加パラメータ

        Returns:
            pd.DataFrame: 各時間窓での介入効果推定値

        Example:
            >>> variations = [(12, 12), (24, 12), (12, 24), (24, 24)]
            >>> results = model.sensitivity_analysis_cv(df, 'sales', variations)
        """
        if time_window_variations is None:
            # デフォルトで複数の窓サイズを試す
            time_window_variations = [
                (12, 12), (18, 12), (12, 18), (24, 12), (12, 24)
            ]

        results = []

        for pre_window, post_window in time_window_variations:
            for intervention_point in self.intervention_points:
                # 時間窓を限定したデータを抽出
                window_data = df[
                    (df[self.time_column] >= intervention_point - pre_window) &
                    (df[self.time_column] <= intervention_point + post_window)
                ].copy()

                if len(window_data) < 10:  # 最低サンプルサイズ
                    continue

                # 一時モデルを作成
                temp_model = self.__class__(
                    time_column=self.time_column,
                    intervention_points=[intervention_point],
                    group_column=self.group_column
                )

                # フィット（チューニングなし）
                fit_kwargs_copy = fit_kwargs.copy()
                # SARIMAXまたはProphetの場合のみtune_with_optunaを設定
                if self.model_type in ['SARIMAX', 'Prophet']:
                    fit_kwargs_copy['tune_with_optuna'] = False

                try:
                    temp_model.fit(window_data, target_column,
                                   **fit_kwargs_copy)
                    effect_df = temp_model.calculate_intervention_effect()

                    # 介入後の効果を抽出
                    intervention_effects = effect_df[effect_df['Period'].str.contains(
                        'Intervention', na=False)]
                    if len(intervention_effects) > 0:
                        results.append({
                            'intervention_point': intervention_point,
                            'pre_window': pre_window,
                            'post_window': post_window,
                            'effect_mean': intervention_effects['Effect_mean'].mean(),
                            'effect_std': intervention_effects['Effect_mean'].std()
                        })
                except Exception as e:
                    print(
                        f"警告: 窓サイズ({pre_window}, {post_window})でのフィット失敗: {e}")
                    continue

        return pd.DataFrame(results)


class ITSModelOLS(ITSModelBase):
    """
    OLSモデルを使用したITS分析クラス
    """

    def __init__(self,
                 time_column: str,
                 intervention_points: List[Union[int, float]],
                 group_column: Optional[str] = None):
        super().__init__(time_column, intervention_points, group_column)
        self.model_type = "OLS"

    def fit(self,
            df: pd.DataFrame,
            target_column: str,
            covariates: Optional[List[str]] = None,
            cov_type: str = 'HAC',
            cov_kwds: Optional[Dict[str, Any]] = None,
            add_intercept: bool = True) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        OLSモデルをフィッティング

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数のカラム名
            covariates (Optional[List[str]]): 共変量のカラム名リスト
            cov_type (str): 標準誤差の計算方法
            cov_kwds (Optional[Dict[str, Any]]): 標準誤差計算のオプション
            add_intercept (bool): 切片項を追加するかどうか

        Returns:
            sm.regression.linear_model.RegressionResultsWrapper: フィット結果
        """
        # データの妥当性チェック
        self.validate_data(df)

        # データの前処理
        processed_df = self.fit_transform(df, add_intercept=add_intercept)

        # 特徴量の設定
        feature_cols = self.get_its_variables()

        # 共変量の追加
        if covariates is not None:
            missing_covariates = [
                col for col in covariates if col not in processed_df.columns]
            if missing_covariates:
                raise ValueError(f"指定された共変量が存在しません: {missing_covariates}")

            # カテゴリカル変数をダミー変数に変換
            for covariate in covariates:
                if processed_df[covariate].dtype == 'object' or processed_df[covariate].dtype.name == 'category':
                    dummies = pd.get_dummies(
                        processed_df[covariate], prefix=covariate, drop_first=True)
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                    feature_cols.extend(dummies.columns.tolist())
                else:
                    feature_cols.append(covariate)

        # 目的変数の存在チェック
        if target_column not in processed_df.columns:
            raise ValueError(f"目的変数 '{target_column}' がデータに存在しません")

        # モデルのフィッティング
        X = processed_df[feature_cols]
        y = processed_df[target_column]

        X = X.astype(float)

        model = sm.OLS(y, X)

        if cov_kwds is None:
            cov_kwds = {'maxlags': 3} if cov_type == 'HAC' else {}

        self.model_results = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
        self.feature_columns = feature_cols
        self.target_column = target_column

        return self.model_results

    def predict(self,
                df: Optional[pd.DataFrame] = None,
                return_ci: bool = False,
                use_preprocessed: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """
        予測値を計算

        Args:
            df (Optional[pd.DataFrame]): 予測用データ（Noneの場合は学習データを使用）
            return_ci (bool): 信頼区間を返すかどうか
            use_preprocessed (bool): dfが既に前処理済みの場合True

        Returns:
            Union[pd.Series, pd.DataFrame]: 予測結果
        """
        if self.model_results is None:
            raise ValueError("モデルがフィットされていません。まずfitメソッドを実行してください。")

        if df is None:
            if self.processed_data is None:
                raise ValueError("学習データが保存されていません。")
            X = self.processed_data[self.feature_columns]
        else:
            if use_preprocessed:
                processed_df = df
            else:
                processed_df = self.fit_transform(df)

            available_cols = []
            for col in self.feature_columns:
                if col in processed_df.columns:
                    available_cols.append(col)
                else:
                    processed_df[col] = 0
                    available_cols.append(col)

            X = processed_df[self.feature_columns]

        prediction = self.model_results.get_prediction(X)

        if return_ci:
            return prediction.summary_frame()
        else:
            return prediction.predicted_mean

    def summary(self) -> None:
        """
        モデル結果のサマリーを表示
        """
        if self.model_results is None:
            raise ValueError("モデルがフィットされていません。")

        print(self.model_results.summary())


class ITSModelSARIMAX(ITSModelBase):
    """
    SARIMAXモデルを使用したITS分析クラス
    """

    def __init__(self,
                 time_column: str,
                 intervention_points: List[Union[int, float]],
                 group_column: Optional[str] = None):
        super().__init__(time_column, intervention_points, group_column)
        self.model_type = "SARIMAX"
        self.best_params: Optional[Dict[str, Any]] = None

    def fit(self,
            df: pd.DataFrame,
            target_column: str,
            order: Tuple[int, int, int] = (1, 0, 1),
            seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
            covariates: Optional[List[str]] = None,
            tune_with_optuna: bool = False,
            n_trials: int = 50,
            add_intercept: bool = False) -> Any:
        """
        SARIMAXモデルをフィッティング

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数のカラム名
            order (Tuple[int, int, int]): ARIMA(p, d, q)パラメータ
            seasonal_order (Tuple[int, int, int, int]): 季節性ARIMA(P, D, Q, s)パラメータ
            covariates (Optional[List[str]]): 共変量のカラム名リスト
            tune_with_optuna: Optunaでハイパーパラメータチューニングを行うか
            n_trials: Optunaの試行回数
            add_intercept (bool): 切片項を追加するかどうか（SARIMAXでは通常False）

        Returns:
            SARIMAX結果オブジェクト
        """
        # データの妥当性チェック
        self.validate_data(df)

        # データの前処理
        processed_df = self.fit_transform(df, add_intercept=add_intercept)

        # 外生変数（ITS変数 + 共変量）の準備
        exog_cols = []
        for i in range(self.n_interventions):
            exog_cols.append(f'D_{i+1}')
            exog_cols.append(f'timedelta_{i+1}')

        if covariates is not None:
            missing_covariates = [
                col for col in covariates if col not in processed_df.columns]
            if missing_covariates:
                raise ValueError(f"指定された共変量が存在しません: {missing_covariates}")
            exog_cols.extend(covariates)

        # 目的変数の存在チェック
        if target_column not in processed_df.columns:
            raise ValueError(f"目的変数 '{target_column}' がデータに存在しません")

        y = processed_df[target_column]
        exog = processed_df[exog_cols] if exog_cols else None

        self.feature_columns = exog_cols
        self.target_column = target_column

        # Optunaでチューニング（介入前データのみ）
        if tune_with_optuna:
            best_params = self._optuna_tune_pre_intervention(
                y, exog, n_trials)
            order = best_params['order']
            seasonal_order = best_params['seasonal_order']
            self.best_params = best_params

        # SARIMAXモデルのフィッティング（全データ）
        model = SARIMAX(y, exog=exog, order=order,
                        seasonal_order=seasonal_order)
        self.model_results = model.fit(disp=False)

        return self.model_results

    def _optuna_tune_pre_intervention(self,
                                      y: pd.Series,
                                      exog: Optional[pd.DataFrame],
                                      n_trials: int) -> Dict[str, Any]:
        """
        Optunaを使ってSARIMAXのハイパーパラメータをチューニング

        介入前データのみを使用してチューニングを行い、
        介入効果に依存しないモデル選択を実現する。

        Args:
            y (pd.Series): 目的変数
            exog (Optional[pd.DataFrame]): 外生変数
            n_trials (int): 試行回数

        Returns:
            Dict[str, Any]: 最適パラメータ
        """
        # 最初の介入点を取得
        first_intervention = min(self.intervention_points)

        # 介入前データのみ抽出
        if self.processed_data is not None:
            pre_intervention_mask = self.processed_data[self.time_column] < first_intervention
            y_pre = y[pre_intervention_mask]

            # 介入ダミーを除外した外生変数のみ使用
            if exog is not None:
                # 介入ダミー以外の共変量のみ抽出
                non_intervention_cols = [
                    col for col in exog.columns
                    if not (col.startswith('D_') or col.startswith('timedelta_'))
                ]
                exog_pre = exog.loc[pre_intervention_mask,
                                    non_intervention_cols] if non_intervention_cols else None
            else:
                exog_pre = None
        else:
            raise ValueError("processed_dataが設定されていません")

        if len(y_pre) < 10:
            raise ValueError(
                f"介入前データが少なすぎます（{len(y_pre)}サンプル）。最低10サンプル必要です。")

        def objective(trial):
            p = trial.suggest_int('p', 0, 3)
            d = trial.suggest_int('d', 0, 2)  # 差分次数は低めに
            q = trial.suggest_int('q', 0, 3)
            P = trial.suggest_int('P', 0, 2)
            D = trial.suggest_int('D', 0, 1)
            Q = trial.suggest_int('Q', 0, 2)
            s = trial.suggest_categorical('s', [0, 12, 30])

            try:
                # 介入前データでのみフィット
                model = SARIMAX(
                    y_pre,
                    exog=exog_pre,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s)
                )
                results = model.fit(disp=False)

                # AICで評価（MAPEより頑健）
                return results.aic
            except:
                return np.inf

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = {
            'order': (study.best_params['p'], study.best_params['d'], study.best_params['q']),
            'seasonal_order': (study.best_params['P'], study.best_params['D'],
                               study.best_params['Q'], study.best_params['s'])
        }

        print(f"\n✅ SARIMAX最適パラメータ（介入前データでチューニング）:")
        print(f"  order: {best_params['order']}")
        print(f"  seasonal_order: {best_params['seasonal_order']}")
        print(f"  AIC: {study.best_value:.2f}")

        return best_params

    def predict(self,
                df: Optional[pd.DataFrame] = None,
                return_ci: bool = False,
                use_preprocessed: bool = False,
                steps: Optional[int] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        予測値を計算

        Args:
            df (Optional[pd.DataFrame]): 予測用データ（Noneの場合は学習データを使用）
            return_ci (bool): 信頼区間を返すかどうか
            use_preprocessed (bool): dfが既に前処理済みの場合True
            steps (Optional[int]): 予測ステップ数

        Returns:
            Union[pd.Series, pd.DataFrame]: 予測結果
        """
        if self.model_results is None:
            raise ValueError("モデルがフィットされていません。まずfitメソッドを実行してください。")

        if df is None:
            # In-sample予測
            pred = self.model_results.fittedvalues
            if return_ci:
                # 信頼区間を含むDataFrameを返す
                ci = self.model_results.get_forecast(
                    steps=len(pred)).summary_frame()
                return ci
            else:
                return pred
        else:
            if use_preprocessed:
                processed_df = df
            else:
                processed_df = self.fit_transform(df)

            # 外生変数の準備
            if self.feature_columns:
                exog = processed_df[self.feature_columns]
            else:
                exog = None

            n_steps = len(processed_df) if steps is None else steps
            forecast = self.model_results.get_forecast(
                steps=n_steps, exog=exog)

            if return_ci:
                return forecast.summary_frame()
            else:
                return forecast.predicted_mean

    def summary(self) -> None:
        """
        モデル結果のサマリーを表示
        """
        if self.model_results is None:
            raise ValueError("モデルがフィットされていません。")

        print(self.model_results.summary())


class ITSModelProphet(ITSModelBase):
    """
    Prophetモデルを使用したITS分析クラス
    """

    def __init__(self,
                 time_column: str,
                 intervention_points: List[Union[int, float]],
                 group_column: Optional[str] = None):
        super().__init__(time_column, intervention_points, group_column)
        self.model_type = "Prophet"
        self.best_params: Optional[Dict[str, Any]] = None

    def fit(self,
            df: pd.DataFrame,
            target_column: str,
            covariates: Optional[List[str]] = None,
            changepoint_prior_scale: float = 0.05,
            seasonality_prior_scale: float = 10.0,
            tune_with_optuna: bool = False,
            n_trials: int = 50,
            add_intercept: bool = False) -> Prophet:
        """
        Prophetモデルをフィッティング

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数のカラム名
            covariates (Optional[List[str]]): 共変量のカラム名リスト
            changepoint_prior_scale (float): トレンド変化の柔軟性
            seasonality_prior_scale (float): 季節性の柔軟性
            tune_with_optuna: Optunaでハイパーパラメータチューニングを行うか
            n_trials: Optunaの試行回数
            add_intercept (bool): 切片項を追加するかどうか（Prophetでは不要）

        Returns:
            Prophet: フィット済みProphetモデル
        """
        # データの妥当性チェック
        self.validate_data(df)

        # データの前処理
        processed_df = self.fit_transform(df, add_intercept=False)

        # 目的変数の存在チェック
        if target_column not in processed_df.columns:
            raise ValueError(f"目的変数 '{target_column}' がデータに存在しません")

        self.target_column = target_column

        # Prophet用のデータフレーム作成
        # dsは日時型である必要があるため、整数の場合は日付に変換
        ds_values = processed_df[self.time_column]
        if not pd.api.types.is_datetime64_any_dtype(ds_values):
            # 整数や数値の場合、基準日からの日数として変換
            base_date = pd.Timestamp('2000-01-01')
            ds_values = base_date + \
                pd.to_timedelta(ds_values - ds_values.min(), unit='D')

        prophet_df = pd.DataFrame({
            'ds': ds_values,
            'y': processed_df[target_column]
        })

        # 外生変数（レグレッサー）の追加
        regressor_cols = []
        for i in range(self.n_interventions):
            prophet_df[f'D_{i+1}'] = processed_df[f'D_{i+1}']
            prophet_df[f'timedelta_{i+1}'] = processed_df[f'timedelta_{i+1}']
            regressor_cols.append(f'D_{i+1}')
            regressor_cols.append(f'timedelta_{i+1}')

        if covariates is not None:
            missing_covariates = [
                col for col in covariates if col not in processed_df.columns]
            if missing_covariates:
                raise ValueError(f"指定された共変量が存在しません: {missing_covariates}")

            for cov in covariates:
                prophet_df[cov] = processed_df[cov]
                regressor_cols.append(cov)

        self.feature_columns = regressor_cols

        # Optunaでチューニング（介入前データのみ）
        if tune_with_optuna:
            best_params = self._optuna_tune_pre_intervention(
                prophet_df, regressor_cols, n_trials)
            changepoint_prior_scale = best_params['changepoint_prior_scale']
            n_changepoints = best_params['n_changepoints']
            self.best_params = best_params
        else:
            n_changepoints = 25  # デフォルト値

        # Prophetモデルの作成とフィッティング（全データ）
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            n_changepoints=n_changepoints
        )

        # レグレッサーの追加
        for regressor in regressor_cols:
            model.add_regressor(regressor)

        self.model_results = model.fit(prophet_df)

        return self.model_results

    def _optuna_tune_pre_intervention(self,
                                      prophet_df: pd.DataFrame,
                                      regressor_cols: List[str],
                                      n_trials: int) -> Dict[str, Any]:
        """
        Optunaを使ってProphetのハイパーパラメータをチューニング

        介入前データのみを使用してチューニングを行い、
        介入効果に依存しないモデル選択を実現する。

        Args:
            prophet_df (pd.DataFrame): Prophet用データフレーム
            regressor_cols (List[str]): レグレッサーのカラム名リスト
            n_trials (int): 試行回数

        Returns:
            Dict[str, Any]: 最適パラメータ
        """
        # 最初の介入点を取得
        first_intervention = min(self.intervention_points)

        # 介入前データのみ抽出
        if self.processed_data is not None:
            pre_intervention_mask = self.processed_data[self.time_column] < first_intervention

            # Prophet用データフレームから介入前のみ抽出
            prophet_df_pre = prophet_df[pre_intervention_mask].copy()

            # 介入ダミーを除外した共変量のみ使用
            non_intervention_regressors = [
                col for col in regressor_cols
                if not (col.startswith('D_') or col.startswith('timedelta_'))
            ]
        else:
            raise ValueError("processed_dataが設定されていません")

        if len(prophet_df_pre) < 10:
            raise ValueError(
                f"介入前データが少なすぎます（{len(prophet_df_pre)}サンプル）。最低10サンプル必要です。")

        def objective(trial):
            changepoint_prior_scale = trial.suggest_float(
                'changepoint_prior_scale', 0.001, 0.5, log=True
            )
            n_changepoints = trial.suggest_int('n_changepoints', 5, 25)

            try:
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    n_changepoints=n_changepoints
                )

                # 非介入共変量のみ追加
                for regressor in non_intervention_regressors:
                    model.add_regressor(regressor)

                model.fit(prophet_df_pre)

                # 時系列CVで評価（Prophet標準機能）
                from prophet.diagnostics import cross_validation, performance_metrics

                # 介入前データのサイズに応じてCV設定を調整
                data_length = len(prophet_df_pre)
                initial_days = max(int(data_length * 0.5), 3)
                period_days = max(int(data_length * 0.1), 1)
                horizon_days = max(int(data_length * 0.2), 2)

                df_cv = cross_validation(
                    model,
                    initial=f'{initial_days} days',
                    period=f'{period_days} days',
                    horizon=f'{horizon_days} days'
                )
                df_p = performance_metrics(df_cv)

                # MAPEを返す
                return df_p['mape'].mean()
            except Exception as e:
                # CV失敗時はinfを返す
                return np.inf

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params

        print(f"\n✅ Prophet最適パラメータ（介入前データでチューニング）:")
        print(
            f"  changepoint_prior_scale: {best_params['changepoint_prior_scale']:.4f}")
        print(f"  n_changepoints: {best_params['n_changepoints']}")
        print(f"  MAPE: {study.best_value:.4f}")

        return best_params

    def predict(self,
                df: Optional[pd.DataFrame] = None,
                return_ci: bool = False,
                use_preprocessed: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """
        予測値を計算

        Args:
            df (Optional[pd.DataFrame]): 予測用データ（Noneの場合は学習データを使用）
            return_ci (bool): 信頼区間を返すかどうか
            use_preprocessed (bool): dfが既に前処理済みの場合True

        Returns:
            Union[pd.Series, pd.DataFrame]: 予測結果
        """
        if self.model_results is None:
            raise ValueError("モデルがフィットされていません。まずfitメソッドを実行してください。")

        if df is None:
            if self.processed_data is None:
                raise ValueError("学習データが保存されていません。")
            processed_df = self.processed_data
        else:
            if use_preprocessed:
                processed_df = df
            else:
                processed_df = self.fit_transform(df)

        # Prophet用のデータフレーム作成
        # dsは日時型である必要があるため、整数の場合は日付に変換
        ds_values = processed_df[self.time_column]
        if not pd.api.types.is_datetime64_any_dtype(ds_values):
            # 整数や数値の場合、基準日からの日数として変換
            base_date = pd.Timestamp('2000-01-01')
            ds_values = base_date + \
                pd.to_timedelta(ds_values - ds_values.min(), unit='D')

        future_df = pd.DataFrame({
            'ds': ds_values
        })

        # レグレッサーの追加
        for regressor in self.feature_columns:
            future_df[regressor] = processed_df[regressor]

        forecast = self.model_results.predict(future_df)

        if return_ci:
            # 信頼区間を含むDataFrameを返す
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                columns={'yhat': 'mean', 'yhat_lower': 'mean_ci_lower',
                         'yhat_upper': 'mean_ci_upper'}
            )
        else:
            return forecast['yhat']

    def summary(self) -> None:
        """
        モデル結果のサマリーを表示
        """
        if self.model_results is None:
            raise ValueError("モデルがフィットされていません。")

        print("Prophet Model Components:")
        print(self.model_results.params)


class ITSVisualizer:
    """
    ITS分析結果の可視化を行うクラス（複数介入対応・新仕様対応）

    仕様：
    - 反実仮想予測値の95%信頼区間を薄帯で表示
    - 実予測は介入前=緑実線、介入後=緑破線、信頼区間なし
    - 実績値は点でプロット
    """

    def __init__(self, model: ITSModelBase):
        """
        ITSVisualizerの初期化

        Args:
            model (ITSModelBase): フィット済みのITSModelインスタンス
        """
        self.model = model
        if model.model_results is None:
            raise ValueError("フィット済みのモデルが必要です。")

    def plot(self,
             group_column: Optional[str] = None,
             figsize: tuple = (12, 8),
             save_path: Optional[Union[str, Path]] = None,
             alpha: float = 0.2,
             point_alpha: float = 0.6,
             intervention_color: str = 'black',
             actual_color: str = 'blue',
             fit_color: str = 'green',
             counterfactual_color: str = 'red') -> plt.Figure:
        """
        ITS分析結果をプロット（新仕様）

        Args:
            group_column (Optional[str]): グループ分けするカラム名
            figsize (tuple): 図のサイズ
            save_path (Optional[Union[str, Path]]): 保存先パス
            alpha (float): 反実仮想信頼区間の透明度
            point_alpha (float): データ点の透明度
            intervention_color (str): 介入ラインの色
            actual_color (str): 実測値の色
            fit_color (str): フィットラインの色（緑）
            counterfactual_color (str): 反実仮想ラインの色（赤）

        Returns:
            plt.Figure: 作成された図オブジェクト
        """
        if self.model.processed_data is None:
            raise ValueError("処理済みデータが存在しません。")

        # グループ列の決定
        if group_column is None:
            group_column = self.model.group_column

        # データの準備
        df = self.model.processed_data.copy()

        # グループの取得
        if group_column is not None and group_column in df.columns:
            groups = sorted(df[group_column].unique())
            n_groups = len(groups)
        else:
            groups = [None]
            n_groups = 1

        # サブプロットの作成
        if n_groups == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        else:
            n_cols = min(2, n_groups)
            n_rows = (n_groups + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(
                figsize[0]*n_cols/2, figsize[1]*n_rows/2))
            if n_rows * n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

        # 各グループのプロット
        for i, group in enumerate(groups):
            if i >= len(axes):
                break

            ax = axes[i]

            # グループデータのフィルタリング
            if group is not None:
                group_data = df[df[group_column] == group].copy()
                title_suffix = f": {group_column} = {group}"
            else:
                group_data = df.copy()
                title_suffix = ""

            # 実予測値の計算（信頼区間なし）
            pred_actual = self.model.predict(group_data, return_ci=False)
            if isinstance(pred_actual, pd.DataFrame):
                pred_actual_values = pred_actual['mean'].values
            elif isinstance(pred_actual, pd.Series):
                pred_actual_values = pred_actual.values
            else:
                pred_actual_values = pred_actual

            # 反実仮想予測値の計算（信頼区間あり）
            pred_counterfactual = self.model.predict_counterfactual(
                group_data, return_ci=True)
            if isinstance(pred_counterfactual, pd.DataFrame):
                cf_mean = pred_counterfactual['mean'].values if 'mean' in pred_counterfactual.columns else pred_counterfactual['yhat'].values
                cf_lower = pred_counterfactual.get(
                    'mean_ci_lower', pred_counterfactual.get('obs_ci_lower', cf_mean)).values
                cf_upper = pred_counterfactual.get(
                    'mean_ci_upper', pred_counterfactual.get('obs_ci_upper', cf_mean)).values
            elif isinstance(pred_counterfactual, pd.Series):
                cf_mean = pred_counterfactual.values
                cf_lower = cf_mean
                cf_upper = cf_mean
            else:
                cf_mean = pred_counterfactual
                cf_lower = cf_mean
                cf_upper = cf_mean

            # 実測値のプロット（点）
            ax.scatter(group_data[self.model.time_column],
                       group_data[self.model.target_column],
                       alpha=point_alpha, color=actual_color,
                       label='Actual Values', s=30, zorder=5)

            # 最初の介入点
            first_intervention = self.model.intervention_points[0]

            # 介入前の実予測（緑実線）
            pre_intervention = group_data[group_data[self.model.time_column]
                                          < first_intervention]
            if len(pre_intervention) > 0:
                pre_idx = group_data[self.model.time_column] < first_intervention
                ax.plot(group_data.loc[pre_idx, self.model.time_column],
                        pred_actual_values[pre_idx],
                        color=fit_color, linewidth=2, linestyle='-',
                        label='Pre-intervention fit', zorder=3)

            # 介入後の実予測（緑破線）
            post_intervention = group_data[group_data[self.model.time_column]
                                           >= first_intervention]
            if len(post_intervention) > 0:
                post_idx = group_data[self.model.time_column] >= first_intervention
                ax.plot(group_data.loc[post_idx, self.model.time_column],
                        pred_actual_values[post_idx],
                        color=fit_color, linewidth=2, linestyle='--',
                        label='Post-intervention fit', zorder=3)

            # 反実仮想のプロット（介入後のみ、信頼区間あり）
            if len(post_intervention) > 0:
                post_idx_arr = group_data[self.model.time_column] >= first_intervention

                # 反実仮想の95%信頼区間（薄帯）
                ax.fill_between(group_data.loc[post_idx_arr, self.model.time_column],
                                cf_lower[post_idx_arr],
                                cf_upper[post_idx_arr],
                                alpha=alpha, color=counterfactual_color,
                                label='Counterfactual 95% CI', zorder=1)

                # 反実仮想の平均（赤破線）
                ax.plot(group_data.loc[post_idx_arr, self.model.time_column],
                        cf_mean[post_idx_arr],
                        color=counterfactual_color, linewidth=2,
                        linestyle='--', label='Counterfactual', zorder=2)

            # 介入ラインの追加（複数介入対応）
            for intervention_point in self.model.intervention_points:
                ax.axvline(x=intervention_point, color=intervention_color,
                           linestyle=':', alpha=0.7, zorder=4)

            # 凡例に介入ラインを追加（1つだけ）
            ax.axvline(x=self.model.intervention_points[0], color=intervention_color,
                       linestyle=':', alpha=0.7, label='Intervention', zorder=4)

            # プロットの装飾
            ax.set_xlabel(self.model.time_column.title())
            ax.set_ylabel(self.model.target_column.title())
            ax.set_title(f'Interrupted Time Series Analysis{title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 使用されていない軸を非表示
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        # 保存
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")

        return fig
