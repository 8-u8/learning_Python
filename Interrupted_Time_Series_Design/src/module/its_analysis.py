"""
Interrupted Time Series Analysis (ITAS) Package

このパッケージは、Interrupted Time Series Designを使った分析を
抽象化し、任意のデータに対して適用可能なクラスを提供します。

Classes:
    ITSDataPreprocessor: ITSに必要な変数を生成するクラス
    ITSModel: OLSモデルを実行・管理するクラス  
    ITSVisualizer: 結果を可視化するクラス
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
from typing import Optional, List, Dict, Union, Any
from pathlib import Path


class ITSDataPreprocessor:
    """
    Interrupted Time Series Designに必要な変数を生成するクラス

    任意のデータから以下の変数を生成します：
    - t: 時系列の起点から終点までのカウントアップ
    - T: 介入のあった単位時間のときに1、それ以外は0を示すダミー変数
    - D: 介入のあった単位時間より前は0、介入のあった単位時間以降に1を示すダミー変数
    - time_after: 介入後の経過時間を表すカウントアップ
    - const: 切片項、全期間を通して1となる変数
    """

    def __init__(self,
                 time_column: str,
                 intervention_point: Union[int, float],
                 group_column: Optional[str] = None):
        """
        ITSDataPreprocessorの初期化

        Args:
            time_column (str): 時間を表すカラム名
            intervention_point (Union[int, float]): 介入が起こった時点の値
            group_column (Optional[str]): グループを表すカラム名（州や地域など）
        """
        self.time_column = time_column
        self.intervention_point = intervention_point
        self.group_column = group_column
        self.processed_data: Optional[pd.DataFrame] = None

    def fit_transform(self,
                      df: pd.DataFrame,
                      add_intercept: bool = True,
                      time_start_from_one: bool = True) -> pd.DataFrame:
        """
        データにITS用の変数を追加して変換

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

        # 介入点ダミー (T): 介入のあった時点のみ1
        df_out['T'] = (df_out[self.time_column] ==
                       self.intervention_point).astype(int)

        # 介入後ダミー (D): 介入時点以降は1
        df_out['D'] = (df_out[self.time_column] >=
                       self.intervention_point).astype(int)

        # 介入後経過時間 (time_after)
        if self.group_column is not None:
            # グループ別に計算
            df_out['time_after'] = (df_out.groupby(self.group_column)['D']
                                    .transform('cumsum'))
        else:
            # 全体で計算
            df_out['time_after'] = df_out['D'].cumsum()

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
        base_vars = ['t', 'D', 'time_after']
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

        # 介入点の存在チェック
        if self.intervention_point not in df[self.time_column].values:
            raise ValueError(
                f"介入点 '{self.intervention_point}' がデータの時間軸に存在しません")

        return True


class ITSModel(ITSDataPreprocessor):
    """
    ITSDataPreprocessorを継承してstatsmodels.api.OLSを実行するクラス

    共変量cooperateの有無を確認し、適切にモデル実行とモデル保存を行います。
    """

    def __init__(self,
                 time_column: str,
                 intervention_point: Union[int, float],
                 group_column: Optional[str] = None):
        """
        ITSModelの初期化

        Args:
            time_column (str): 時間を表すカラム名
            intervention_point (Union[int, float]): 介入が起こった時点の値
            group_column (Optional[str]): グループを表すカラム名
        """
        super().__init__(time_column, intervention_point, group_column)
        self.model_results: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self.feature_columns: Optional[List[str]] = None
        self.target_column: Optional[str] = None

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
            # 共変量の存在チェック
            missing_covariates = [
                col for col in covariates if col not in processed_df.columns]
            if missing_covariates:
                raise ValueError(f"指定された共変量が存在しません: {missing_covariates}")

            # カテゴリカル変数をダミー変数に変換
            for covariate in covariates:
                if processed_df[covariate].dtype == 'object' or processed_df[covariate].dtype.name == 'category':
                    # カテゴリカル変数をダミー変数に変換（最初のカテゴリを基準として除外）
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

        # データ型を数値型に強制変換（カテゴリカル変数のダミー化後など）
        X = X.astype(float)

        model = sm.OLS(y, X)

        # 標準誤差計算のデフォルト設定
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
                # 既に前処理済みのデータを使用（counterfactual計算等）
                processed_df = df
            else:
                # 新しいデータを前処理
                processed_df = self.fit_transform(df)

            # 学習時と同じ特徴量列を作成する必要がある
            available_cols = []
            for col in self.feature_columns:
                if col in processed_df.columns:
                    available_cols.append(col)
                else:
                    # ダミー変数の場合、該当する列がない場合は0で埋める
                    processed_df[col] = 0
                    available_cols.append(col)

            X = processed_df[self.feature_columns]

        prediction = self.model_results.get_prediction(X)

        if return_ci:
            return prediction.summary_frame()
        else:
            return prediction.predicted_mean

    def save_model(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        モデルを保存

        Args:
            filepath (Optional[Union[str, Path]]): 保存先のパス（デフォルト: models/its_model.pkl）
        """
        if self.model_results is None:
            raise ValueError("保存するモデルが存在しません。まずfitメソッドを実行してください。")

        # デフォルトのファイルパスを設定
        if filepath is None:
            filepath = Path("models") / "its_model.pkl"

        filepath = Path(filepath)

        # ディレクトリが存在しない場合は作成
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model_results': self.model_results,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'time_column': self.time_column,
            'intervention_point': self.intervention_point,
            'group_column': self.group_column,
            'processed_data': self.processed_data
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
        self.intervention_point = model_data['intervention_point']
        self.group_column = model_data['group_column']
        self.processed_data = model_data['processed_data']

    def summary(self) -> None:
        """
        モデル結果のサマリーを表示
        """
        if self.model_results is None:
            raise ValueError("モデルがフィットされていません。")

        print(self.model_results.summary())


class ITSVisualizer:
    """
    ITS分析結果の可視化を行うクラス

    グループ別の描画と画像保存機能を提供します。
    """

    def __init__(self, model: ITSModel):
        """
        ITSVisualizerの初期化

        Args:
            model (ITSModel): フィット済みのITSModelインスタンス
        """
        self.model = model
        if model.model_results is None:
            raise ValueError("フィット済みのモデルが必要です。")

    def plot(self,
             group_column: Optional[str] = None,
             figsize: tuple = (12, 8),
             save_path: Optional[Union[str, Path]] = None,
             show_counterfactual: bool = True,
             show_confidence_interval: bool = True,
             alpha: float = 0.3,
             point_alpha: float = 0.6,
             intervention_color: str = 'black',
             actual_color: str = 'blue',
             fit_color: str = 'green',
             counterfactual_color: str = 'red') -> plt.Figure:
        """
        ITS分析結果をプロット

        Args:
            group_column (Optional[str]): グループ分けするカラム名
            figsize (tuple): 図のサイズ
            save_path (Optional[Union[str, Path]]): 保存先パス
            show_counterfactual (bool): 反実仮想ラインを表示するか
            show_confidence_interval (bool): 信頼区間を表示するか
            alpha (float): 信頼区間の透明度
            point_alpha (float): データ点の透明度
            intervention_color (str): 介入ラインの色
            actual_color (str): 実測値の色
            fit_color (str): フィットラインの色
            counterfactual_color (str): 反実仮想ラインの色

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
            n_cols = min(3, n_groups)  # 最大3列
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

            # 予測値の計算
            predictions = self.model.predict(
                group_data, return_ci=show_confidence_interval)

            # 反実仮想の計算（介入効果を0にセット）
            if show_counterfactual:
                counterfactual_data = group_data.copy()
                # 介入がなかった場合のシナリオ：T, D, time_after すべて0に設定
                counterfactual_data['D'] = 0           # 介入後レベル効果を0に
                counterfactual_data['time_after'] = 0  # 介入後トレンド効果を0に
                counterfactual_pred = self.model.predict(
                    counterfactual_data, use_preprocessed=True)
                # 実測値のプロット
                group_data['pred_counterfactual'] = counterfactual_pred
            ax.scatter(group_data[self.model.time_column],
                       group_data[self.model.target_column],
                       alpha=point_alpha, color=actual_color,
                       label='Actual Values', s=30)

            # 介入前のフィット
            pre_intervention = group_data[group_data[self.model.time_column]
                                          <= self.model.intervention_point]
            if len(pre_intervention) > 0:
                if show_confidence_interval and isinstance(predictions, pd.DataFrame):
                    pred_values = predictions.loc[pre_intervention.index, 'mean']
                else:
                    pred_values = predictions[pre_intervention.index]

                ax.plot(pre_intervention[self.model.time_column], pred_values,
                        color=fit_color, linewidth=2, linestyle='-',
                        label='Pre-intervention fit')

            # 介入後のフィット
            post_intervention = group_data[group_data[self.model.time_column]
                                           >= self.model.intervention_point]
            if len(post_intervention) > 0:
                if show_confidence_interval and isinstance(predictions, pd.DataFrame):
                    pred_values = predictions.loc[post_intervention.index, 'mean']
                    ci_lower = predictions.loc[post_intervention.index,
                                               'mean_ci_lower']
                    ci_upper = predictions.loc[post_intervention.index,
                                               'mean_ci_upper']

                    # 信頼区間の表示
                    ax.fill_between(post_intervention[self.model.time_column],
                                    ci_lower, ci_upper,
                                    alpha=alpha, color=fit_color,
                                    label='95% Confidence Interval')
                else:
                    pred_values = predictions[post_intervention.index]

                ax.plot(post_intervention[self.model.time_column], pred_values,
                        color=fit_color, linewidth=2, linestyle='-',
                        label='Post-intervention fit')

                # 反実仮想のプロット
                if show_counterfactual:
                    ax.plot(post_intervention[self.model.time_column],
                            post_intervention['pred_counterfactual'],
                            color=counterfactual_color, linewidth=2,
                            linestyle='--', label='Counterfactual')

            # 介入ラインの追加
            ax.axvline(x=self.model.intervention_point, color=intervention_color,
                       linestyle='--', alpha=0.7, label='Intervention')

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
            # ディレクトリが存在しない場合は作成
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")

        return fig
