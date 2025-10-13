"""
Test suite for Interrupted Time Series Analysis Package
"""


from module.its_analysis import (
    ITSDataPreprocessor,
    ITSModelOLS,
    ITSModelSARIMAX,
    ITSModelProphet,
    ITSVisualizer
)
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def synthetic_data_single_intervention():
    np.random.seed(42)
    n = 100
    time = np.arange(1, n + 1)

    intervention_point = 50
    trend = 0.5 * time
    intervention_effect = np.where(time >= intervention_point, 10, 0)
    intervention_trend = np.where(
        time >= intervention_point, 0.3 * (time - intervention_point), 0)
    noise = np.random.normal(0, 2, n)

    y = trend + intervention_effect + intervention_trend + noise

    df = pd.DataFrame({
        'time': time,
        'value': y
    })

    return df, 50


@pytest.fixture
def synthetic_data_multiple_interventions():
    np.random.seed(42)
    n = 150
    time = np.arange(1, n + 1)

    intervention_points = [40, 80, 120]
    trend = 0.3 * time

    effect_1 = np.where((time >= intervention_points[0]) & (
        time < intervention_points[1]), 5, 0)
    trend_1 = np.where((time >= intervention_points[0]) & (time < intervention_points[1]),
                       0.2 * (time - intervention_points[0]), 0)

    effect_2 = np.where((time >= intervention_points[1]) & (
        time < intervention_points[2]), -3, 0)
    trend_2 = np.where((time >= intervention_points[1]) & (time < intervention_points[2]),
                       -0.1 * (time - intervention_points[1]), 0)

    effect_3 = np.where(time >= intervention_points[2], 8, 0)
    trend_3 = np.where(
        time >= intervention_points[2], 0.4 * (time - intervention_points[2]), 0)

    noise = np.random.normal(0, 2, n)

    y = trend + effect_1 + trend_1 + effect_2 + trend_2 + effect_3 + trend_3 + noise

    df = pd.DataFrame({
        'time': time,
        'value': y
    })

    return df, intervention_points


class TestITSDataPreprocessor:

    def test_single_intervention(self, synthetic_data_single_intervention):
        df, intervention_point = synthetic_data_single_intervention

        preprocessor = ITSDataPreprocessor(
            time_column='time',
            intervention_points=[intervention_point]
        )

        processed_df = preprocessor.fit_transform(df)

        assert 't' in processed_df.columns
        assert 'D_1' in processed_df.columns
        assert 'timedelta_1' in processed_df.columns

    def test_multiple_interventions(self, synthetic_data_multiple_interventions):
        df, intervention_points = synthetic_data_multiple_interventions

        preprocessor = ITSDataPreprocessor(
            time_column='time',
            intervention_points=intervention_points
        )

        processed_df = preprocessor.fit_transform(df)

        assert 'D_1' in processed_df.columns
        assert 'D_2' in processed_df.columns
        assert 'D_3' in processed_df.columns


class TestITSModelOLS:

    def test_fit_predict(self, synthetic_data_single_intervention):
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelOLS(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')
        predictions = model.predict()
        assert len(predictions) == len(df)


class TestITSModelSARIMAX:

    def test_fit_predict(self, synthetic_data_single_intervention):
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelSARIMAX(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value', order=(1, 0, 1))
        predictions = model.predict()
        assert len(predictions) == len(df)


class TestITSModelProphet:

    def test_fit_predict(self, synthetic_data_single_intervention):
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelProphet(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')
        predictions = model.predict()
        assert len(predictions) == len(df)


class TestCounterfactual:

    def test_counterfactual(self, synthetic_data_single_intervention):
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelOLS(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')
        cf_predictions = model.predict_counterfactual(return_ci=False)
        assert len(cf_predictions) == len(df)

    def test_effect_dataframe(self, synthetic_data_single_intervention):
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelOLS(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')
        effect_df = model.calculate_intervention_effect()

        # 新しい集約フォーマットのカラムを確認
        assert 'Period' in effect_df.columns
        assert 'Actual_mean' in effect_df.columns
        assert 'Predicted_mean' in effect_df.columns
        assert 'Counterfactual_mean' in effect_df.columns
        assert 'Effect_mean' in effect_df.columns

        # 期間が正しく分類されているか確認
        assert 'Pre-intervention' in effect_df['Period'].values
        assert 'Intervention_D_1' in effect_df['Period'].values


class TestVisualizer:

    def test_plot(self, synthetic_data_single_intervention):
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelOLS(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')
        visualizer = ITSVisualizer(model)
        fig = visualizer.plot()
        assert fig is not None


class TestPlaceboCV:
    """プラセボクロスバリデーションのテストクラス"""

    def test_placebo_cv_ols(self, synthetic_data_single_intervention):
        """OLSモデルのプラセボCVテスト"""
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelOLS(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')

        # プラセボCVを実行
        placebo_results = model.placebo_cross_validate(
            df,
            target_column='value',
            n_placebo_points=3
        )

        # 結果の検証
        assert 'placebo_effects' in placebo_results
        assert 'mean_placebo_effect' in placebo_results
        assert 'std_placebo_effect' in placebo_results
        assert 'p_value' in placebo_results
        assert 'is_valid' in placebo_results
        assert len(placebo_results['placebo_effects']) == 3

    def test_placebo_cv_sarimax(self, synthetic_data_single_intervention):
        """SARIMAXモデルのプラセボCVテスト"""
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelSARIMAX(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value', order=(1, 0, 1))

        # プラセボCVを実行
        placebo_results = model.placebo_cross_validate(
            df,
            target_column='value',
            n_placebo_points=2,
            order=(1, 0, 1)
        )

        # 結果の検証
        assert 'placebo_effects' in placebo_results
        assert 'p_value' in placebo_results
        assert len(placebo_results['placebo_effects']) == 2

    def test_placebo_cv_prophet(self, synthetic_data_single_intervention):
        """ProphetモデルのプラセボCVテスト"""
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelProphet(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')

        # プラセボCVを実行
        placebo_results = model.placebo_cross_validate(
            df,
            target_column='value',
            n_placebo_points=2
        )

        # 結果の検証
        assert 'placebo_effects' in placebo_results
        assert 'p_value' in placebo_results
        assert len(placebo_results['placebo_effects']) == 2

    def test_placebo_cv_multiple_interventions(self, synthetic_data_multiple_interventions):
        """複数介入のプラセボCVテスト"""
        df, intervention_points = synthetic_data_multiple_interventions

        model = ITSModelOLS(
            time_column='time',
            intervention_points=intervention_points
        )

        model.fit(df, target_column='value')

        # 複数介入プラセボCVを実行
        placebo_results = model.placebo_cv_multiple_interventions(
            df,
            target_column='value',
            n_placebo_per_intervention=2
        )

        # 結果の検証
        assert isinstance(placebo_results, pd.DataFrame)
        assert 'real_intervention_index' in placebo_results.columns
        assert 'placebo_effect' in placebo_results.columns
        assert len(placebo_results) > 0

    def test_sensitivity_analysis_cv(self, synthetic_data_single_intervention):
        """感度分析CVのテスト"""
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelOLS(
            time_column='time',
            intervention_points=[intervention_point]
        )

        model.fit(df, target_column='value')

        # 感度分析CVを実行
        sensitivity_results = model.sensitivity_analysis_cv(
            df,
            target_column='value',
            time_window_variations=[(10, 10), (20, 10)]
        )

        # 結果の検証
        assert isinstance(sensitivity_results, pd.DataFrame)
        assert 'intervention_point' in sensitivity_results.columns
        assert 'pre_window' in sensitivity_results.columns
        assert 'post_window' in sensitivity_results.columns
        assert 'effect_mean' in sensitivity_results.columns
        assert len(sensitivity_results) > 0


class TestHyperparameterTuning:
    """ハイパーパラメータチューニングのテストクラス"""

    def test_sarimax_tuning_pre_intervention(self, synthetic_data_single_intervention):
        """SARIMAXの介入前データチューニングテスト"""
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelSARIMAX(
            time_column='time',
            intervention_points=[intervention_point]
        )

        # Optunaチューニングを有効にして実行（少ない試行回数）
        model.fit(
            df,
            target_column='value',
            tune_with_optuna=True,
            n_trials=3  # テストなので少なめ
        )

        # チューニング結果の検証
        assert model.best_params is not None
        assert 'order' in model.best_params
        assert 'seasonal_order' in model.best_params
        assert model.model_results is not None

    def test_prophet_tuning_pre_intervention(self, synthetic_data_single_intervention):
        """Prophetの介入前データチューニングテスト"""
        df, intervention_point = synthetic_data_single_intervention

        model = ITSModelProphet(
            time_column='time',
            intervention_points=[intervention_point]
        )

        # Optunaチューニングを有効にして実行（少ない試行回数）
        model.fit(
            df,
            target_column='value',
            tune_with_optuna=True,
            n_trials=3  # テストなので少なめ
        )

        # チューニング結果の検証
        assert model.best_params is not None
        assert 'changepoint_prior_scale' in model.best_params
        assert 'n_changepoints' in model.best_params
        assert model.model_results is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
