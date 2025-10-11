"""
Test suite for Interrupted Time Series Analysis Package
"""


import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from module.its_analysis import (
    ITSDataPreprocessor,
    ITSModelOLS,
    ITSModelSARIMAX,
    ITSModelProphet,
    ITSVisualizer
)

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


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
