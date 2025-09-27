"""
ITS Analysis Package のテストコード

各クラスの機能を検証するユニットテストと結合テストを提供します。
"""

from src.module.its_analysis import ITSDataPreprocessor, ITSModel, ITSVisualizer
import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt

# テスト対象のモジュールをインポート
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestITSDataPreprocessor(unittest.TestCase):
    """ITSDataPreprocessorクラスのテスト"""

    def setUp(self):
        """テスト用データの準備"""
        # シンプルなテストデータを作成
        np.random.seed(42)

        # 単一グループのテストデータ
        self.simple_data = pd.DataFrame({
            'time': range(1, 21),  # 1-20の時間
            'value': np.random.normal(10, 2, 20) + np.arange(20) * 0.1,
            'group': ['A'] * 20
        })

        # 複数グループのテストデータ
        self.multi_group_data = pd.DataFrame({
            'time': list(range(1, 21)) * 2,
            'value': np.random.normal(10, 2, 40) + np.tile(np.arange(20) * 0.1, 2),
            'group': ['A'] * 20 + ['B'] * 20
        })

        # 介入点は時間10
        self.intervention_point = 10

    def test_init(self):
        """初期化のテスト"""
        preprocessor = ITSDataPreprocessor(
            time_column='time',
            intervention_point=self.intervention_point
        )

        self.assertEqual(preprocessor.time_column, 'time')
        self.assertEqual(preprocessor.intervention_point,
                         self.intervention_point)
        self.assertIsNone(preprocessor.group_column)
        self.assertIsNone(preprocessor.processed_data)

    def test_fit_transform_simple(self):
        """単純なデータの変換テスト"""
        preprocessor = ITSDataPreprocessor(
            time_column='time',
            intervention_point=self.intervention_point
        )

        result = preprocessor.fit_transform(self.simple_data)

        # 必要な列が作成されているかチェック
        required_cols = ['t', 'T', 'D', 'time_after', 'const']
        for col in required_cols:
            self.assertIn(col, result.columns)

        # 時間変数のチェック
        self.assertEqual(result['t'].min(), 1)
        self.assertEqual(result['t'].max(), 20)

        # 介入点ダミーのチェック
        self.assertEqual(result['T'].sum(), 1)  # 介入点は1つだけ
        self.assertEqual(
            result[result['time'] == self.intervention_point]['T'].iloc[0], 1)

        # 介入後ダミーのチェック
        pre_intervention_count = (
            result['time'] < self.intervention_point).sum()
        post_intervention_count = (
            result['time'] >= self.intervention_point).sum()
        self.assertEqual(result['D'].sum(), post_intervention_count)

        # 切片項のチェック
        self.assertTrue((result['const'] == 1).all())

    def test_fit_transform_with_groups(self):
        """グループありデータの変換テスト"""
        preprocessor = ITSDataPreprocessor(
            time_column='time',
            intervention_point=self.intervention_point,
            group_column='group'
        )

        result = preprocessor.fit_transform(self.multi_group_data)

        # グループ別に時間変数が正しく作成されているかチェック
        for group in ['A', 'B']:
            group_data = result[result['group'] == group]
            self.assertEqual(group_data['t'].min(), 1)
            self.assertEqual(group_data['t'].max(), 20)

    def test_validate_data(self):
        """データ妥当性チェックのテスト"""
        preprocessor = ITSDataPreprocessor(
            time_column='time',
            intervention_point=self.intervention_point
        )

        # 正常なデータのテスト
        self.assertTrue(preprocessor.validate_data(self.simple_data))

        # 時間列が存在しない場合
        invalid_data = self.simple_data.drop('time', axis=1)
        with self.assertRaises(ValueError):
            preprocessor.validate_data(invalid_data)

        # 介入点が存在しない場合
        preprocessor_invalid = ITSDataPreprocessor(
            time_column='time',
            intervention_point=99  # 存在しない時間点
        )
        with self.assertRaises(ValueError):
            preprocessor_invalid.validate_data(self.simple_data)

    def test_get_its_variables(self):
        """ITS変数リスト取得のテスト"""
        preprocessor = ITSDataPreprocessor(
            time_column='time',
            intervention_point=self.intervention_point
        )

        # 処理前
        vars_before = preprocessor.get_its_variables()
        self.assertNotIn('const', vars_before)

        # 処理後
        preprocessor.fit_transform(self.simple_data)
        vars_after = preprocessor.get_its_variables()
        self.assertIn('const', vars_after)


class TestITSModel(unittest.TestCase):
    """ITSModelクラスのテスト"""

    def setUp(self):
        """テスト用データの準備"""
        np.random.seed(42)

        # より現実的なテストデータを作成（介入効果あり）
        n_pre = 15
        n_post = 15

        # ベースライン + トレンド + 介入効果 + ノイズ
        pre_time = np.arange(1, n_pre + 1)
        post_time = np.arange(n_pre + 1, n_pre + n_post + 1)

        baseline = 10
        trend = 0.2
        intervention_level_change = 5
        intervention_trend_change = 0.3

        pre_values = baseline + trend * \
            pre_time + np.random.normal(0, 1, n_pre)
        post_values = (baseline + intervention_level_change +
                       trend * post_time +
                       intervention_trend_change * (post_time - n_pre) +
                       np.random.normal(0, 1, n_post))

        self.test_data = pd.DataFrame({
            'time': np.concatenate([pre_time, post_time]),
            'value': np.concatenate([pre_values, post_values]),
            'covariate': np.random.normal(0, 1, n_pre + n_post)
        })

        self.intervention_point = n_pre + 1

    def test_fit(self):
        """モデルフィッティングのテスト"""
        model = ITSModel(
            time_column='time',
            intervention_point=self.intervention_point
        )

        # 共変量なしでフィット
        result = model.fit(self.test_data, target_column='value')

        self.assertIsNotNone(result)
        self.assertIsNotNone(model.model_results)
        self.assertIsNotNone(model.feature_columns)
        self.assertEqual(model.target_column, 'value')

    def test_fit_with_covariates(self):
        """共変量ありでのフィッティングテスト"""
        model = ITSModel(
            time_column='time',
            intervention_point=self.intervention_point
        )

        # 共変量ありでフィット
        result = model.fit(
            self.test_data,
            target_column='value',
            covariates=['covariate']
        )

        self.assertIn('covariate', model.feature_columns)

    def test_predict(self):
        """予測のテスト"""
        model = ITSModel(
            time_column='time',
            intervention_point=self.intervention_point
        )

        model.fit(self.test_data, target_column='value')

        # 学習データでの予測
        predictions = model.predict()
        self.assertEqual(len(predictions), len(self.test_data))

        # 信頼区間付き予測
        predictions_ci = model.predict(return_ci=True)
        self.assertIn('mean_ci_lower', predictions_ci.columns)
        self.assertIn('mean_ci_upper', predictions_ci.columns)

    def test_save_and_load_model(self):
        """モデル保存・読み込みのテスト"""
        model = ITSModel(
            time_column='time',
            intervention_point=self.intervention_point
        )

        model.fit(self.test_data, target_column='value')

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            model.save_model(tmp_file.name)

            # 新しいモデルインスタンスで読み込み
            new_model = ITSModel(
                time_column='dummy',  # 異なる値で初期化
                intervention_point=999
            )

            new_model.load_model(tmp_file.name)

            # 元のモデルと同じ設定になっているかチェック
            self.assertEqual(new_model.time_column, 'time')
            self.assertEqual(new_model.intervention_point,
                             self.intervention_point)
            self.assertEqual(new_model.target_column, 'value')

            # 予測結果が同じかチェック
            original_pred = model.predict()
            loaded_pred = new_model.predict()
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        # 一時ファイルを削除
        os.unlink(tmp_file.name)


class TestITSVisualizer(unittest.TestCase):
    """ITSVisualizerクラスのテスト"""

    def setUp(self):
        """テスト用モデルの準備"""
        np.random.seed(42)

        # テストデータ作成
        n_points = 30
        self.test_data = pd.DataFrame({
            'time': range(1, n_points + 1),
            'value': np.random.normal(10, 2, n_points) + np.arange(n_points) * 0.1,
            'group': ['A'] * 15 + ['B'] * 15
        })

        self.intervention_point = 15

        # モデルを事前にフィット
        self.model = ITSModel(
            time_column='time',
            intervention_point=self.intervention_point,
            group_column='group'
        )
        self.model.fit(self.test_data, target_column='value')

    def test_init(self):
        """初期化のテスト"""
        visualizer = ITSVisualizer(self.model)
        self.assertEqual(visualizer.model, self.model)

        # フィットしていないモデルでの初期化
        unfitted_model = ITSModel('time', 10)
        with self.assertRaises(ValueError):
            ITSVisualizer(unfitted_model)

    def test_plot_basic(self):
        """基本的なプロット機能のテスト"""
        visualizer = ITSVisualizer(self.model)

        # プロット作成（保存なし）
        fig = visualizer.plot()

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)  # メモリリークを防ぐ

    def test_plot_with_save(self):
        """画像保存機能のテスト"""
        visualizer = ITSVisualizer(self.model)

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            fig = visualizer.plot(save_path=tmp_file.name)

            # ファイルが作成されているかチェック
            self.assertTrue(os.path.exists(tmp_file.name))
            self.assertGreater(os.path.getsize(tmp_file.name), 0)

        # 一時ファイルを削除
        os.unlink(tmp_file.name)
        plt.close(fig)

    def test_plot_options(self):
        """プロットオプションのテスト"""
        visualizer = ITSVisualizer(self.model)

        # さまざまなオプションでプロット
        fig = visualizer.plot(
            group_column='group',
            figsize=(10, 6),
            show_counterfactual=True,
            show_confidence_interval=True,
            alpha=0.5
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestIntegration(unittest.TestCase):
    """統合テスト"""

    def test_full_workflow(self):
        """完全なワークフローのテスト"""
        # 1. データの準備
        np.random.seed(42)
        data = pd.DataFrame({
            'year': list(range(2000, 2020)) * 2,
            'sales': np.random.normal(100, 10, 40) + np.tile(np.arange(20), 2),
            'state': ['CA'] * 20 + ['NY'] * 20
        })

        # 2. 前処理
        preprocessor = ITSDataPreprocessor(
            time_column='year',
            intervention_point=2010,
            group_column='state'
        )

        processed_data = preprocessor.fit_transform(data)

        # 3. モデリング
        model = ITSModel(
            time_column='year',
            intervention_point=2010,
            group_column='state'
        )

        model_result = model.fit(processed_data, target_column='sales',
                                 covariates=['state'])

        # 4. 可視化
        visualizer = ITSVisualizer(model)
        fig = visualizer.plot(group_column='state')

        # 5. 保存・読み込み
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_model:
            model.save_model(tmp_model.name)

            new_model = ITSModel('dummy', 999)
            new_model.load_model(tmp_model.name)

            # 予測結果の一致を確認
            original_pred = model.predict()
            loaded_pred = new_model.predict()
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        # クリーンアップ
        os.unlink(tmp_model.name)
        plt.close(fig)

        # 全ステップが正常に完了したことを確認
        self.assertIsNotNone(processed_data)
        self.assertIsNotNone(model_result)
        self.assertIsInstance(fig, plt.Figure)


if __name__ == '__main__':
    # テストの実行
    unittest.main(verbosity=2)
