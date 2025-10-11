"""
Interrupted Time Series Analysis Package

このパッケージは、断続的時系列分析（Interrupted Time Series Design）を実施するための
統合モジュールです。
"""

from .its_analysis import (
    ITSDataPreprocessor,
    ITSModelBase,
    ITSModelOLS,
    ITSModelSARIMAX,
    ITSModelProphet,
    ITSVisualizer
)

from .generate_report import generate_markdown_report

__all__ = [
    'ITSDataPreprocessor',
    'ITSModelBase',
    'ITSModelOLS',
    'ITSModelSARIMAX',
    'ITSModelProphet',
    'ITSVisualizer',
    'generate_markdown_report'
]

__version__ = '1.0.0'
