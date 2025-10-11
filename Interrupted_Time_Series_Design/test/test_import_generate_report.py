"""
generate_report.pyのインポートテスト
"""

from module import generate_markdown_report
import sys
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# moduleからインポート

if __name__ == "__main__":
    print("✅ module.generate_reportのインポート成功！")
    print("📝 レポート生成を開始...")

    # レポート生成
    report_path = generate_markdown_report()

    print(f"✅ レポート生成完了: {report_path}")
