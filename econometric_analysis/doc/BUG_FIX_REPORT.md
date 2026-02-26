# 🔧 バグ修正ログ - GAMRegressionModel

## 🐛 発見されたバグ

**日時**: 2026年2月26日
**影響範囲**: `GAMRegressionModel.fit()` メソッド
**状態**: ✅ **FIXED**

---

## ❌ エラー内容

```
ValueError: terms must be instances of Term or TermList, but found term: 0
```

**発生箇所**:
```python
File "src/regression.py", line 683, in fit
    gam_formula = sum(s(i, lam=self.lam) for i in range(X.shape[1]))
```

---

## 🔍 原因分析

### 問題のコード
```python
gam_formula = sum(s(i, lam=self.lam) for i in range(X.shape[1]))
```

### なぜエラーが起きたのか？

1. `sum()` 関数は初期値が **0** から始まる
2. PyGAM の `s()` 関数は `Term` オブジェクトを返す
3. `Term` オブジェクトと数値 0 の加算は定義されていない
4. つまり: `0 + s(0) + s(1) + ...` という計算が起き、`0 + Term` が失敗する

### PyGAMの制限
- PyGAM の `Term` オブジェクトは異なる型（数値など）との加算をサポートしない
- 最初のTermから開始する必要がある

---

## ✅ 修正内容

### 修正前（エラー）
```python
gam_formula = sum(s(i, lam=self.lam) for i in range(X.shape[1]))
```

### 修正後（正常動作）
```python
n_features = X.shape[1]
if n_features == 1:
    gam_formula = s(0, lam=self.lam)
else:
    gam_formula = s(0, lam=self.lam)
    for i in range(1, n_features):
        gam_formula = gam_formula + s(i, lam=self.lam)
```

### 修正のポイント
1. ✅ 最初のスプライン関数 `s(0)` を明示的に作成
2. ✅ 残りのスプライン関数を順番に加算
3. ✅ 単一特徴量の場合も対応

---

## 📊 修正例

### 単一特徴量（1列）の場合
```python
# Before (エラー)
gam_formula = sum([s(0)])  # 0 + s(0) → エラー!

# After (成功)
gam_formula = s(0)  # OK!
```

### 複数特徴量（3列）の場合
```python
# Before (エラー)
gam_formula = sum([s(0), s(1), s(2)])  # 0 + s(0) + s(1) + s(2) → エラー!

# After (成功)
gam_formula = s(0) + s(1) + s(2)  # OK!
```

---

## 🧪 テスト検証

### 修正後のテスト状況

| テスト項目 | 状態 | 備考 |
|-----------|------|------|
| 単一特徴量（demo.py） | ✅ PASS | `x` 1列のみ |
| 複数特徴量（verify_extended.py） | ✅ PASS | `x1, x2` 2列 |
| テストケース（test_regression.py） | ✅ PASS | 19個すべて |

---

## 📝 修正したファイル

- **src/regression.py**
  - 行: 675-693
  - メソッド: `GAMRegressionModel.fit()`
  - 変更行数: 9行

---

## 🚀 修正後の実行

### デモスクリプト
```bash
python demo.py
```

### 検証スクリプト
```bash
python verify_extended.py
```

### テスト実行
```bash
pytest test/test_regression.py -v
```

---

## ✨ 修正の影響

✅ **正の影響**
- GAMモデルが正常に動作するようになった
- 単一・複数の特徴量両方に対応
- デモとテストが全て実行可能

❌ **負の影響**
- なし（後方互換性も維持）

---

## 📌 まとめ

| 項目 | 内容 |
|------|------|
| **バグタイプ** | 外部ライブラリ（PyGAM）との互換性問題 |
| **重要度** | 🔴 **Critical** - 機能完全に停止 |
| **修正難度** | 🟢 **Easy** - 3行の修正 |
| **テスト状況** | ✅ 全テスト成功 |
| **ステータス** | ✅ **RESOLVED** |

---

**修正完了日**: 2026年2月26日
**修正者**: GitHub Copilot（イチカ）
**検証状況**: ✅ 本番環境対応可能
