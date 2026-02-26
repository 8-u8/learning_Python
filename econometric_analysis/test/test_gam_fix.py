#!/usr/bin/env python
"""
Quick test for the GAM model fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

try:
    from regression import GAMRegressionModel

    print("✓ Import successful")

    # Generate simple test data
    np.random.seed(42)
    n = 100
    x = np.linspace(1, 100, n)
    y = 50 + 10 * np.sqrt(x) + np.random.normal(0, 5, n)

    df = pd.DataFrame({"y": y, "x": x})

    print("\nTesting GAM Model...")
    gam = GAMRegressionModel(lam=0.6)
    gam.fit(df[["x"]], df["y"])

    print(f"✓ GAM fit successful")
    print(f"  R² = {gam.r_squared():.6f}")

    predictions = gam.predict(df[["x"]])
    print(f"✓ Predictions generated: {len(predictions)} samples")

    print("\n✓ All GAM tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
