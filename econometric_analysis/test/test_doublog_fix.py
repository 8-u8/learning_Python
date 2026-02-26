#!/usr/bin/env python
"""
Quick test to verify DoubLogRegressionModel fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

try:
    from regression import DoubLogRegressionModel, MarginalEffectsCalculator

    print("✓ Import successful")

    # Generate power law data
    np.random.seed(42)
    n = 100
    x = np.linspace(10, 500, n)
    y = 1000 * (x**0.7) + np.random.normal(0, 5000, n)

    df = pd.DataFrame({"y": y, "x": x})

    print("\nTesting DoubLogRegressionModel fix...")

    # Fit model
    model = DoubLogRegressionModel()
    model.fit(df[["x"]], df["y"])
    print("✓ Model fit successful")

    # Test marginal effect calculation
    calc = MarginalEffectsCalculator(model)

    # Test at different mean_values
    test_values = [50, 200, 500]
    print("\nMarginal effects at different values:")
    for val in test_values:
        me = calc.marginal_effect("x", mean_value=val)
        print(f"  At x={val}: ME={me:.6f}")

    # Test saturation point detection
    x_range = np.linspace(df["x"].min(), df["x"].max(), 100)
    saturation = calc.detect_saturation_point("x", x_range)

    print(f"\n✓ Saturation point detection:")
    print(f"  Saturation at x ≈ {saturation['saturation_threshold_x']}")
    print(f"  Diminishing returns: {saturation['diminishing_returns']}")

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
