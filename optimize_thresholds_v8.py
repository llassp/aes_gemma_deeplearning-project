"""OOF Threshold Brute-Force Optimization for V8 Ordinal Model.

Phase 4: Takes continuous OOF predictions (from predict_oof.py) and finds
optimal cut-point thresholds to map continuous scores [1.0, 6.0] to discrete
integer scores [1, 6]. Uses Nelder-Mead + brute-force sweep on the 5→6
boundary to combat high-score collapse.

Usage:
    python optimize_thresholds_v8.py --oof_csv outputs/oof/v8_oof.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


def qwk(y_true, y_pred) -> float:
    """Quadratic Weighted Kappa."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def digitize_scores(continuous: np.ndarray, thresholds: list) -> np.ndarray:
    """Convert continuous scores to discrete 1–6 using thresholds.

    thresholds: 4 cut points between [1.5, 2.5, 3.5, 4.5] defining
    boundaries between scores. np.digitize returns bin indices 0-based,
    so we add 1 to get scores 1–6.
    """
    return np.clip(np.digitize(continuous, thresholds) + 1, 1, 6)


def optimize_thresholds(
    oof_predictions: np.ndarray, y_true: np.ndarray
) -> list:
    """Run Nelder-Mead then brute-force the 5→6 boundary.

    Args:
        oof_predictions: continuous scores from the model, e.g. [1.2, 4.8, ...]
        y_true: ground-truth integer scores in [1, 6]

    Returns:
        list of 4 threshold values: [1→2, 2→3, 3→4, 4→5, 5→6 cut points]
        Actually returns 4 thresholds for the 5 boundaries between 6 classes.
    """
    # 4 cut points between 5 score boundaries (1|2, 2|3, 3|4, 4|5, 5|6)
    # np.digitize uses right edges, so thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]
    # But 5 boundaries need only 4 thresholds (the 5→6 boundary is threshold[4]
    # in a 5-threshold list). Let's use 5 thresholds for 6 bins.
    initial_thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]  # 5 cut points

    def loss_func(th):
        th = np.array(th)
        # Enforce monotonicity
        if not np.all(th[1:] > th[:-1]):
            return 1.0
        preds = np.digitize(oof_predictions, th) + 1
        preds = np.clip(preds, 1, 6)
        return -qwk(y_true, preds)

    print("Running Nelder-Mead optimization on 5 thresholds...")
    res = minimize(
        loss_func,
        initial_thresholds,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4},
    )
    best_th = res.x.tolist()
    current_best_qwk = -res.fun

    print(f"Nelder-Mead Best QWK: {current_best_qwk:.5f}")
    print(
        f"Base Thresholds: "
        f"[{', '.join(f'{t:.4f}' for t in best_th)}]"
    )

    # ── Brute-force the 5→6 boundary (last threshold) ──
    print("\nBrute-forcing the 5→6 boundary (threshold[4])...")
    best_th_56 = best_th[4]

    for t in np.arange(4.50, 5.80, 0.01):
        test_th = best_th.copy()
        test_th[4] = t
        if test_th[3] >= test_th[4]:
            continue

        preds = digitize_scores(oof_predictions, test_th)
        score = qwk(y_true, preds)

        if score > current_best_qwk:
            current_best_qwk = score
            best_th_56 = t
            print(f"  [Improved] th_56={t:.3f} -> QWK={score:.5f}")

    best_th[4] = best_th_56

    # Also brute-force the 1→2 boundary (first threshold)
    print("\nBrute-forcing the 1→2 boundary (threshold[0])...")
    best_th_12 = best_th[0]
    for t in np.arange(1.20, 2.30, 0.01):
        test_th = best_th.copy()
        test_th[0] = t
        if test_th[0] >= test_th[1]:
            continue

        preds = digitize_scores(oof_predictions, test_th)
        score = qwk(y_true, preds)

        if score > current_best_qwk:
            current_best_qwk = score
            best_th_12 = t
            print(f"  [Improved] th_12={t:.3f} -> QWK={score:.5f}")

    best_th[0] = best_th_12

    print(f"\nFinal Optimized Thresholds: "
          f"[{', '.join(f'{t:.4f}' for t in best_th)}]")
    print(f"Final QWK: {current_best_qwk:.5f}")

    # Print score distribution with these thresholds
    final_preds = digitize_scores(oof_predictions, best_th)
    for s in range(1, 7):
        n = (final_preds == s).sum()
        n_true = (y_true == s).sum()
        print(f"  Score {s}: predicted={n}, true={n_true}")

    return best_th


def main():
    parser = argparse.ArgumentParser(
        description="OOF Threshold Optimization for V8 Ordinal Model"
    )
    parser.add_argument(
        "--oof_csv",
        type=Path,
        required=True,
        help="OOF predictions CSV from predict_oof.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for thresholds JSON (optional)",
    )
    args = parser.parse_args()

    # Load OOF data
    continuous_scores = []
    true_scores = []
    with args.oof_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            continuous_scores.append(float(row["continuous_score"]))
            if "true_score" in row and row["true_score"]:
                true_scores.append(float(row["true_score"]))

    oof_arr = np.array(continuous_scores)
    true_arr = np.array(true_scores) if true_scores else None

    print(f"Loaded {len(oof_arr)} OOF predictions")
    print(
        f"Continuous score stats: "
        f"mean={oof_arr.mean():.3f}, std={oof_arr.std():.3f}, "
        f"min={oof_arr.min():.3f}, max={oof_arr.max():.3f}"
    )

    if true_arr is None:
        print("ERROR: No true_score column in OOF CSV. Cannot optimize thresholds.")
        return

    thresholds = optimize_thresholds(oof_arr, true_arr)

    if args.output:
        import json
        with args.output.open("w") as f:
            json.dump(
                {
                    "thresholds": thresholds,
                    "description": "Cut points for np.digitize. Scores = digitize(continuous, thresholds) + 1, clamped to [1, 6]",
                },
                f,
                indent=2,
            )
        print(f"\nThresholds saved to {args.output}")


if __name__ == "__main__":
    main()
