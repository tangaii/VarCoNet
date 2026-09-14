"""Create the paper-versus-reproduction VarCoNet results table."""

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_curve


PAPER = {
    "AAL": {"cv": (0.633, 69.91, 67.54), "caltech": (0.653, 67.22, 73.63), "abide2": (0.639, 68.42, 61.57)},
    "AICHA": {"cv": (0.612, 72.62, 68.47), "caltech": (0.650, 67.02, 71.41), "abide2": (0.609, 73.10, 65.01)},
}


def f1_at_youden(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores)
    threshold = thresholds[np.argmax(tpr - fpr)]
    return f1_score(y, scores >= threshold) * 100


def display(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} ± {values.std():.3f}"


def select_epoch(results):
    selected = []
    for folds_epochs in results:
        selected.append(min(folds_epochs, key=lambda x: x["best_val_loss"]))
    return selected


def cv_or_caltech(selected):
    losses = [entry["best_test_loss"] for entry in selected]
    aucs = [entry["best_test_auc"] * 100 for entry in selected]
    f1s = [f1_at_youden(entry["y_test"], entry["test_probs"]) for entry in selected]
    return display(losses), display(aucs), display(f1s)


def abide2(result):
    losses = result["test_losses"]
    aucs = np.asarray(result["test_aucs"]) * 100
    f1s = [f1_at_youden(result["y_test"], x) for x in result["test_probs"]]
    return display(losses), display(aucs), display(f1s)


root = Path("reproduction_results")
lines = ["| Atlas | Evaluation | Paper BCE / AUC / F1 | Reproduction BCE / AUC / F1 |", "|---|---|---|---|"]
for atlas in ("AAL", "AICHA"):
    base = root / atlas
    per_run = []
    result_paths = sorted(base.glob(f"run_*/results_ABIDEI/{atlas}/ABIDEI_VarCoNet_results.pkl"))
    if len(result_paths) != 8:
        raise RuntimeError(f"Expected 8 shard result files for {atlas}, found {len(result_paths)}")
    for result_path in result_paths:
        with open(result_path, "rb") as handle:
            per_run.append(pickle.load(handle))
    abide1 = {
        "epoch_results": sum((x["epoch_results"] for x in per_run), []),
        "epoch_results_ext": sum((x["epoch_results_ext"] for x in per_run), []),
    }
    with open(base / "results_ABIDEII" / atlas / "ABIDEII_VarCoNet_results.pkl", "rb") as handle:
        abide2_result = pickle.load(handle)
    ours = {
        "cv": cv_or_caltech(select_epoch(abide1["epoch_results"])),
        "caltech": cv_or_caltech(select_epoch(abide1["epoch_results_ext"])),
        "abide2": abide2(abide2_result),
    }
    for name, label in (("cv", "ABIDE I: repeated 10-fold CV"), ("caltech", "ABIDE I: Caltech"), ("abide2", "ABIDE II: external")):
        ref = PAPER[atlas][name]
        ref_text = f"{ref[0]:.3f} / {ref[1]:.2f}% / {ref[2]:.2f}%"
        ours_text = " / ".join(ours[name][:1]) + " / " + " / ".join(x + "%" for x in ours[name][1:])
        lines.append(f"| {atlas} | {label} | {ref_text} | {ours_text} |")

output = root / "final_comparison_table.md"
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(output)
