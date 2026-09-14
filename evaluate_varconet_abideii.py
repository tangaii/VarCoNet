"""Evaluate the ten ABIDE-I external-test VarCoNet models on ABIDE II.

This is the VarCoNet-only equivalent of ``ASD_classification_ABIDEII.py``.
It deliberately omits BolT, whose external baseline package is not part of
this repository, while retaining the model loading and metric definitions.
"""

import argparse
import glob
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from model_scripts.VarCoNet import VarCoNet
from model_scripts.classifier import MLP
from utils import ABIDEDataset


def main(args):
    data_npz = np.load(os.path.join(args.path_data, f"ABIDEII_nilearn_{args.atlas}.npz"))
    data = [data_npz[key] for key in data_npz]
    labels = np.load(os.path.join(args.path_data, "ABIDEII_nilearn_classes.npy"))
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    loader = DataLoader(ABIDEDataset(data, labels), batch_size=64)

    with open(f"best_params_VarCoNet_{args.atlas}.pkl", "rb") as handle:
        params = pickle.load(handle)
    model_config = {
        "layers": params["layers"],
        "n_heads": params["n_heads"],
        "dim_feedforward": params["dim_feedforward"],
        "max_length": data[0].shape[0],
    }
    roi_num = data[0].shape[1]
    result = {"test_losses": [], "test_aucs": [], "test_probs": [], "y_test": labels}
    models_root = args.models_root or args.path_save
    for seed_index in range(10):
        encoder = VarCoNet(model_config, roi_num).to(device)
        classifier = MLP(roi_num * (roi_num - 1) // 2, 2).to(device)
        candidates = glob.glob(os.path.join(models_root, "run_*", "models_ABIDEI", args.atlas, "VarCoNet", f"min_val_loss_model_rs{seed_index}.pth"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one model for repeat {seed_index}; found {candidates}")
        models_dir = os.path.dirname(candidates[0])
        encoder.load_state_dict(torch.load(os.path.join(models_dir, f"min_val_loss_model_rs{seed_index}.pth"), map_location=device))
        classifier.load_state_dict(torch.load(os.path.join(models_dir, f"min_val_loss_classifier_rs{seed_index}.pth"), map_location=device))
        encoder.eval(); classifier.eval()
        embeddings, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                embeddings.append(encoder(x.to(device)))
                ys.append(y.to(device))
            y = torch.cat(ys)
            probs = classifier(torch.cat(embeddings))
            loss = nn.BCELoss()(probs, F.one_hot(y, num_classes=2).float()).item()
            score = probs[:, -1].cpu().numpy()
        result["test_losses"].append(loss)
        result["test_aucs"].append(roc_auc_score(y.cpu().numpy(), score))
        result["test_probs"].append(score)

    output_dir = os.path.join(args.path_save, "results_ABIDEII", args.atlas)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ABIDEII_VarCoNet_results.pkl"), "wb") as handle:
        pickle.dump(result, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", required=True)
    parser.add_argument("--path_save", required=True)
    parser.add_argument("--atlas", choices=["AAL", "AICHA"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models_root", default=None)
    main(parser.parse_args())
