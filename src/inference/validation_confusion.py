import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from tqdm import tqdm
import warnings
import pickle
import seaborn as sns  # For confusion matrix heatmap
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from datasets import *
from datasets import CROP_DIR
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_curve,
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.sbi import SBI_Custom_Dataset
from model import Detector

DATASETS = []
def main(args):
    device = torch.device('cuda')
    val_dataset = SBI_Custom_Dataset('val', DATASETS, 
                                     image_size = 380, degradations = False, poisson = True, random_mask = True, crop_mode = 'retina')
    val_loader = DataLoader(val_dataset,
                        batch_size=16,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
    print("------Inference mode------")
    print(f"Testing model {os.path.basename(args.weight_name)}")

    try:
        backbone = torch.load(args.weight_name)["backbone"]
    except:
        print("No backbone detected, defaulting to efficientnet-b4")
        backbone = "efficientnet-b4"

    model = Detector(backbone=backbone, phase='not train')
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    output_list = []
    target_list = []
    count = 0
    for step, data in enumerate(tqdm(val_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()

            with torch.no_grad():
                output = model(img)

            #iter_loss.append(loss_value)
            output_list += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
            target_list += target.cpu().data.numpy().tolist()

    you_auc = True
    try:
        auc = roc_auc_score(target_list, output_list)
    except:
        auc = None
        you_auc = False

    # Convert output probabilities to binary predictions using threshold 0.5
    binary_predictions = [1 if p >= 0.5 else 0 for p in output_list]

    # Metrics
    accuracy = accuracy_score(target_list, binary_predictions)
    avg_precision = average_precision_score(target_list, output_list)
    avg_recall = recall_score(target_list, binary_predictions)

    # Confusion Matrix
    if args.confmat:
        cm = confusion_matrix(target_list, binary_predictions)
        print("Confusion Matrix:")
        print(cm)
        if True:
            cm_dir = os.path.join('figures', 'confusion_matrix')
            os.makedirs(cm_dir, exist_ok=True)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for {args.dataset}')
            cm_plot_filename = f'Confusion_Matrix_{args.dataset}_{os.path.basename(args.weight_name)}.png'
            plt.savefig(os.path.join(cm_dir, cm_plot_filename))
            plt.close()
            print(f"Confusion matrix plot saved to {cm_plot_filename}")

    if you_auc:
        print(f'{args.dataset}| AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Avg Precision: {avg_precision:.4f}, Avg Recall: {avg_recall:.4f}')
    else:
        print(f'{args.dataset}| AUC: N/A (only one label), Accuracy: {accuracy:.4f}, Avg Precision:  N/A (only one label), Avg Recall:  N/A (only one label)')

    if args.plot and you_auc:
        plot_dir = "figures"

        # ROC Curve Plot
        fpr, tpr, thresholds = roc_curve(target_list, output_list)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve for {args.dataset}')
        plt.legend(loc="lower right")
        roc_plot_filename = f'ROC_Curve_{args.dataset}_{os.path.basename(args.weight_name)}.png'
        os.makedirs(os.path.join(plot_dir, 'ROC'), exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'ROC', roc_plot_filename))
        plt.close()
        print(f"ROC curve saved to {roc_plot_filename}")

        # BPCER vs APCER Plot
        bpcer_values = fpr  # BPCER is FPR
        apcer_values = 1 - tpr  # APCER is FNR (1 - TPR)
        plt.figure(figsize=(8, 6))
        plt.plot(bpcer_values, apcer_values, color='red', lw=2, label='APCER vs. BPCER')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('BPCER (False Positive Rate)')
        plt.ylabel('APCER (False Negative Rate)')
        plt.title(f'APCER vs. BPCER for {args.dataset}')
        plt.legend(loc="upper right")
        bpcer_apcer_plot_filename = f'APCER_BPCER_Plot_{args.dataset}_{os.path.basename(args.weight_name)}.png'
        os.makedirs(os.path.join(plot_dir, 'bpcer_apcer'), exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'bpcer_apcer', bpcer_apcer_plot_filename))
        plt.close()
        print(f"APCER vs. BPCER plot saved to {bpcer_apcer_plot_filename}")

    return auc, accuracy, avg_precision, avg_recall, target_list, output_list


if __name__ == '__main__':

    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weight_name', type=str)
    parser.add_argument('-plot', dest='plot', action='store_true', default=False)
    parser.add_argument('-print', dest='print', action='store_true', default=False)
    parser.add_argument('-confmat', dest='confmat', action='store_true', default=False, help="Enable confusion matrix print and plot")
    parser.add_argument('-crop_mode', dest='crop_mode')
    args = parser.parse_args()

    main(args)
