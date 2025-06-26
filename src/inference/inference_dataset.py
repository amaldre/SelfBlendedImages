import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff()
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]
    for filename in tqdm(video_list):
        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                pred=model(img).softmax(1)[:,1]


            pred_list=[]
            idx_img=-1
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)

    # Convert output probabilities to binary predictions using a threshold (e.g., 0.5)
    binary_predictions = [1 if p >= 0.5 else 0 for p in output_list]

    # Calculate Accuracy
    accuracy = accuracy_score(target_list, binary_predictions)

    # Calculate Average Precision
    avg_precision = average_precision_score(target_list, output_list)

    # Calculate Average Recall
    avg_recall = recall_score(target_list, binary_predictions)


    print(f'{args.dataset}| AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Avg Precision: {avg_precision:.4f}, Avg Recall: {avg_recall:.4f}')

    # --- ROC Curve Plot ---
    fpr, tpr, thresholds = roc_curve(target_list, output_list)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {args.dataset}')
    plt.legend(loc="lower right")
    roc_plot_filename = f'ROC_Curve_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(roc_plot_filename)
    plt.close()
    print(f"ROC curve saved to {roc_plot_filename}")

    # --- BPCER vs. APCER Plot ---
    # APCER is FPR, BPCER is FNR (1 - TPR)
    bpcer = 1 - tpr
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, bpcer, color='red', lw=2, label='BPCER vs. APCER')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('APCER (False Positive Rate)')
    plt.ylabel('BPCER (False Negative Rate)')
    plt.title(f'BPCER vs. APCER for {args.dataset}')
    plt.legend(loc="upper right")
    bpcer_apcer_plot_filename = f'BPCER_APCER_Plot_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(bpcer_apcer_plot_filename)
    plt.close()
    print(f"BPCER vs. APCER plot saved to {bpcer_apcer_plot_filename}")


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)