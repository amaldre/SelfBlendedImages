import cv2
cv2.setNumThreads(0)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import numpy as np
import os
from PIL import Image
import sys
import random
from utils.sbi import SBI_Dataset, SBI_Custom_Dataset, SourceConcat
from utils.scheduler import LinearDecayLR, LinearDecayLR_LaaNet, FlatCosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector


import wandb

from inference.inference_dataset import main as infer
from inference.datasets import *

import matplotlib.pyplot as plt

# def get_degraded_batch(img_batch, image_list, path_lm, device):
#     degraded_list = []
#     for i in range(img_batch.size(0)):
#         single_img_tensor = img_batch[i]             # (C, H, W)
    
#         single_img_np = single_img_tensor.cpu().numpy()  # (C, H, W)
#         single_img_np = np.transpose(single_img_np, (1, 2, 0))  # (H, W, C) si nécessaire
#         degraded_img_np = degradation(single_img_np, image_list, path_lm)
    
#         # Reconvertir en tensor (C, H, W)
#         degraded_img_np = np.transpose(degraded_img_np, (2, 0, 1))  # (C, H, W)
#         degraded_img_tensor = torch.from_numpy(degraded_img_np).to(device).float()

#         degraded_list.append(degraded_img_tensor)

#         # Empiler pour obtenir un batch final
#     degraded_img_batch = torch.stack(degraded_list, dim=0) 
#     return degraded_img_batch

def test(model_path, dataset, plot_bool, crop_mode):
    args = argparse.Namespace(
    weight_name=model_path,
    dataset=dataset,
    plot = plot_bool, 
    confmat = False,
    crop_mode = crop_mode
    )
    return infer(args)

def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(args):
    cfg=load_json(args.config)
    USE_WANDB = cfg['use_wandb'] == 1
    DEGRADATIONS = cfg['degradations'] == 1
    POISSON = cfg['poisson'] == 1
    RANDOM_MASK = cfg['random_mask'] == 1
    FREEZE = cfg['freeze']
    LR_SCHEDULER = cfg['lr_scheduler']
    ADAM = cfg['adam'] == 1
    BACKBONE = cfg['backbone']
    CROP_MODE_FF = cfg["crop_mode_ff"]
    WEIGHTED_SAMPLER = cfg["weighted_sampler"] == 1
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')


    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    # train_dataset=SBI_Dataset(phase='train',image_size=image_size, degradations = DEGRADATIONS, poisson = POISSON, random_mask = RANDOM_MASK)
    # val_dataset=SBI_Dataset(phase='val',image_size=image_size, degradations = DEGRADATIONS, poisson = POISSON, random_mask = RANDOM_MASK)
    # train_dataset = SBI_Custom_Dataset('train', ['FF', 'MSU-MFSD', 'REPLAY-ATTACK', 'MOBIO', 'SIM-MV2'], 
    #                                    image_size = image_size, degradations = DEGRADATIONS, poisson = POISSON, random_mask = RANDOM_MASK)
    # val_dataset = SBI_Custom_Dataset('val', ['FF', 'MSU-MFSD', 'REPLAY-ATTACK', 'MOBIO', 'SIM-MV2'], 
    #                                  image_size = image_size, degradations = DEGRADATIONS, poisson = POISSON, random_mask = RANDOM_MASK)
    
    if WEIGHTED_SAMPLER:
        print("Using weighted sampler")
        dataset_list = []
        source_counts = {}
        for dataset_name in cfg["train_datasets"]:
            if (dataset_name == "FF"):
                dataset = SBI_Custom_Dataset('train', [dataset_name], image_size=image_size, degradations=DEGRADATIONS, poisson=POISSON, random_mask=RANDOM_MASK, crop_mode=CROP_MODE_FF)
            else:
                dataset = SBI_Custom_Dataset('train', [dataset_name], image_size=image_size, degradations=DEGRADATIONS, poisson=POISSON, random_mask=RANDOM_MASK, crop_mode='yunet')
            dataset_list.append(dataset)
            source_counts[dataset_name] = len(dataset)
        combined_dataset = ConcatDataset(dataset_list)
        source_ids = []
        for i in range(len(cfg["train_datasets"])):
            source_ids.extend([i] * source_counts[cfg["train_datasets"][i]])
        source_weights = [1.0 / count for count in source_counts.values()]
        sample_weights = [source_weights[id] for id in source_ids]

        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights))

        train_loader = DataLoader(combined_dataset, batch_size=batch_size//2, 
                                  collate_fn = dataset_list[0].collate_fn, 
                                  num_workers = 4,
                                  pin_memory = True,
                                  drop_last = True,
                                  worker_init_fn = dataset_list[0].worker_init_fn,
                                  sampler=sampler)

    else:
        train_dataset = SBI_Custom_Dataset('train', cfg["train_datasets"], 
                                       image_size = image_size, degradations = DEGRADATIONS, poisson = POISSON, random_mask = RANDOM_MASK, crop_mode = CROP_MODE_FF)
        train_loader = DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )
    val_dataset = SBI_Custom_Dataset('val', cfg["val_datasets"], 
                                     image_size = image_size, degradations = DEGRADATIONS, poisson = POISSON, random_mask = RANDOM_MASK, crop_mode = CROP_MODE_FF)
    val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
    
    model=Detector(lr = cfg['lr'], adam = ADAM, backbone = BACKBONE)
    if len(cfg["weight_path"]):
        cnn_sd = torch.load(cfg["weight_path"])["model"]
        model.load_state_dict(cnn_sd)
    if FREEZE > 0:
        model.freeze()
    model=model.to('cuda')
    
    

    #iter_loss=[]
    train_losses=[]
    #test_losses=[]
    train_accs=[]
    #test_accs=[]
    val_accs=[]
    val_losses=[]
    n_epoch=cfg['epoch']
    if (LR_SCHEDULER.upper() == 'LAA-NET'):
        lr_scheduler = LinearDecayLR_LaaNet(model.optimizer, n_epoch, n_epoch//4, last_epoch= -1, booster=4)
    elif (LR_SCHEDULER.upper() == 'SBI'):
        lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    elif (LR_SCHEDULER.upper() == 'COSINE'):
        lr_scheduler = FlatCosineAnnealingLR(model.optimizer, n_epoch, FREEZE)
    elif (LR_SCHEDULER.upper() == 'PMM'):
        lr_scheduler = ReduceLROnPlateau(model.optimizer, mode='min', factor=0.2, patience=10)
    else:
        print('Unknown LR Scheduler, defaulting to SBI base scheduler')
        lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    #last_loss=99999


    now=datetime.now()
    save_path='output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    if (not os.path.exists('output')):
        os.mkdir('output')
    os.mkdir(save_path)
    os.mkdir(save_path+'weights/')
    os.mkdir(save_path+'logs/')
    logger = log(path=save_path+"logs/", file="losses.logs")

    criterion=nn.CrossEntropyLoss()


    #last_auc=0
    last_val_auc=0
    weight_dict={}
    val_set = set()
    n_weight=5

    if USE_WANDB:
        logger.info('Initializing weights and biases...')
        wandb.init(
            project="SBI",
            config=cfg,
            resume=False
        )

    #final_transforms = get_final_transforms()
    needs_unfreezing = FREEZE > 0
    for epoch in range(n_epoch):
        if needs_unfreezing and epoch >= FREEZE:
            model.unfreeze()
            needs_unfreezing = False
            print("Changing post freeze learning rate")
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = 0.0001
        np.random.seed(seed + epoch)
        random.seed(seed + epoch) 
        train_loss = 0.
        train_acc = 0.
        train_probs = []
        train_targets = []
        model.train(mode=True)

        for step, data in enumerate(tqdm(train_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()

            output = model.training_step(img, target)
            loss = criterion(output, target)
            loss_value = loss.item()
            #iter_loss.append(loss_value)
            train_loss += loss_value

            probs = F.softmax(output, dim=1)
            acc = compute_accuray(torch.log(probs), target)
            train_acc += acc

            train_probs.append(probs[:, 1].detach().cpu())  # Assuming binary classification
            train_targets.append(target.detach().cpu())

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))

        train_probs = torch.cat(train_probs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_auc = roc_auc_score(train_targets, train_probs)

        log_text = "Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, train auc: {:.4f}, ".format(
            epoch + 1,
            n_epoch,
            train_loss / len(train_loader),
            train_acc / len(train_loader),
            train_auc
        )

        # Validation
        model.train(mode=False)
        val_loss = 0.
        val_acc = 0.
        output_dict = []
        target_dict = []
        np.random.seed(seed)
        for step, data in enumerate(tqdm(val_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()

            with torch.no_grad():
                output = model(img)
                loss = criterion(output, target)

            loss_value = loss.item()
            #iter_loss.append(loss_value)
            val_loss += loss_value
            acc = compute_accuray(F.log_softmax(output, dim=1), target)
            val_acc += acc
            output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
            target_dict += target.cpu().data.numpy().tolist()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))
        val_auc = roc_auc_score(target_dict, output_dict)

        log_text += "val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
            val_loss / len(val_loader),
            val_acc / len(val_loader),
            val_auc
        )
        val_loss = val_loss/len(val_loader)
        if USE_WANDB:
            wandb.log({
                'Val/Loss': val_loss,
                'Val/Accuracy': val_acc / len(val_loader),
                'Val/AUC': val_auc,
                'Train/Loss': train_loss / len(train_loader),
                'Train/Accuracy': train_acc / len(train_loader),
                'Train/AUC': train_auc,
                'Train/LearningRate': model.optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })

        if (LR_SCHEDULER.upper() == 'PMM'):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()


        if len(weight_dict) < n_weight or epoch == n_epoch - 1:
            save_model_path = os.path.join(save_path + 'weights/', "{}_{:.6f}_val.tar".format(epoch + 1, val_auc))
            weight_dict[save_model_path] = val_auc
            torch.save({
                "model": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "epoch": epoch,
                'backbone': model.backbone,
            }, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])
        elif val_auc >= last_val_auc:
            save_model_path = os.path.join(save_path + 'weights/', "{}_{:.6f}_val.tar".format(epoch + 1, val_auc))
            for k in weight_dict:
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path] = val_auc
                    break
            torch.save({
                "model": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "epoch": epoch,
                'backbone': model.backbone,
            }, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        if (not epoch % cfg["test_every"] or epoch == n_epoch - 1):
            best_model = max(weight_dict, key=weight_dict.get)
            if (not best_model in val_set):
                val_set.add(best_model)
                for dataset in cfg['test_datasets']:
                    auc_test, acc_test, ap_test, ar_test, target_list, output_list = test(best_model, dataset, False, CROP_MODE)
                    fpr, tpr, _ = roc_curve(target_list, output_list)

                    # Create the ROC figure
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"ROC @ epoch {epoch}")
                    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve on {dataset} @ epoch {epoch}")
                    plt.legend(loc="lower right")

                    # Save it to wandb
                    roc_image = wandb.Image(plt)
                    plt.close()
                    if (USE_WANDB):
                        wandb.log({
                            f"Test/AUC_{dataset}": auc_test,
                            f"Test/Accuracy_{dataset}": acc_test,
                            f"Test/AP_{dataset}": ap_test,
                            f"Test/AR_{dataset}": ar_test,
                            f"Test/ROC_{dataset}": wandb.plot.roc_curve(target_list, [[1 - p, p] for p in output_list], labels=["pristine", "fake"]),
                            f"Test/ROC_Image_{dataset}": roc_image,
                            f"Test/Model": int(os.path.basename(best_model).split('_')[0]),
                            f"Test/test_step": epoch / cfg["test_every"]
                        })
        logger.info(log_text)

        
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n',dest='session_name')
    parser.add_argument('-w', dest = 'pretrained_weights', default = '')
    args=parser.parse_args()
    main(args)
        
