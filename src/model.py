import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM

import os

class Detector(nn.Module):

    def __init__(self, backbone, lr = None, adam = None, phase = 'train'):
        super(Detector, self).__init__()
        self.net=EfficientNet.from_pretrained(backbone, advprop=True,num_classes=2)
        self.backbone = backbone
        if phase == 'train':
            self.cel=nn.CrossEntropyLoss()
            if adam:
                print("Using Adam optimizer")
                self.optimizer = SAM(self.parameters(), torch.optim.AdamW, lr=lr, #weight_decay=1e-4
                                     )
            else:
                print("Using SGD optimizer")
                self.optimizer=SAM(self.parameters(),torch.optim.SGD,lr=lr, momentum=0.9)
        
        
    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")  # Change path as needed
        # Optional: remove 'module.' prefix if the model was trained with DataParallel or DDP
        # from collections import OrderedDict

        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     new_key = k.replace("module.", "")  # Remove 'module.' if present
        #     new_state_dict[new_key] = v

        # Load weights
        if os.path.exists(weight_path):
            self.load_state_dict(state_dict, strict=True) # strict=True if you're sure all keys match
            print(f"Loaded {weight_path} for training")
        else:
            print(f"Invalid path {weight_path}, training from scratch")

    def freeze(self):
        # Freeze all layers
        for param in self.net.parameters():
            param.requires_grad = False

        # Unfreeze only the final classification layer
        for param in self.net._fc.parameters():
            param.requires_grad = True
        print("------------- Model frozen -------------")
        
    def unfreeze(self):
        for param in self.net.parameters():
            param.requires_grad = True
        print("------------- Model unfrozen -------------")

    def forward(self,x):
        x=self.net(x)
        return x
    
    
    def training_step(self,x,target):
        for i in range(2):
            pred_cls=self(x)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first
    