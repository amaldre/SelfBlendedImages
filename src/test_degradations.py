import numpy as np
import torch

from tqdm import tqdm
from torchvision.utils import save_image
import os
from utils.sbi import SBI_Dataset
from degradations import degradation

save_dir = "debug_degraded"
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda')
val_dataset = SBI_Dataset(phase = 'val', image_size = 380, degradations = True, poisson = True, random_mask = True)
val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=True,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
for step, data in enumerate(tqdm(val_loader)):
    img = data['img'].to(device, non_blocking=True).float()  # [2, 3, H, W]
    img = (img + 1) / 2  # Normalisation de [-1, 1] à [0, 1]

    assert img.shape[0] == 2, f"Expected 2 images (real + fake), got {img.shape[0]}"

    img_r = img[0]  # [3, H, W]
    img_f = img[1]  # [3, H, W]

    save_image(img_r, os.path.join(save_dir, f"step{step}_real.png"))
    save_image(img_f, os.path.join(save_dir, f"step{step}_fake.png"))

        # Empiler pour obtenir un batch final
        #degraded_img_batch = torch.stack(degraded_list, dim=0) 