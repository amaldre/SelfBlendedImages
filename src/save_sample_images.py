import os
import cv2
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import random
from tqdm import tqdm

from utils.sbi import SBI_Custom_Dataset, SourceConcat
from inference.datasets import *

def main():
    # Fix seed
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Settings
    DATASETS = ["FF", "MSU-MFSD", "REPLAY-ATTACK", "SIM-MV2", "MOBIO"]
    image_size = 380
    batch_size = 16
    save_dir = "saved_with_labels"
    max_to_save = 300

    os.makedirs(save_dir, exist_ok=True)

    # Prepare datasets and sampler
    dataset_list = []
    source_counts = {}
    for dataset_name in DATASETS:
        dataset = SBI_Custom_Dataset('val', [dataset_name], image_size=image_size, crop_mode='yunet', poisson = True, random_mask= True)
        dataset_list.append(dataset)
        source_counts[dataset_name] = len(dataset)

    combined_dataset = SourceConcat(dataset_list)

    source_ids = []
    for i, name in enumerate(DATASETS):
        source_ids.extend([i] * source_counts[name])
    source_weights = [1.0 / count for count in source_counts.values()]
    sample_weights = [source_weights[i] for i in source_ids]

    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights))

    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size // 2,
        collate_fn=combined_dataset.custom_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=dataset_list[0].worker_init_fn,
        sampler=sampler
    )

    # Drawing setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (255, 255, 255)

    print("💾 Saving image pairs with labels...")
    num_saved = 0

    with torch.no_grad():
        for data in tqdm(train_loader):
            images = data["img"]            # [B, C, H, W]
            labels = data["label"]
            source_ids = data["source_id"]

            b = images.size(0)
            assert b % 2 == 0, "Batch size must be even for pairing"
            n = b // 2

            for i in range(n):
                if num_saved >= max_to_save:
                    break

            # Get image pair: i and i + n
                img1 = (((images[i] + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img2 = (((images[i + n] + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Convert to BGR for OpenCV
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

            # Add text to both
                source1 = DATASETS[source_ids[i].item()]
                label1 = labels[i].item()
                source2 = DATASETS[source_ids[i + n].item()]
                label2 = labels[i + n].item()

                cv2.putText(img1, f"{source1} ({label1})", (10, 20), font, font_scale, color, font_thickness, cv2.LINE_AA)
                cv2.putText(img2, f"{source2} ({label2})", (10, 20), font, font_scale, color, font_thickness, cv2.LINE_AA)
                # or use np.concatenate((img1, img2), axis=1)

            # Save
                filename = f"{num_saved:03d}_pristine.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, img1)
                filename = f"{num_saved:03d}_fake.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, img2)
                num_saved += 2

        print(f"\n✅ {num_saved} image pairs saved in '{save_dir}'")

if __name__ == '__main__':
    main()
