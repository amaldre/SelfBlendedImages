import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import random
from utils.sbi import SBI_Dataset, SBI_Custom_Dataset, SourceConcat
from tqdm import tqdm

from inference.datasets import *


def main():
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    DATASETS = ["FF", "MSU-MFSD", "REPLAY-ATTACK", "SIM-MV2", "MOBIO"]
    #DATASETS = ["MSU-MFSD"]
    image_size=380
    batch_size=16
    
    if True:
        print("Using weighted sampler")
        dataset_list = []
        source_counts = {}
        for dataset_name in DATASETS:
            dataset = SBI_Custom_Dataset('train', [dataset_name], image_size=image_size, crop_mode='yunet')
            dataset_list.append(dataset)
            source_counts[dataset_name] = len(dataset)
        combined_dataset = SourceConcat(dataset_list)
        source_ids = []
        for i in range(len(DATASETS)):
            source_ids.extend([i] * source_counts[DATASETS[i]])
        source_weights = [1.0 / count for count in source_counts.values()]
        sample_weights = [source_weights[id] for id in source_ids]

        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights))

        train_loader = DataLoader(combined_dataset, batch_size=batch_size//2, 
                                  collate_fn = combined_dataset.custom_collate_fn, 
                                  num_workers = 4,
                                  pin_memory = True,
                                  drop_last = True,
                                  worker_init_fn = dataset_list[0].worker_init_fn,
                                  sampler=sampler)
        distribution = {}

    for step, data in enumerate(tqdm(train_loader)):
        source_ids = data['source_id']  # tensor of shape (batch_size,)
    
        if isinstance(source_ids, torch.Tensor):
            source_ids = source_ids.tolist()
    
        for sid in source_ids:
            if sid in distribution:
                distribution[sid] += 1
            else:
                distribution[sid] = 1

    # Compute total
    total_samples = sum(distribution.values())

    # Print counts and percentages
    print("\nSample distribution by source:")
    for sid in sorted(distribution.keys()):
        name = DATASETS[sid]
        count = distribution[sid]
        percent = (count / total_samples) * 100
        print(f"{name} (id {sid}): {count} samples ({percent:.2f}%)")

    print(f"\nTotal samples seen: {total_samples}")





        
if __name__=='__main__':
    main()
        
