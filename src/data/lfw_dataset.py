"""
LFW dataloading
"""
import argparse
import time
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import pandas as pd

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    # if not isinstance(imgs, list):
    #     imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.folder_path = path_to_folder
        self.files = os.listdir(self.folder_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        file_name = self.files[index]
        file_path = self.folder_path + '/' + file_name
        img = Image.open(file_path)
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='data/raw/lfw/', type=str)
    parser.add_argument('-batch_size', default=2048, type=int)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', default=True, action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        imgs = next(iter(dataloader))
        show(imgs)
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        n_works = args.num_workers
        mu = np.mean(res)
        s = np.std(res)
        print(f'Timing: {mu}+-{s}')

        
        results_path = 'reports/time_results.txt'
        df = pd.read_csv(results_path)

        if n_works not in df.n.values:
            print('n workers not found in df')
            with open(results_path, 'a') as f:
                f.write(f'\n{n_works},{mu},{s}')
        else:
            print('n workers found in df')
