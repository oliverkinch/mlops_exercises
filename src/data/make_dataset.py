import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, labels, images, transform=None):

        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def extract_data(npz_dir):
    npz_file = np.load(npz_dir)
    images = npz_file["images"]
    images = torch.from_numpy(images)
    # images = images.view(images.shape[0], -1)

    labels = npz_file["labels"]
    labels = torch.from_numpy(labels)

    return images, labels


def image_process(images):
    images = images.view(images.shape[0], -1)
    images = F.normalize(images, p=2, dim=1)
    return images


def mnist():
    # exchange with the corrupted mnist dataset
    data_dir = "data/raw/corruptmnist/"
    files = os.listdir(data_dir)
    data_dir2 = "data/raw/corruptmnist_v2/"
    files2 = os.listdir(data_dir2)
    test_images, test_labels = extract_data(data_dir + files[0])

    all_train_images = torch.Tensor([])
    all_train_labels = torch.Tensor([])
    for train_file in files[1:]:
        train_images, train_labels = extract_data(data_dir + train_file)
        all_train_images = torch.cat((all_train_images, train_images), 0)
        all_train_labels = torch.cat((all_train_labels, train_labels), 0)

    for train_file in files2:
        train_images, train_labels = extract_data(data_dir2 + train_file)
        all_train_images = torch.cat((all_train_images, train_images), 0)
        all_train_labels = torch.cat((all_train_labels, train_labels), 0)

    all_train_images = image_process(all_train_images)
    test_images = image_process(test_images)

    print("Train images shape: ", all_train_images.shape)
    print("Test images shape: ", test_images.shape)
    print("Train labels shape: ", all_train_labels.shape)
    print("Test labels shape: ", test_labels.shape)

    save_dir = "data/processed/"
    torch.save(all_train_images, save_dir + "train_tensor.pth")
    torch.save(test_images, save_dir + "test_images.pth")
    torch.save(all_train_labels, save_dir + "train_labels.pth")
    torch.save(test_labels, save_dir + "test_labels.pth")


if __name__ == "__main__":
    mnist()
