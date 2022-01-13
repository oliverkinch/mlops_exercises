pass
# from torch.utils.data import DataLoader, Dataset
# import torch

# class MNISTDataset(Dataset):
#     def __init__(self, labels, images, transform=None):

#         self.labels = labels
#         self.images = images
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, label


# data_dir = 'data/processed/'

# # data_dir = os.getcwd().split('outputs/')[0] + hparams['data_dir']
# train_images = torch.load(data_dir + "train_tensor.pth")
# test_images = torch.load(data_dir + "test_images.pth")
# train_labels = torch.load(data_dir + "train_labels.pth")
# test_labels = torch.load(data_dir + "test_labels.pth")

# trainset = MNISTDataset(train_labels, train_images)
# batch_size = 16
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# testset = MNISTDataset(test_labels, test_images)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


# assert len(trainset) == 40000
# assert len(testset) == 5000

# im, _ = next(iter(trainloader))
# assert im.shape[1] == 784
