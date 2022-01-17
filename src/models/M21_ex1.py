import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import ptflops

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




data_dir = 'data/processed/'
train_images = torch.load(data_dir + "train_tensor.pth")
test_images = torch.load(data_dir + 'test_images.pth')
train_labels = torch.load(data_dir + "train_labels.pth")
test_labels = torch.load(data_dir + 'test_labels.pth')

trainset = MNISTDataset(train_labels, train_images)
batch_size = 64
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = MNISTDataset(test_labels, test_images)
testloader = DataLoader(testset, batch_size=64, shuffle=True)



mobilenet = torchvision.models.mobilenet_v3_large(pretrained=True)

resnet = torchvision.models.resnet152(pretrained=True)




def time_model(model, loader):
    results = []
    with torch.no_grad():
        for _ in range(1):
            start = time.time()
            for images, _ in loader:
                images = images.view(len(images), 1, 28, 28)
                images = images.type(torch.FloatTensor)
                images = images.expand(-1, 3, -1, -1)
                _ = torch.exp(model(images))
            end = time.time()

            results.append(end - start)

    return np.mean(results)
            
# print("mobilenet mean time for one epoch", time_model(mobilenet, testloader))
# print("resnet mean time for one epoch", time_model(resnet, testloader))


def n_parameters(model):
    macs, params = ptflops.get_model_complexity_info(model, (3, 28, 28))
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return params


# print('resnet \n') 
# print(n_parameters(resnet))

# print('mobile \n') 
# print(n_parameters(mobilenet))
