from pyexpat import features
import torchvision
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
from model import MyAwesomeModel
import torchdrift
import matplotlib.pyplot as plt
import pandas as pd


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

def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

data_dir = 'data/processed/'
print(data_dir + "train_tensor.pth")
train_images = torch.load(data_dir + "train_tensor.pth")
test_images = torch.load(data_dir + "test_images.pth")
train_labels = torch.load(data_dir + "train_labels.pth")
test_labels = torch.load(data_dir + "test_labels.pth")


trainset = MNISTDataset(train_labels, train_images)
batch_size = 1
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = MNISTDataset(test_labels, test_images)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


model = MyAwesomeModel().load_from_checkpoint('models/example2.ckpt')
model.only_features = True

inputs, _ = next(iter(trainloader))
inputs = inputs.view(-1, 28, 28)
inputs = torch.unsqueeze(inputs, 1)

inputs = inputs.type(torch.FloatTensor)
inputs_ood = corruption_function(inputs)

N = batch_size
model.eval()
inps = torch.cat([inputs[:N], inputs_ood[:N]])
model.cpu()


# plt.figure(figsize=(15, 5))
# for i in range(2 * N):
#     plt.subplot(2, N, i + 1)
#     plt.imshow(inps[i].permute(1, 2, 0))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


kernels = [
    torchdrift.detectors.mmd.GaussianKernel, 
    torchdrift.detectors.mmd.ExpKernel,
    torchdrift.detectors.mmd.RationalQuadraticKernel
    ]

table_data = {
    'score': [],
    'P-value': []
    }

for kernel in kernels:
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    torchdrift.utils.fit(trainloader, model, drift_detector)

    features = model(inputs)
    print('feature space dim: ', features.shape)
    score = drift_detector(features) # Program crashes here for tensor of size (1, 576)
    p_val = drift_detector.compute_p_value(features)
    score, p_val
    table_data['score'].append(score)
    table_data['P-value'].append(p_val)

columns = [
    'GaussianKernel',
    'ExpKernel',
    'ReationalQuadraticKernel'
]
df_table = pd.DataFrame(data=table_data, columns=columns)