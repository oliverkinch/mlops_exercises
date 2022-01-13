import argparse
import numpy as np
import torch
from model import MyAwesomeModel
from torch.utils.data import DataLoader, Dataset


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


def evaluate():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--modeldir", default="models/mnist_cnn.pth", type=str)
    argparser.add_argument(
        "-i", "--imagedir", default="data/processed/test_images.pth", type=str
    )
    argparser.add_argument(
        "-l", "--labeldir", default="data/processed/test_labels.pth", type=str
    )

    args = argparser.parse_args()
    state_dict = torch.load(args.modeldir)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    model.eval()

    test_images = torch.load(args.imagedir)
    test_labels = torch.load(args.labeldir)
    testset = MNISTDataset(test_labels, test_images)
    batch_size = 64
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    accuracies = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(len(images), 1, 28, 28)
            images = images.type(torch.FloatTensor)
            ps = torch.exp(model(images))
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(accuracy.item())

        print(f"Accuracy: {np.mean(accuracies)*100}%")


if __name__ == "__main__":
    evaluate()
