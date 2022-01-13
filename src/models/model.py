import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim

# class MyAwesomeModel(nn.Module):
#     def __init__(self):
#         super(MyAwesomeModel, self).__init__()
# self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
# self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
# self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
# self.fc1 = nn.Linear(3 * 3 * 64, 256)
# self.fc2 = nn.Linear(256, 10)

# def forward(self, x):
#     x = F.relu(self.conv1(x))
#     # x = F.dropout(x, p=0.5, training=self.training)
#     x = F.relu(F.max_pool2d(self.conv2(x), 2))
#     x = F.dropout(x, p=0.5, training=self.training)
#     x = F.relu(F.max_pool2d(self.conv3(x), 2))
#     x = F.dropout(x, p=0.5, training=self.training)
#     x = x.view(-1, 3 * 3 * 64)
#     x = F.relu(self.fc1(x))
#     x = F.dropout(x, training=self.training)
#     x = self.fc2(x)
#     return F.log_softmax(x, dim=1)


class MyAwesomeModel(LightningModule):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

        self.criterium = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        x = x.view(len(x), 1, 28, 28)
        x = x.type(torch.FloatTensor)
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target.long())

        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        # loss = self.criterium(preds, target.long())
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("test_acc", acc)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(model)
