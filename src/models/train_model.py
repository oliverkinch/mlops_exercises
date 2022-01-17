import os
import hydra
import torch
from model import MyAwesomeModel
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset


# import wandb



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


# wandb.init()


@hydra.main(config_path="config", config_name="training_config.yaml")
def train_eval(config):
    print(f"Training configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    data_dir = os.getcwd().split("outputs/")[0] + hparams["data_dir"]
    train_images = torch.load(data_dir + "train_tensor.pth")
    test_images = torch.load(data_dir + "test_images.pth")
    train_labels = torch.load(data_dir + "train_labels.pth")
    test_labels = torch.load(data_dir + "test_labels.pth")

    trainset = MNISTDataset(train_labels, train_images)
    batch_size = hparams["batch_size"]
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = MNISTDataset(test_labels, test_images)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = MyAwesomeModel(hparams["lr"])
    # wandb.watch(model, log_freq=hparams["log_freq"])
    checkpoint_callback = ModelCheckpoint(
        dirpath="", monitor="train_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=hparams["patience"], verbose=True, mode="min"
    )

    trainer = Trainer(
        limit_train_batches=hparams["limit_train_batches"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=hparams["n_epochs"],
    )
    # logger=pytorch_lightning.loggers.WandbLogger(project="dtu_mlops"),

    trainer.fit(model, trainloader)
    
    trainer.test(model, testloader)
    trainer.save_checkpoint("example2.ckpt")
    checkpoint_callback.best_model_path


if __name__ == "__main__":
    train_eval()

# print('#######'*5, os.getcwd())
# @hydra.main(config_path="config", config_name='default_config.yaml')
# def train(config):
#     hparams = config.experiment
#     data_dir = hparams['data_dir']
#     print(data_dir)
#     print('#######'*5, os.getcwd())
#     print('~/')
#     torch.manual_seed(hparams["seed"])
#     train_images = torch.load(data_dir + "train_tensor.pth")
#     # test_images = torch.load(data_dir + 'test_images.pth')
#     train_labels = torch.load(data_dir + "train_labels.pth")
#     # test_labels = torch.load(data_dir + 'test_labels.pth')

#     trainset = MNISTDataset(train_labels, train_images)
#     batch_size = hparams['batch_size']
#     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

#     # testset = MNISTDataset(test_labels, test_images)
#     # testloader = DataLoader(testset, batch_size=64, shuffle=True)

#     model = MyAwesomeModel()
#     criterion = nn.NLLLoss()
#     optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
#     epochs = hparams['n_epochs']
#     running_losses = []
#     for e in range(epochs):
#         print(f"Epoch {e+1}/{epochs}")
#         running_loss = 0
#         for i, (images, labels) in enumerate(trainloader):
#             images = images.view(len(images), 1, 28, 28)
#             images = images.type(torch.FloatTensor)
#             optimizer.zero_grad()
#             log_ps = model(images)
#             loss = criterion(log_ps, labels.long())
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#             if not i % 100:
#                 log.info(
#                     loss.item()
#                 )

#         running_losses.append(running_loss)

#     torch.save(model.state_dict(), f"{os.getcwd()}/mnist_cnn.pth")
#     # plt.figure()
#     # plt.title("Training loss")
#     # plt.plot(running_losses)
#     # plt.savefig("reports/figures/mnist_cnn_learning_curve")


# if __name__ == "__main__":
#     train()
