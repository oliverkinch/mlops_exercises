import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.models.model import MyAwesomeModel


def scale_to_01_range(x):
    # compute the distribution range
    value_range = np.max(x) - np.min(x)

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


if __name__ == "__main__":

    state_dict = torch.load("src/models/mnist_cnn.pth")
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    model.eval()
    # get all the model children as list
    model_children = list(model.children())

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # visualize the last conv layer filters
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[2]):
        plt.subplot(8, 8, i + 1)
        # (8, 8) because in conv0 we have 7x7 filters and total of 64
        # (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap="gray")
        plt.axis("off")
        plt.savefig("reports/figures/filter.png")
    plt.show()

    # Visualize features in 2D with TSNE. Not done - see e.g.
    #  https://learnopencv.com/t-sne-for-feature-visualization/

    # feature_layer = torch.flatten(model_children[-1].weight, 1)
    # tsne = TSNE(n_components=2).fit_transform(feature_layer.detach().numpy())
    # tx = tsne[:, 0]
    # ty = tsne[:, 1]
    # tx = scale_to_01_range(tx)
    # ty = scale_to_01_range(ty)

    # data_dir = 'data/processed/'

    # labels = torch.load(data_dir + 'train_labels.pth').long()[:500]
    # print('label shape: ', labels.shape)

    # # initialize a matplotlib plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # colors_per_class = list(range(10))
    # # for every class, we'll add a scatter plot separately
    # for label in colors_per_class:
    #     # find the samples of the current class in the data
    #     indices = [i for i, l in enumerate(labels) if l == label]

    #     # extract the coordinates of the points of this class only
    #     current_tx = np.take(tx, indices)
    #     current_ty = np.take(ty, indices)

    #     # convert the class color to matplotlib format
    #     color = np.array(colors_per_class[label], dtype=np.float) / 255

    #     # add a scatter plot with the corresponding color and label
    #     ax.scatter(current_tx, current_ty, c=color, label=label)

    # # build a legend using the labels we set previously
    # ax.legend(loc='best')

    # # finally, show the plot
    # plt.show()
