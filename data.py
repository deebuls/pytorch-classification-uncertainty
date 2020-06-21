import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST("./data/mnist",
                   download=True,
                   train=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor()]))

data_val = MNIST("./data/mnist",
                 train=False,
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize((28, 28)),
                     transforms.ToTensor()]))

dataloader_train = DataLoader(
    data_train, batch_size=1000, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)

dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}


digit_zero, _ = data_val[7]
digit_one, _ = data_val[8]
digit_two, _ = data_val[1]
digit_three, _ = data_val[2]
digit_four, _ = data_val[3]
digit_five, _ = data_val[4]
digit_six, _ = data_val[5]
data_seven, _ = data_va[0]
digit_eight, _ = data_val[6]
digit_nine, _ = data_val[7]


