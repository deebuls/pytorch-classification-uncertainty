import torch
import scipy.ndimage as nd
import numpy as np
import cv2

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def rotate_img(x, deg):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()

def reduce_img(x, size, pos):
    original_size = 28
    if (pos[0] + size)>original_size:
      pos[0] = pos[0] - (pos[0]+size-original_size)
    if (pos[1] + size)>original_size:
      pos[1] = pos[1] - (pos[1]+size-original_size)
    canvas = np.zeros((original_size, original_size), dtype=np.float32)
    small_img = cv2.resize(x.numpy().reshape(original_size,original_size, 1), (size,size))
    canvas[ pos[1]:pos[1]+small_img.shape[0], pos[0]:pos[0]+small_img.shape[1]] = small_img
    return canvas
