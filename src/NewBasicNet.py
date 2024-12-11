# Yuqi Peng and Ruining Yang

# CS 5330 Computer Vision
# Spring 2024
# Project 5

# NewBasicNet
# Extension 3: replace the first layer of BasicNet



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import math
import BasicNet


# this function used to make a gobor kernel
def gabor_kernel(lamda, theta, sigma=1, gamma=1, phi=0, kernel_size=5):
    sigma_x = sigma
    sigma_y = sigma / gamma
    
    # generate x and y
    xmax, ymax = kernel_size // 2, kernel_size // 2
    xmin, ymin = -xmax, -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))
    
    # rotate the coordinate
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    gb = np.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * np.pi / lamda * x_theta + phi)
    return torch.tensor(gb, dtype=torch.float32)


# this function used for make two sobel_kernels
def sobel_kernel():
    sobel_x = torch.tensor([[0., 0., 0., 0., 0.],
                            [0., -1., 0., 1., 0.], 
                            [0., -2., 0., 2., 0.], 
                            [0., -1., 0., 1., 0.],
                            [0., 0., 0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[0., 0., 0., 0., 0.],
                            [0., -1., -2., -1., 0.], 
                            [0., 0.,  0.,  0., 0.], 
                            [0., 1.,  2.,  1., 0.],
                            [0., 0., 0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
    return sobel_x, sobel_y


# this function used for create a filter bank with gabor kernels and sobel kernels
def create_filter_bank():
    # define gabor kernels
    lamdas = [10, 20]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_filters = [gabor_kernel(lamda, theta, kernel_size=5) for lamda in lamdas for theta in thetas]
    
    # define sobel kernels
    sobel_filters = list(sobel_kernel())
    
    # combine filters together
    gabor_filters = [f.unsqueeze(0).unsqueeze(0) for f in gabor_filters]
    filter_bank = gabor_filters + sobel_filters
    
    while len(filter_bank) < 10:
        filter_bank += filter_bank[:10 - len(filter_bank)]
    
    filter_bank_tensor = torch.cat(filter_bank, dim=0)
    
    return filter_bank_tensor


# main function
def main():
    # define the model
    model = BasicNet.BasicNet()  # load the BasicNet
    filter_bank = create_filter_bank()  # create the filter bank
    print("Filter bank shape:", filter_bank.shape)
    model.conv1.weight.data = filter_bank  # change the first layer

    # load the data
    train_loader, test_loader = BasicNet.getData('r')

    # define optimizer
    optimizer = optim.SGD([param for name, param in model.named_parameters() if name != 'conv1.weight'], lr=0.01, momentum=0.9)

    # training process
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{5}", unit="batch")):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        print(f'test accuracy: {BasicNet.evaluate(model, test_loader)}')

    





if __name__ == "__main__":
    main()