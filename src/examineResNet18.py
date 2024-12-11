# Yuqi Peng and Ruining Yang

# CS 5330 Computer Vision
# Spring 2024
# Project 5

# extension 2: load the pre-trained model network in PyTorch package and evaluate its 
# first couple of convolutional layers


import torch
import BasicNet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights



# this function used to visualize the convolution layers kernels
def visualize_kernels(layer, title):
    weights = layer.weight.data.cpu().numpy()  # get the weights of this layer
    num_kernels = weights.shape[0]  # get the number of kernels
    num_cols = 8  # get num_cols and num_rows
    if num_kernels % num_cols != 0:
        num_rows = num_kernels // num_cols + 1 
    else:
        num_rows = num_kernels // num_cols
    
    # plot the kernels
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    fig.suptitle(title, fontsize=16)  # add the suptitle for plot
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            kernel = weights[i, 0]
            ax.imshow(kernel)
            ax.axis('off')
            ax.set_title(f"Kernel {i+1}", fontsize=5)  # add the sub-title for subplot
        else:
            ax.axis('off')
    plt.show()


# this function used to visualize the effect of convolution layers kernels
def visualize_kernels_effect(layer, images, title):
    with torch.no_grad():
        weights = layer.weight.data.cpu().numpy()  # get the weights of this layer
        num_kernels = weights.shape[0]  # get the number of kernels
        num_cols = 8   # get num_cols and num_rows
        if num_kernels % num_cols != 0: 
            num_rows = num_kernels // num_cols + 1 
        else:
            num_rows = num_kernels // num_cols
        conv1_output = layer(images)

        # plot the effects of kernels
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows)) 
        fig.suptitle(title, fontsize=16)  # add the suptitle for plot
        for i, ax in enumerate(axes.flat):
            if i < conv1_output.shape[1]:
                ax.imshow(conv1_output[0, i].cpu().numpy())
                ax.axis('off')
                ax.set_title(f"Feature Map {i+1}", fontsize=5)  # add the sub-title for subplot
            else:
                ax.axis('off')
        plt.show()
    



def main():
    # load the ResNet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # load the first image in MNIST dataset
    transform = transforms.Compose([
    transforms.Resize(224),  # change the size
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)
    images, labels = next(iter(mnist_loader))

    # visualize the first convolution layer
    visualize_kernels(model.conv1, 'First Convolution layer Kernels')
    visualize_kernels_effect(model.conv1, images, 'Effect of First Convolution layer Kernels')

    # visulize the second convolution layer
    processed_images = model.maxpool(model.relu(model.bn1(model.conv1(images))))  # process the image
    visualize_kernels(model.layer1[0].conv1, 'Second Convolution layer Kernels')
    visualize_kernels_effect(model.layer1[0].conv1, processed_images, 'Effect of Second Convolution layer Kernels')






if __name__ == "__main__":
    main()

