# Yuqi Peng and Ruining Yang

# CS 5330 Computer Vision
# Spring 2024
# Project 5

# Examine the BasicNet by visualize the filters and their effect of images
# Task 2 a, b


import torch
import BasicNet
import matplotlib.pyplot as plt
import cv2


# function that plot the filters
def plot_the_filters(filters):
    plt.figure()  # figsize=(12, 9)
    for i in range(1, 11):
        plt.subplot(3, 4, i)
        filter_img = filters[i-1, 0]
        plt.imshow(filter_img, cmap='gray')
        plt.title(f'Filter {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# function that plot the effect of filters on a image
def plot_filters_effect(filters, image):
    plt.figure()
    for i in range(1, 11):
        # plot the filters
        plt.subplot(5, 4, 2 * i - 1)
        filter_img = filters[i-1, 0]
        plt.imshow(filter_img, cmap='gray')
        plt.axis('off')

        # plot the effects
        plt.subplot(5, 4, 2 * i)
        filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=filter_img)[0]
        plt.imshow(filtered_image)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()






def main():
    with torch.no_grad():
        # get the data and model
        train_loader, test_loader = BasicNet.getData('r')
        net = BasicNet.getModel('r', train_loader, test_loader)

        # get the first layer weight of model
        first_layer_weight = net.conv1.weight
        filters = first_layer_weight.numpy()

        # Task 2.a
        # print the filters weight shape and their weight
        for i in range(10):
            print(f"The {i+1} filter in the first layer is:")
            print(first_layer_weight[i, 0].shape)
            print(first_layer_weight[i, 0])

        # plot the ten filters
        plot_the_filters(filters)


        # Task 2.b
        # get the image and its label
        batch_idx, (images, labels) = next(enumerate(train_loader))
        image = images[0].numpy()

        # plot the effect figure
        plot_filters_effect(filters, image)
    

    


if __name__ == "__main__":
    main()