# Yuqi Peng and Ruining Yang

# CS 5330 Computer Vision
# Spring 2024
# Project 5

# test the Basic net on test set and new input
# Task 1 e, f


import torch
import BasicNet
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import os
import math



# plot the first n digits with the prediction for each example
def visualize_first_n(data, label, n=9):
  # draw the subplot
  for i in range(n):
    plt.subplot(math.ceil(n/3),3,i+1)
    plt.tight_layout()
    plt.imshow(data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(label[i, 0].item()))
    plt.xticks([])
    plt.yticks([])
  plt.show()


# class used for transform image to tensor data
class MyDataset(Dataset):
  def __init__(self, directory, transform=None):
    self.directory = directory
    self.transform = transform
    self.images = []
    self.labels = []

    # store the data
    for filename in os.listdir(directory):
      if filename.endswith('.jpg') and filename.startswith("image_"):
        label = int(filename.split('_')[1].split('.')[0])  # get the label
        self.images.append(os.path.join(directory, filename))
        self.labels.append(label)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = Image.open(self.images[idx]).convert('L') 
    label = self.labels[idx]
    if self.transform:
      image = self.transform(image)
    return image, label
    

# main function
def main():
  # get the trained model
  parameters_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/BasicNet_parameters.pth'
  net = BasicNet.BasicNet()
  net.load_state_dict(torch.load(parameters_path))

  # load the data
  train_path = "/Users/yuqipeng/Desktop/CS 5330/project/project5/MNIST_data/train_data.pt"
  test_path = "/Users/yuqipeng/Desktop/CS 5330/project/project5/MNIST_data/test_data.pt"
  train_loader, test_loader = BasicNet.load_minist_data(train_path, test_path)

  # load the new data
  my_dataset = MyDataset(directory='/Users/yuqipeng/Desktop/CS 5330/project/project5/myNumbers', transform=ToTensor())
  myData_loader = DataLoader(my_dataset, batch_size=10, shuffle=False)


  # test the first ten examples in test set
  net.eval()  # change to evaluation mode
  with torch.no_grad():
    # Task 1.e
    # get data
    data, target = next(iter(test_loader))
    data, target = data[:10], target[:10]  # get the first 10 data
    
    # predict
    output = net(data)
    pred = output.argmax(dim=1, keepdim=True)

    # plot 9 digits as a 3x3 grid with the prediction for each example
    visualize_first_n(data, pred)

    # print the results
    print("----Result for the first 10 examples in the test set----")
    print("Output:")
    print(torch.round(output, decimals=2))
    print("Prediction:")
    print(pred.reshape(1, -1))
    print("Label:")
    print(target.reshape(1, -1))


    # Task 1.f
    # get data
    images, labels = next(iter(myData_loader))

    # predict
    my_output = net(images)
    my_pred = my_output.argmax(dim=1, keepdim=True)

    # plot 9 digits as a 3x3 grid with the prediction for each example
    visualize_first_n(images, my_pred, 10)

    # print the results
    print("----Result for the my own handwriting examples----")
    print("Output:")
    print(torch.round(my_output, decimals=2))
    print("Prediction:")
    print(my_pred.reshape(1, -1))
    print("Label:")
    print(labels.reshape(1, -1))
      


if __name__ == "__main__":
    main()
