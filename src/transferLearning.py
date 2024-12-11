# Yuqi Peng and Ruining Yang

# CS 5330 Computer Vision
# Spring 2024
# Project 5

# Transfer Learning on Greek Letters
# Task 3



import torchvision.transforms as transforms
import torchvision
import BasicNet
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import testBasicNet
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor



# this function return the new model used for classify greek letters under transfer learning
def getTransferNet():
    # load the model
    parameters_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/BasicNet_parameters.pth'
    net = BasicNet.BasicNet()
    net.load_state_dict(torch.load(parameters_path))

    # freezes the parameters for the whole network
    for param in net.parameters():
        param.requires_grad = False
    
    # relplace the last layer
    num_ftrs = net.fc2.in_features
    net.fc2 = nn.Linear(num_ftrs, 3)

    return net




# greek data set transform\
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale( x )
        x = transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = transforms.functional.center_crop( x, (28, 28) )
        return transforms.functional.invert( x )
    

def train(net, train_loader, train_losses, train_counter, epoch, lr=0.01, momentum=0.5):
  # initialize
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    total_loss = 0

    # training process
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{10}", unit="batch")): 
        optimizer.zero_grad()  # clean the previous gradient
        output = net.forward(data)  # get the output of net
        loss = F.nll_loss(output, target)  # compute the loss
        loss.backward()  # compute the gradient
        optimizer.step()  # update parameters
        total_loss += loss.item()
        if batch_idx % 2 == 0:
            train_losses.append(loss.item())  # record the train loss
            train_counter.append((batch_idx*len(data)) + ((epoch-1)*len(train_loader.dataset)))  # record the number of data used for training 
    
    # evaluate the model and print the evaluation after training
    avg_loss = total_loss / len(train_loader)
    train_accuracy = BasicNet.evaluate(net, train_loader)
    print(f'Epoch {epoch}/10: Avg. Loss: {avg_loss:.4f}, '
            f'Train Accuracy: {train_accuracy*100:.2f}%.')
    
    # save the model
    patameters_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/TransformLearning_parameter.pth'
    torch.save(net.state_dict(), patameters_path)

def error_plot(error_data):
    train_losses = error_data['train_losses']
    train_counter = error_data['train_counter']

    plt.figure(figsize=(10,5))
    plt.plot(train_counter, train_losses, color='blue', zorder=1)
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.title('Training Loss over time')
    plt.show()

    

def main():
    # get the transferNet
    net = getTransferNet()

    # DataLoader for the Greek data set
    training_set_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/greek_train'
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( training_set_path,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )
    

    # train
    train_losses = []
    train_counter = []
    epoches = 10
    for epoch in range(1, epoches+1):
        train(net, greek_train, train_losses, train_counter, epoch)
    training_data_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/TransformLearning_training_data.pth'
    training_data = {'train_losses': train_losses,
                      'train_counter': train_counter}
    torch.save(training_data, training_data_path)

    # plot the error
    error_plot(training_data)


    # get data
    test_set_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/myGreekLetters'
    myData_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( test_set_path,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 15,
        shuffle = False )
    images, labels = next(iter(myData_loader))

    # predict
    my_output = net(images)
    my_pred = my_output.argmax(dim=1, keepdim=True)

    # show the result
    print("----Result for the my own handwriting examples----")
    print("Output:")
    print(torch.round(my_output, decimals=2))
    print("Prediction:")
    print(my_pred.reshape(1, -1))
    print("Label:")
    print(labels.reshape(1, -1))




if __name__ == "__main__":
    main()