# Yuqi Peng and Ruining Yang

# CS 5330 Computer Vision
# Spring 2024
# Project 5

# Experiment
# Task 4 & extension 1: evaluate more dimensions




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np



# class Advancednet, which is used for experiment and 
class AdvancedNet(nn.Module):
  def __init__(self, conv1_out_channels, conv2_out_channels, kernel_size, dropout_rate, fc1_out_features):
    super(AdvancedNet, self).__init__()
    self.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=kernel_size)
    self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=kernel_size)
    self.conv2_drop = nn.Dropout2d(dropout_rate)
    conv1_output_size = (28 - kernel_size + 1) // 2
    conv2_output_size = (conv1_output_size - kernel_size + 1) // 2
    fc_input_features = conv2_output_size * conv2_output_size * conv2_out_channels
    self.fc1 = nn.Linear(fc_input_features, fc1_out_features)
    self.fc2 = nn.Linear(fc1_out_features, 10)

  def forward(self, x):
    x = self.conv1(x)  # layer 1: convolution 
    x = F.relu(F.max_pool2d(x, 2))  # layer 2: max pooling with ReLU function
    x = self.conv2(x)  # layer 3: convolution
    x = self.conv2_drop(x)  # layer 4: dropout layer
    x = F.relu(F.max_pool2d(x, 2))  # layer 5: max pooling with ReLU function
    x = x.view(x.size(0), -1)  # flattening operation
    x = F.relu(self.fc1(x))  # layer 6: fully connected linear layer with ReLU function
    x = self.fc2(x)  # layer 7: fully connected linear layer
    return F.log_softmax(x, dim=1)  # return the output
  

# function to load data
def load_data(train_batch_size):
    train_loader = DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), batch_size=1000, shuffle=True)
    return train_loader, test_loader


# this function used to train the model
def train_model(model, train_loader, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    model.train()  # train mode
    for epoch in range(epochs):
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    file_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/AdvancedNet_parameters.pth'
    torch.save(model.state_dict(), file_path)
    


# this function used to evaluate the model
def eval_model(model, test_loader):
    model.eval()  # evaluation mode
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy


# this function used for experiment
def find_best_param(param, options, best_param):
    best_accuracy = 0
    if param == 'batch_size':
        for i in range(len(options[param])):
            train_loader, test_loader = load_data(options['batch_size'][i])
            model = AdvancedNet(options['conv1'][best_param[1]], 
                                options['conv2'][best_param[2]], 
                                options['kernel_size'][best_param[3]], 
                                options['dropout_rate'][best_param[4]], 
                                options['fc1'][best_param[5]])
            train_model(model, train_loader, options['epoch'][best_param[6]])
            accuracy = eval_model(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param[0] = i

    elif param == 'conv1':
        train_loader, test_loader = load_data(options['batch_size'][best_param[0]])
        for i in range(len(options[param])):
            model = AdvancedNet(options['conv1'][i], 
                                options['conv2'][best_param[2]], 
                                options['kernel_size'][best_param[3]], 
                                options['dropout_rate'][best_param[4]], 
                                options['fc1'][best_param[5]])
            train_model(model, train_loader, options['epoch'][best_param[6]])
            accuracy = eval_model(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param[1] = i

    elif param == 'conv2':
        train_loader, test_loader = load_data(options['batch_size'][best_param[0]])
        for i in range(len(options[param])):
            model = AdvancedNet(options['conv1'][best_param[1]], 
                                options['conv2'][i], 
                                options['kernel_size'][best_param[3]], 
                                options['dropout_rate'][best_param[4]], 
                                options['fc1'][best_param[5]])
            train_model(model, train_loader, options['epoch'][best_param[6]])
            accuracy = eval_model(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param[2] = i

    elif param == 'kernel_size':
        train_loader, test_loader = load_data(options['batch_size'][best_param[0]])
        for i in range(len(options[param])):
            model = AdvancedNet(options['conv1'][best_param[1]], 
                                options['conv2'][best_param[2]], 
                                options['kernel_size'][i], 
                                options['dropout_rate'][best_param[4]], 
                                options['fc1'][best_param[5]])
            train_model(model, train_loader, options['epoch'][best_param[6]])
            accuracy = eval_model(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param[3] = i

    elif param == 'dropout_rate':
        train_loader, test_loader = load_data(options['batch_size'][best_param[0]])
        for i in range(len(options[param])):
            model = AdvancedNet(options['conv1'][best_param[1]], 
                                options['conv2'][best_param[2]], 
                                options['kernel_size'][best_param[3]], 
                                options['dropout_rate'][i], 
                                options['fc1'][best_param[5]])
            train_model(model, train_loader, options['epoch'][best_param[6]])
            accuracy = eval_model(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param[4] = i


    elif param == 'fc1':
        train_loader, test_loader = load_data(options['batch_size'][best_param[0]])
        for i in range(len(options[param])):
            model = AdvancedNet(options['conv1'][best_param[1]], 
                                options['conv2'][best_param[2]], 
                                options['kernel_size'][best_param[3]], 
                                options['dropout_rate'][best_param[4]], 
                                options['fc1'][i])
            train_model(model, train_loader, options['epoch'][best_param[6]])
            accuracy = eval_model(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param[5] = i

    
    elif param == 'epoch':
        train_loader, test_loader = load_data(options['batch_size'][best_param[0]])
        for i in range(len(options[param])):
            model = AdvancedNet(options['conv1'][best_param[1]], 
                                options['conv2'][best_param[2]], 
                                options['kernel_size'][best_param[3]], 
                                options['dropout_rate'][best_param[4]], 
                                options['fc1'][best_param[5]])
            train_model(model, train_loader, options['epoch'][i])
            accuracy = eval_model(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param[6] = i

    return best_param, best_accuracy
    





                            


# main function
def main():
    # define the range for hyperparameters
    train_batch_size_options = [16, 32, 64, 128, 256]  # 5 options
    conv1_out_channels_options = [4, 8, 16, 32, 64]  # 5 options
    conv2_out_channels_options = [8, 16, 32, 64, 128]  # 5 options
    kernel_size_options = [3, 5, 7]  # 3 options
    dropout_rate_options = [round(rate, 2) for rate in np.arange(0.20, 0.72, 0.02)] # 26 options
    fc1_out_features_options = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 
                                150, 160, 170, 180, 190, 200]  # 16 options
    epoch_options = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 11 options
    best_param = [0, 0, 0, 0, 0, 0, 0]  # index of best parameters
    param = ['batch_size', 'conv1', 'conv2', 'kernel_size', 'dropout_rate', 'fc1', 'epoch']
    param_options = {'batch_size': train_batch_size_options,
                     'conv1': conv1_out_channels_options,
                     'conv2': conv2_out_channels_options,
                     'kernel_size': kernel_size_options,
                     'dropout_rate': dropout_rate_options,
                     'fc1': fc1_out_features_options,
                     'epoch': epoch_options}


    for i in range(len(best_param)):
        best_param, best_accuracy = find_best_param(param[i], param_options, best_param)


    
    
    print(best_accuracy)
    print(best_param)





if __name__ == "__main__":
    main()




