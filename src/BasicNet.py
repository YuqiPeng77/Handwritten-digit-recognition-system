# Yuqi Peng and Ruining Yang

# CS 5330 Computer Vision
# Spring 2024
# Project 5

# BasicNet
# Task 1: a, b, c, d



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm


# function that used for down the dataset
def download_mnist_datasets(train_data_path, test_data_path):
  """
  This function download the mnist datasets.
  train_data_path: the path of file to store train dataset
  test_data_path: the path of file to store test dataset
  """
  # define the transformer
  transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))
  ])
  
  # download the train dataset
  train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
  torch.save(train_dataset, train_data_path)
  
  # download the test dataset
  test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
  torch.save(test_dataset, test_data_path)



# function used for load dataset
def load_minist_data(train_data_path, test_data_path, batch_size_train=64, batch_size_test=1000):
  """
  This function load the dataset from the given path with given batch size.
  """
  # load the train dataset
  train_dataset = torch.load(train_data_path)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
  
  # load the test dataset
  test_dataset = torch.load(test_data_path)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
  
  return train_loader, test_loader

# Task 1.a: visualize the first 6 data in test dataset
def visualize_example_data(data_loader):
  examples = enumerate(data_loader)
  batch_idx, (example_data, example_targets) = next(examples)
  
  # print the shape of the data
  print(example_data.shape)
  
  # draw the subplot
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
  plt.show()


# Task 1.c: plot the training and testing error
def error_plot(error_data):
  train_losses = error_data['train_losses']
  train_counter = error_data['train_counter']
  test_losses = error_data['test_losses']
  test_counter = error_data['test_counter']

  plt.figure(figsize=(10,5))
  plt.plot(train_counter, train_losses, color='blue', zorder=1)
  plt.scatter(test_counter, test_losses, color='red', zorder=2)
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('Number of training examples seen')
  plt.ylabel('Negative log likelihood loss')
  plt.title('Training and Testing Loss over time')
  plt.show()



# class BasicNet, which is defined by requirement
class BasicNet(nn.Module):
  def __init__(self):
    super(BasicNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = self.conv1(x)  # layer 1: convolution 
    x = F.relu(F.max_pool2d(x, 2))  # layer 2: max pooling with ReLU function
    x = self.conv2(x)  # layer 3: convolution
    x = self.conv2_drop(x)  # layer 4: dropout layer
    x = F.relu(F.max_pool2d(x, 2))  # layer 5: max pooling with ReLU function
    x = x.view(-1, 320)  # flattening operation
    x = F.relu(self.fc1(x))  # layer 6: fully connected linear layer with ReLU function
    x = self.fc2(x)  # layer 7: fully connected linear layer
    return F.log_softmax(x, dim=1)  # return the output

  

# function to evaluate the net by accuracy metric
def evaluate(net, data_loader):
  # initialize
  net.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for data, target in data_loader:
      output = net(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the pred by the argmax of probabilities
      correct += pred.eq(target.view_as(pred)).sum().item()  # if pred == target, then it is correct
      total += target.size(0)

  accuracy = correct / total
  return accuracy
    

# train function: train net with data in train_loader with lr and momentum
def train(net, train_loader, train_losses, train_counter, epoch, lr=0.01, momentum=0.5):
  # initialize
  net.train()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  total_loss = 0

  # training process
  for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{5}", unit="batch")): 
    optimizer.zero_grad()  # clean the previous gradient
    output = net.forward(data)  # get the output of net
    loss = F.nll_loss(output, target)  # compute the loss
    loss.backward()  # compute the gradient
    optimizer.step()  # update parameters
    total_loss += loss.item()

    if batch_idx % 10 == 0:
      train_losses.append(loss.item())  # record the train loss
      train_counter.append((batch_idx*len(data)) + ((epoch-1)*len(train_loader.dataset)))  # record the number of data used for training 
  
  # evaluate the model and print the evaluation after training
  avg_loss = total_loss / len(train_loader)
  train_accuracy = evaluate(net, train_loader)
  print(f'Epoch {epoch}/5: Avg. Loss: {avg_loss:.4f}, '
        f'Train Accuracy: {train_accuracy*100:.2f}%.')
  
  # save the model
  patameters_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/BasicNet_parameters.pth'
  optimizer_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/BasicNet_optimizer.pth'
  torch.save(net.state_dict(), patameters_path)
  torch.save(optimizer.state_dict(), optimizer_path)

  
# test function: test the net
def test(net, test_loader, test_losses, epoch):
  # initialize
  net.eval()
  test_loss = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = net.forward(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
  
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  test_accuracy = evaluate(net, test_loader)
  print(f'Epoch {epoch}/5: Avg. Loss: {test_loss:.4f}, '
        f'Test Accuracy: {test_accuracy*100:.2f}%.')
  

def getData(command):
  #define the path of file
  train_path = "/Users/yuqipeng/Desktop/CS 5330/project/project5/MNIST_data/train_data.pt"
  test_path = "/Users/yuqipeng/Desktop/CS 5330/project/project5/MNIST_data/test_data.pt"

  if command == "d":  # download and load the dataset if the first argument is 'd'
    download_mnist_datasets(train_path, test_path)
    return load_minist_data(train_path, test_path)

  elif command == 'r':  # load the dataset if the first argument is 'r'
    return load_minist_data(train_path, test_path)
  
  return
    

def getModel(command, train_loader, test_loader):
  if command == 't':  # train the model 5 epochs
    net = BasicNet()
    n_epochs = 5
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    test(net, test_loader, test_losses, epoch=0)
    for epoch in range(1, n_epochs + 1):
      train(net, train_loader, train_losses, train_counter, epoch)
      test(net, test_loader, test_losses, epoch)

    # save the training data
    training_data_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/BasicNet_training_data.pth'
    training_data = {'train_losses': train_losses,
                      'train_counter': train_counter,
                      'test_losses': test_losses,
                      'test_counter': test_counter}
    torch.save(training_data, training_data_path)
    
  elif command == 'r':  # load the model
    parameters_path = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/BasicNet_parameters.pth'
    net = BasicNet()
    net.load_state_dict(torch.load(parameters_path))

  return net
        


  
# main function
def main(argv):
    # get the MNIST data
    train_loader, test_loader = getData(argv[1]);

    # visualize the first 6 example in test dataset
    # data, target = next(iter(test_loader))
    visualize_example_data(test_loader)


    # get the BasicNet model
    if argv[2] == 't':
      BasicNet = getModel(argv[2], train_loader, test_loader)
    
    # visualize the train errors and test errors
    training_data = '/Users/yuqipeng/Desktop/CS 5330/project/project5/models/BasicNet_training_data.pth'
    error_data = torch.load(training_data)  # load the errors
    error_plot(error_data)  # plot the errors


# run the main function
if __name__ == "__main__":
    main(sys.argv)




