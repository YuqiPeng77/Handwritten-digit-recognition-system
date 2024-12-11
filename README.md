# Project 5: Recognition using Deep Networks


## Author
Yuqi Peng and Ruining Yang


## File Introduction
- `BasicNet.py`: The model of Bassinet, containing some data download and load functions, some useful plot functions, train and test functions. (Task 1a, 1b, 1c, 1d)

- `testBasicNet.py`: Test the BasicNet model by show the result of first 10 images prediction in test set. Also test the model on my own handwriting number inputs. (Task 1e, 1f)

- `examineBasicNet.py`: Examine the network by analyze the first convolution layer and its effect on the image. (Task 2)

- `transferLearning.py`: re-use the BasicNet to recognize three Greek letters: $\alpha$, $\beta$ and $\gamma$. (Task 3)

- `experiment.py`: explore the best options of `batch_size`, `num_of_filters`, `kernel_size`, `dropout_rate`, `num_of_fc1_out_features`, `epoches`. (Task 4 & Extension 1)

- `examineResNet18.py`: Examine the ResNet18 model by analyze the first two convolution layers and its effect on the image. (Extension 2)

- `NewBasicNet.py`: Change the first convolution layer to defined filter bank, retrain the rest layers. (Extension 3)



## How to run the program

- Run the `BasicNet.py` file in terminal by command `python <BasicNet.py> <arguments>` with following arguments:\
    **First argument choices:**
    - `d`: Download and load the dataset.
    - `r`: Just load the dataset (already download).

    **Second argument choices:**
    - `t`: Train the `BasicNet` model with 5 epochs.

- Run the other files in terminal by command `python <file name>` directly.



## My Greek Letter Link
https://drive.google.com/drive/folders/1tsSXRIVpJHY6KD3zBNT-QHGufvsxFVxB?usp=sharing

    











