# What is it?
This repository contains all the code for my CSC 585 HW 3 submission. It includes a Dockerfile to build the Docker container, this README file, and a directory containing files to train or test the model. The Scottish Gaelic data here comes from Lamb and Danso's Annotated Reference Corpus of Scottish Gaelic (ARCOSG). The code to train the LSTM and test the model were developed by me in Fall 2017, using a dyNet tutorial by Michael Capizzi as a starting point.

This Docker container was built and tested on an Ubuntu Linux machine without a GPU.

Files in `docker_files` directory:
- `lstm_train.py`:  code to train the LSTM
- `lstm_test.py`:  code to test the model
- `model_baseline`:  a saved model of the trained LSTM, used to test the model
- `train.txt`:  training data
- `dev.txt`:  dev data
- `test.txt`:  testing data

# Installation
 From the repo directory, simply build the Docker container with the following command:

 `docker build -t zupon-csc585-hw3 .`

The Docker container adds all the relevant files, installs the needed dependencies `numpy`, `tqdm`, and `dynet`, and runs the code to test the model.

# Running the container
 
 After you've built the model, run the container with this command:

 `docker run zupon-csc585-hw3`

Running the Docker container will test the model, using the saved model included. As it is now, the dyNet configuration will set a memory limit of 4 GB.
