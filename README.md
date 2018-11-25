# What is it?
This repository contains all the code for my CSC 585 HW 4 submission. It includes a Dockerfile to build the Docker container, this README file, and a directory containing files to train or test the model. The Scottish Gaelic data here comes from Lamb and Danso's Annotated Reference Corpus of Scottish Gaelic (ARCOSG). Much of my original LSTM was cobbled together from Michael Capizzi's DyNet tutorial.

This Docker container was built and tested on an Ubuntu 18 Linux machine without a GPU.

Files in `docker_files` directory:
- `MT_train.py`:  code to train approach #1 (forward-only)
- `MT_test.py`:  code to test approach #1
- `MT_train2.py`:  code to train approach #2 (bidirectional)
- `MT_test2.py`:  code to test approach #2
- `model_baseline`:  a saved model of the trained LSTM, used to test the model
- `data` directory
  - `train_all.txt`:  training data
  - `dev_all.txt`:  dev data
  - `bigtest_all.txt`:  testing data
  - `irish_a_500Sent`:  500 Irish sentences
  - `irish_a_600Sent`:  600 Irish sentences
  - `irish_a_1kSent`:  1000 Irish sentences
- `models` directory:  contains all the different trained models

# Installation
 From the repo directory, simply build the Docker container with the following command:

 `docker build -t zupon-csc585-hw4 .`

The Docker container adds all the relevant files, installs the needed dependencies `numpy`, `tqdm`, and `dynet`, and runs the code to test the model.

# Running the container
 
 After you've built the model, run the container with this command:

 `docker run zupon-csc585-hw4 MT_test.py`

 to test approach #1; or run the container with this command:

 `docker run zupon-csc585-hw4 MT_test2.py`

 to test approach #2

Running the Docker container will test the model, using a saved model included. By default, it will test the `teacher_weighted` model with `600` Irish sentences added. To test other models, change the hyperparameters at the top of each `MT_test` file.

As it is now, the dyNet configuration will set a memory limit of 4 GB.
