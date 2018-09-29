# What is it?
This repository contains all the code for my CSC 585 HW 3 submission. It includes a Dockerfile to build the Docker container, this README file, and a directory containing the training/dev/testing data and the Python code to train or test the model. The Scottish Gaelic data here comes from Lamb and Danso's Annotated Reference Corpus of Scottish Gaelic (ARCOSG). The code to train the LSTM and test the model were developed by me in Fall 2017, using a dyNet tutorial by Michael Capizzi as a starting point.

# Installation
 From the repo directory, simply build the Docker container with the following command:

 `docker build -t zupon-csc585-hw3`

The Docker container adds all the relevant files, installs the needed dependencies `numpy`, `tqdm`, and `dynet`, and runs the code to test the model.

# Running the container
 
 After you've built the model, run the container with this command:

 `docker run zupon-csc585-hw3`

Running the Docker container will only _test_ the model, using the saved model included. As it is set up now, testing will test on the `dev` data, not the testing data. However, results on `dev_all` (the large dev set) and `bigtest_all` (the large testing set) are comparable:
- test on `dev_all`		= 0.8054 overall, 0.2148 on unknown words
- test on `bigtest_all`	= 0.8052 overall, 0.2135 on unknown words

As it is now, the dyNet configuration will set a memory limit of 4 GB.
