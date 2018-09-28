# use a container for python 3
FROM python:3

# this will copy the the training/dev/testing data, the saved model, and the Python code to train and test the model to the Docker container
ADD /docker_files ./

# to install dependencies numpy, tqdm, and dynet
RUN pip install numpy tqdm dynet

# This is the command that will be run when you start the container
CMD [ "python", "./lstm_test.py" ]
