# use a container for python 3
FROM python:3

# this will copy the the file my_nlp_hw.py to the Docker container
ADD /docker_files ./

# to install dependencies with pip, see the following example
RUN pip install numpy tqdm dynet

# This is the command that will be run when you start the container
# Here it's a python script, but it could be a shell script, etc.
CMD [ "python", "./lstm_test.py" ]