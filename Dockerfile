# Use latest minimal-notebook base image from Jupyter Docker stack
FROM jupyter/minimal-notebook:latest

# Meta-data
LABEL maintainer="Connor Capitolo <connorcapitolo@yahoo.com>" \
      description="Playing around with Spaceship Titanic dataset"

# defines the working directory for the rest of the instructions in the Dockerfile; this adds metadata to the image config
# in this case, we're creating the app directory in the Docker container and will be performing subsequent commands (as seen below) from that directory
# source: https://dockerlabs.collabnix.com/beginners/dockerfile/WORKDIR_instruction.html
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Add source code
# you're adding the current directory on your PC (the spaceship-titanic/ folder) to the app/ folder in the Docker container
# Add the rest of the source code. This is done last so we don't invalidate all layers when we change a line of code.
ADD . /app

# adding my basic git credentials
# could also pass this as environment variables into docker-shell.sh
RUN git config --global user.email "connorcapitolo@yahoo.com" && git config --global user.name "connorcapitolo"

# using .visualize() method was giving errors with graphviz executable if I tried to use "pip-install graphviz", and would give a permission and password error when trying "apt-get install graphviz" (with or without sudo)
# RUN conda install python-graphviz

# && will run the next pip command if the previous one finished successfully
RUN python -m pip install -r /app/requirements.txt && pip install "prefect[viz]"
# would typically need to "RUN pre-commit install", but since we've mounted the local path, this only needs to be done the first time (could set up a check to see if this is done"
# source: https://stackoverflow.com/questions/68754821/how-to-pre-install-pre-commit-into-hooks-into-docker

