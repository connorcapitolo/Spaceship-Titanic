# Use latest minimal-notebook base image from Jupyter Docker stack
FROM jupyter/minimal-notebook

# Meta-data
LABEL maintainer="Connor Capitolo <connorcapitolo@yahoo.com>" \
      description="Playing around with Spaceship Titanic dataset"

# defines the working directory for the rest of the instructions in the Dockerfile; this adds metadata to the image config
# in this case, we're creating the app directory in the Docker container and will be performing subsequent commands (as seen below) from that directory
# source: https://dockerlabs.collabnix.com/beginners/dockerfile/WORKDIR_instruction.html
WORKDIR /app

# && will run the next pip command if the previous one finished successfully
RUN pip install --no-cache-dir --upgrade pip && pip install pandas numpy matplotlib && pip install seaborn && pip install -U scikit-learn

# Add source code
# you're adding the current directory on your PC (the spaceship-titanic/ folder) to the app/ folder in the Docker container
# Add the rest of the source code. This is done last so we don't invalidate all layers when we change a line of code.
ADD . /app
