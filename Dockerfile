# this default argument may or may not be overwritten from the "docker build" command using "--build-arg"
# Variables set using ARG do not persist once the container has been built. To set an environment variable that is available when the container is running, use ENV.
ARG BUILD_VERSION=no-viz

# Use latest minimal-notebook base image from Jupyter Docker stack
FROM jupyter/minimal-notebook:latest as base

# Meta-data
LABEL maintainer="Connor Capitolo <connorcapitolo@yahoo.com>" \
      description="Playing around with Spaceship Titanic dataset"

# Python wants UTF-8 locale
# source: AC215 Mushroom App - api-service/Dockerfile
ENV LANG=C.UTF-8

# Tell Python to disable buffering so we don't lose any logs.
# source: AC215 Mushroom App - api-service/Dockerfile
ENV PYTHONUNBUFFERED=1

# defines the working directory for the rest of the instructions in the Dockerfile; this adds metadata to the image config
# in this case, we're creating the app directory in the Docker container and will be performing subsequent commands (as seen below) from that directory
# source: https://dockerlabs.collabnix.com/beginners/dockerfile/WORKDIR_instruction.html
WORKDIR /app

# COPY is similar to ADD, but it focuses solely on source files (doesn't inlude URLs or untarring)
# source: AC215 Mushroom App - api-service/Dockerfile
COPY Pipfile Pipfile.lock /app/

RUN pip install --no-cache-dir --upgrade pip && \
      pip install pipenv

# this is the equivalent of an if-else for the Dockerfile using build stages
FROM base as branch-version-no-viz

FROM base as branch-version-viz
# using .visualize() method was giving errors with graphviz executable if I tried to use "pip-install graphviz", and would give a permission and password error when trying "apt-get install graphviz" (with or without sudo)
RUN conda install python-graphviz


FROM branch-version-${BUILD_VERSION} AS final

# --mount=type=cache: using BUILDKIT, getting this speedup because we are no longer downloading all the Python packages. They were cached by the package manager (pip in this case) and stored in a cache volume mount. The volume mount is provided to the run step so that pip can reuse our already downloaded packages. This happens outside any Docker layer caching
# source: https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project#:~:text=Docker%20will%20use%20cache%20during,were%20changed%20or%20not.
# && will run the next pip command if the previous one finished successfully
# would typically need to "RUN pre-commit install", but since we've mounted the local path, this only needs to be done the first time (could set up a check to see if this is done)
# source: https://stackoverflow.com/questions/68754821/how-to-pre-install-pre-commit-into-hooks-into-docker
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt && pip install "prefect[viz]"

# --system: This tells pipenv that rather than create a virtualenv with our installed packages, we should install them directly in the the container’s system python.
# --deploy flag tells pipenv to blow up if the lock file is out of date. That is, if the requirements specified in the Pipfile no longer align with the hashes specified in the lock file.\
# source: https://jonathanmeier.io/using-pipenv-with-docker/
# RUN pipenv install --system --deploy

# install packages exactly as specified in Pipfile.lock
RUN pipenv sync


# copy source code
# you're adding the current directory on your PC (the spaceship-titanic/ folder) to the app/ folder in the Docker container
# Copy the rest of the source code. This is done last so we don't invalidate all layers when we change a line of code.
COPY . /app