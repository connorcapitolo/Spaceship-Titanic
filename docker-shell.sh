#!/bin/bash

# since this file is being run as an executable in a Unix operating system, the shebang (#!) lets the program loader mechanism know that the rest of the line should be run as an interpreter directive/executable; it specifies what program should be called to run the script
# in this particular case, /bin/bash is the most common shell (a shell allows you to run programs, give them input, and inspect their output in a semi-structured way) for user login of the Linux system. The term 'bash' is an acronym for Bourne-again shell, and it can access the vast majority of scripts since it is well developed with good syntax and features
# sources for this description: https://www.a2hosting.com/kb/developer-corner/linux/using-the-shebang, https://en.wikipedia.org/wiki/Shebang_(Unix), https://missing.csail.mit.edu/2020/course-shell/; https://medium.com/@codingmaths/bin-bash-what-exactly-is-this-95fc8db817bf

# sources: https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html#index-set; https://unix.stackexchange.com/questions/255581/what-does-set-command-without-arguments-do/255588; the command 'help set' run in Terminal
# `set -e` will exit immediately if a command exits with a non-zero status
set -e

# NAMEs (in this case, IMAGE_NAME) are marked for automatic export to the environment of subsequently executed commands
# source: the command 'help export' run in Terminal
export IMAGE_NAME="connorcapitolo-jupyter"

# `docker build` is saying to build a Docker image
# `-t $IMAGE_NAME` is the equivalent of `-t "connorcapitolo-jupyter"`, and it is saying to tag the Docker image that we create from the Dockerfile as with the repository name "connorcapitolo-jupyter"; what this is missing is actually a "tag", which is would be something like "$IMAGE_NAME$:latest to let us know what version of this is being used
# `-f Dockerfile .` is specifying to look inside the current directory (this is specified by the dot) for the Dockerfile, and the instructions inside the Dockerfile should be executed
# source: https://colab.research.google.com/drive/1zPmsNQ_JmHohoGikzAUpRY0qrmdTnJjg?usp=sharing
docker build -t $IMAGE_NAME -f Dockerfile .

# `docker run` is specifying to run a Docker container
#  `--rm --name $IMAGE_NAME` is saying to delete the container $IMAGE_NAME when it has stopped running
# '-v' is saying to mount a volume; we are mounting the current working directory into the container, so even if we put anything in this particular folder (spaceship-titanic/) it will be seen in the container; this will allow us to see any Notebook updates reflected on our local machine
# '-it' instructs Docker to allocate a pseudo-TTY connected to the container’s stdin; creating an interactive bash shell in the container
# '--name' is the name of the container; the container and the image could have different names b/c you can have multiple containers from a single image
# the second '$IMAGE_NAME' is what is actually saying to run the image 
# sources: https://docs.docker.com/engine/reference/commandline/run/#assign-name-and-allocate-pseudo-tty---name--it; https://colab.research.google.com/drive/1zPmsNQ_JmHohoGikzAUpRY0qrmdTnJjg?usp=sharing
docker run --rm -it -p 8888:8888 -v "$(pwd)/:/app/" $IMAGE_NAME
