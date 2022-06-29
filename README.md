# [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Exploration and Modeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![connorcapitolo](https://circleci.com/gh/connorcapitolo/Spaceship-Titanic.svg?style=shield)](https://app.circleci.com/pipelines/github/connorcapitolo/Spaceship-Titanic?branch=main&filter=all)

![package testing](https://github.com/connorcapitolo/Spaceship-Titanic/actions/workflows/python-package.yml/badge.svg)


Directory structure is based on [CookieCutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

### Note on Git

**Since adding [pre-commit](https://pre-commit.com/), Git must be used within the Jupyter Notebook container, as that is what contains the `pre-commit` and `black` packages**

We've mounted the source path, so when running Docker image for the first time, need to run in JupyterLab Terminal

 ```
 pipenv shell
 pre-commit install
 ``` 
 
This sets up the Git hooks and [allows `pre-commit` to run automatically on `git-commit`](https://pre-commit.com/#3-install-the-git-hook-scripts)

* note that the assumption is your Git username and email exist within your *~/.gitconfig* file (at the global level, e.g. you should see both of these when you run `git config --global --list`)

## How to Run

This package uses [Pipenv](https://pipenv.pypa.io/en/latest/), a tool that automatically creates and manages a virtualenv for your projects, as well as adds/removes packages from your `Pipfile` as you install/uninstall packages. It also generates the ever-important `Pipfile.lock`, which is used to produce deterministic builds.

1. In a Terminal on your host machine, run either

```
$ sh docker-shell.sh
```

or

```
$ sh docker-shell.sh viz
```

The former will not allow you to use Prefect's [Flow Visualization](https://docs.prefect.io/core/advanced_tutorials/visualization.html), while the latter will (it takes about a minute longer to build the Docker image)

When building this for the first time, you should expect this to take upwards of five minutes. When re-running these steps, it should take only a couple seconds (as long as you don't remove the Docker images from your Docker), as all the layers have been cached by Docker from your previous build.

1. Copy and paste the link into your browser that begins with `127.0.0.1`. For example, it will look like `http://127.0.0.1:8888/lab?token=3a991c83fe52a85611ba3d8d5215499fe6a6d859f598798b`

2. From JupyterLab, use its provided Terminal to navigate to the `src` folder

```
$ cd src
```

4. From the JupyterLab Terminal, run the Prefect pipeline either

```
$ pipenv run python -m spaceship_titanic
```

or 

```
$ pipenv run python -m spaceship_titanic -d
```

or 

```
$ pipenv run python -m spaceship_titanic -xgb
```

The first one utilizes the `data/raw/train.csv` file, while the second one downloads the training dataset from a GCP bucket, and the third one solely runs [XGBoost](https://xgboost.readthedocs.io/en/stable/) with automatic handling of missing values. Note that the `Dockerfile` created a Python virtual environment with all the necessary packages using the *pipenv sync* command (install packages exactly as specified in `Pipfile.lock`). The `pipenv run` command will access the Pipenv-created virtual environment without launching a shell, allowing us to automatically run our scripts from the command line.

5. Additionally, you can perform the same commands as Steps 3 and 4 from the project root directory in JupyterLab's Terminal

```
bash run_spaceship_titanic_package.sh
```

or 

```
bash run_spaceship_titanic_xgboost_only.sh
```

## Testing

This repository utilizes [pytest](https://docs.pytest.org/en/7.1.x/) for testing purposes

1. Make sure you have completed through step 2 from the `How to Run` section

2. From JupyterLab, use its provided Terminal to navigate to the `tests` folder

```
$ cd tests
```

3. From the JupyterLab Terminal, run either 

```
bash run_tests.sh
```

or 

```
bash run_tests.sh include
```

The former will only use the provided data within the repository, while the latter will look to download the data from the GCP bucket (please note this is not available to external users)

4. Rather than performing steps 2 and 3, from the project root directory in JupyterLab's Terminal

```
bash run_test_suite.sh
```

## GCP Buckets For Uploading/Retrieving Data

Follow the steps for [Setting Up GCP Service Account and Credentials](https://github.com/dlops-io/mushroom-app/tree/02-setup-gcp-credentials)

* **Note**: The role to be selected should be *Storage Object Admin* so you can upload and download to the GCP bucket, rather than *Storage Object Viewer*
* You can see the *GOOGLE_APPLICATION_CREDENTIALS* environment variable within `docker_shell.sh` (GCP will search for this and utilize it when necessary)