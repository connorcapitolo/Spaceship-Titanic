# [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Exploration and Modeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Directory structure is based on [CookieCutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

### Note on Git

Since adding [pre-commit](https://pre-commit.com/), Git must be used within the Jupyter Notebook container, as that is what contains the `pre-commit` and `black` packages
* When running Docker image for the first time, need to run `pre-commit install` from the JupyterLab Terminal since we've mounted the source path
* note that the assumption is your Git username and email exist within your *~/.gitconfig* file (at the global level, e.g. you should see both of these when you run `git config --global --list`)

## How to Run

1. In a Terminal on your host machine, run either

```
$ sh docker-shell.sh
```

or

```
$ sh docker-shell.sh viz
```

The former will not allow you to use Prefect's [Flow Visualization](https://docs.prefect.io/core/advanced_tutorials/visualization.html), while the latter will (it takes about a minute longer to build the Docker image)

2. Copy and paste the link into your browser that begins with `127.0.0.1`. For example, it will look like `http://127.0.0.1:8888/lab?token=3a991c83fe52a85611ba3d8d5215499fe6a6d859f598798b`

3. From JupyterLab, use its provided Terminal to navigate to the `src` folder

```
$ cd src
```

4. From the JupyterLab Terminal, run the Prefect pipeline

```
$ python -m spaceship_titanic
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

## GCP Buckets For Uploading/Retrieving Data

Follow the steps for [Setting Up GCP Service Account and Credentials](https://github.com/dlops-io/mushroom-app/tree/02-setup-gcp-credentials)

* **Note**: The role to be selected should be *Storage Object Admin* so you can upload and download to the GCP bucket, rather than *Storage Object Viewer*
* You can see the *GOOGLE_APPLICATION_CREDENTIALS* environment variable within `docker_shell.sh` (GCP will search for this and utilize it when necessary)