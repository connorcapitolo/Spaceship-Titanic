# [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Exploration and Modeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Directory structure is based on [CookieCutter](https://drivendata.github.io/cookiecutter-data-science/)

Since adding [pre-commit](https://pre-commit.com/), Git must be used within the Jupyter Notebook container, as that is what contains the `pre-commit` and `black` packages
* may need to run `pre-commit install` from the JupyterLab Terminal when running Docker image for the first time
* note that the assumption is your Git username and email exist within your *~/.gitconfig* file (at the global level, e.g. you should these when you run `git config --global --list`)

## How to Run

1. In a terminal (if you're on Mac),

```
$ sh docker-shell.sh
```

2. Copy and paste the link into your browser that begins with `127.0.0.1`. For example, it will look like `http://127.0.0.1:8888/lab?token=3a991c83fe52a85611ba3d8d5215499fe6a6d859f598798b`

3. From Jupyterlab, use the Terminal to navigate to the src folder

```
cd src
```

4. Run the Prefect pipeline

```
python -m spaceship_titanic
```

## Steps Taken

* set up a Docker container to run the jupyter notebook
	* this includes the Dockerfile and docker-shell.sh for ease-of-use
* load in data and do initial exploration with Pandas
* create visualizations for EDA with matplotlib and seaborn
* handle missing data by removing or filling in with median (only use numeric columns so far)
* create a pytest file to remind myself on testing
* reconfigure directory structure with my notebook, testing, and data
* perform standardization and normalization
* get some output to compare models against
	* since it's a balanced binary classification problem, will just stick with using accuracy (also since the Kaggle competition uses accuracy)
* Figure out how to connect to GCP with the `upload_download_gcp.py`
* Add some basic logging to `upload_download_gcp.py`
* Add Prefect for a basic ETL pipeline `prefect_etl.py`
* Add some documentation based on Google's Python Style Guide `prefect_etl.py`

## Next Steps
* look to add additional security pieces based on the Mushroom AC215 data-collector and api-service Dockerfiles and docker-shell.sh
