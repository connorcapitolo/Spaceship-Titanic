# [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Exploration and Modeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## How to Run

1. In a terminal (if you're on Mac),

```
$ sh docker-shell.sh
```

2. Copy and paste the link into your browser that begins with `127.0.0.1`. For example, it will look like `http://127.0.0.1:8888/lab?token=3a991c83fe52a85611ba3d8d5215499fe6a6d859f598798b`

3. Have some fun and play around!

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
