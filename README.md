# [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Exploration and Modeling

## How to Run

1. In a terminal (if you're on Mac),

```
$ sh docker-shell.sh
```

2. Copy and paste the link into your browser that begins with `127.0.0.1`

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

## Next Steps
* look to add additional security pieces based on the Mushroom AC215 data-collector and api-service Dockerfiles and docker-shell.sh