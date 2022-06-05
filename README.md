# [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Exploration and Modeling


## Steps Taken

* set up a Docker container to run the jupyter notebook
	* look to add additional security pieces Mushroom AC215 data-collector and api-service Dockerfiles and docker-shell.sh
* load in data and do initial exploration with Pandas
* create visualizations for EDA with matplotlib and seaborn
* handle missing data by removing or filling in with median (only use numeric columns so far)
* create a pytest file to remind myself on testing
* reconfigure directory structure with my notebook, testing, and data
* perform standardization and normalization
* get some output to compare models against
	* since it's a balanced binary classification problem, will just stick with using accuracy (also since the Kaggle competition uses accuracy)

## Next Steps
* 