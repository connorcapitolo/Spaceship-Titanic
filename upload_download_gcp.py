#!/usr/bin/env python
"""Way to upload and download the datasets from GCP

How to run in Terminal: 
$ python upload_download_gcp.py
"""

# python standard library modules
import os
import sys
import logging
import time
from datetime import datetime

# print(os.getcwd()) # /app

# third-party modules
from google.cloud import storage

gcp_project = "spaceship-titanic-352419"
bucket_name = "spaceship-titanic-dataset-bucket"
prefix_path = "data_06-06-2022_11-12PM"
data_folder_name = "data"


def upload_files():
    logging.info("Begin Upload")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # where I'll be uploading files
    for filename in os.listdir(f"{dataset_location_local}/"):
        file_to_upload_local_location = os.path.join(dataset_location_local, filename)
        logging.info(f"Uploading local file {file_to_upload_local_location} to GCP...")

        # destination path in GCS
        file_to_upload_gcp_location = os.path.join(dataset_location_gcp, filename)
        blob = bucket.blob(file_to_upload_gcp_location)
        # need to specify where I'm uploading from
        blob.upload_from_filename(file_to_upload_local_location)

        logging.info(
            f"Upload complete to GCP bucket {bucket_name} with path of {file_to_upload_gcp_location} complete!"
        )

    logging.info("All uploads completed successfully!!!")


def download_files(file_to_download: str = "train.csv"):
    source_data_path = os.path.join(os.getcwd(), data_folder_name)  # '/app/data'
    if not os.path.exists(source_data_path):
        os.mkdir(source_data_path)

    storage_client = storage.Client(project=gcp_project)
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=f"{prefix_path}/")
    for blob in blobs:
        # print(blob.name)
        if blob.name.endswith(file_to_download):
            blob.download_to_filename(os.path.join(source_data_path, file_to_download))


if __name__ == "__main__":
    # upload_files()

    # filemode = 'w' means the log file is no longer appended to, so the messages from earlier runs are lost
    logging_filename = "logger.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] :: %(message)s",
        filename=logging_filename,
        filemode="w",
        encoding="utf-8",
    )

    module_description = sys.modules[
        "__main__"
    ]  # <module '__main__' from '/Users/connorcapitolo/Desktop/josh_philly_atlas/4_add_logging.py'>
    # splitting the module path, and then getting the name
    # could also just use sys.argv[0] here
    module_name = module_description.__file__.split("/")[-1]
    logging.info(f"Program {module_name} started")

    dataset_location_local = f"data"

    dt_string = datetime.now().strftime("%m-%d-%Y_%I-%M%p")
    dataset_location_gcp = f"{dataset_location_local}_{dt_string}"

    download_files()

"""
https://cloud.google.com/docs/authentication/getting-started#create-service-account-console

def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)

implicit()
"""
