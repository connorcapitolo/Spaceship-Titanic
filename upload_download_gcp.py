#!/usr/bin/env python
'''Way to upload and download the datasets from GCP

How to run in Terminal: 
$ python upload_download_gcp.py
'''

# python standard library modules
import os
import logging
import time
from datetime import datetime

# print(os.getcwd()) # /app

# third-party modules
from google.cloud import storage


gcp_project = "spaceship-titanic-352419"
bucket_name = "spaceship-titanic-dataset-bucket"

dataset_location_local = f"data"

dt_string = datetime.now().strftime("%m-%d-%Y_%I-%M%p")
dataset_location_gcp = f'{dataset_location_local}_{dt_string}'

def upload_files():
    print("Upload")
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    

    # where I'll be uploading files
    for filename in os.listdir(f'{dataset_location_local}/'):
        file_to_upload_local_location = os.path.join(
            dataset_location_local, filename)
        print(f'Uploading local file {file_to_upload_local_location} to GCP...')

        # destination path in GCS
        file_to_upload_gcp_location = os.path.join(
            dataset_location_gcp, filename)
        blob = bucket.blob(file_to_upload_gcp_location)
        # need to specify where I'm uploading from
        blob.upload_from_filename(file_to_upload_local_location)

        print(
            f'Upload complete to GCP bucket {bucket_name} with path of {file_to_upload_gcp_location} complete!')


upload_files()

'''
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
'''
