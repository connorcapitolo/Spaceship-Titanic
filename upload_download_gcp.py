#!/usr/bin/env python
'''Way to upload and download the datasets from GCP

How to run in Terminal: 
$ python upload_download_gcp.py
'''

# python standard library modules
import os

# third-party modules
from google.cloud import storage


gcp_project = "spaceship-titanic-352419"
bucket_name = "spaceship-titanic-dataset-bucket"
dataset_location = "data"

def upload_files():
    print("Upload")
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # where I'll be uploading files
    for filename in os.listdir(f'{dataset_location}/'):
        file_to_upload = os.path.join(dataset_location, filename)
        print(f'Uploading file {file_to_upload}...')

        # destination path in GCS
        blob = bucket.blob(file_to_upload)
        # need to specify where I'm uploading from
        blob.upload_from_filename(file_to_upload)

        print(f'Upload of {file_to_upload} complete!')


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