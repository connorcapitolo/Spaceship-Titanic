# python standard library modules
from importlib import resources

# third party modules
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# my modules
from spaceship_titanic.features import _helper
from spaceship_titanic.data import upload_download_gcp

# special variable __version__ is a convention in Python for adding version numbers to your package
# Version of the spaceship_titanic package
__version__ = "1.0.0"

# hello = "hello from __init__.py"
# this will automatically be run when calling "python -m spaceship_titanic" from src/ folder

# Read URL of the Real Python feed from config file
# source: https://realpython.com/python-yaml/
_cfg = tomllib.loads(resources.read_text("spaceship_titanic", "config.toml"))
# print(_cfg) # {'data-directory': {'data': '../data', 'data_raw': '../data/raw'}}
data_dir = _cfg["parent-directory"]["data"]
data_raw_dir = _cfg["parent-directory"]["data_raw"]

n_splits = _cfg["model-parameters"]["n_splits"]
random_state = _cfg["model-parameters"]["random_state"]
test_size = _cfg["model-parameters"]["test_size"]
scoring = _cfg["model-parameters"]["scoring"]
model_folder_name = _cfg["model-parameters"]["model_folder_name"]

gcp_project = _cfg["gcp-related"]["gcp_project"]
bucket_name = _cfg["gcp-related"]["bucket_name"]
prefix_path = _cfg["gcp-related"]["prefix_path"]
