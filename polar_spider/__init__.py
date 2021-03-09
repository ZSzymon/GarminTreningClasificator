import pathlib
import os
import json

from .settings import Settings
from dotenv import load_dotenv
settings = Settings()
print('Init file called.')
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
#load_dotenv(BASE_DIR /'.env')

#with open(BASE_DIR / 'config.json') as config_file:
#    config = json.load(config_file)
#SELENIUM_PATH = config['CHROME_DRIVER_PATH']