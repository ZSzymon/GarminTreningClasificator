import json
import pathlib


class Settings:
    FILE_DIR = pathlib.Path(__file__).parent.absolute()
    with open(FILE_DIR / 'config.json') as config_file:
        config = json.load(config_file)
    SELENIUM_PATH = config['CHROME_DRIVER_PATH']
