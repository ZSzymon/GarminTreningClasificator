from selenium import webdriver
from settings import Settings

class PolarDownloader:

    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:1111")
        SELENIUM_PATH = Settings.SELENIUM_PATH
        driver = webdriver.Chrome(options=options, executable_path=SELENIUM_PATH)
        pass


def connect():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    SELENIUM_PATH = Settings.SELENIUM_PATH
    driver = webdriver.Chrome(options=options)
    driver.get('https://www.youtube.com/')
    pass

if __name__ == '__main__':
    connect()