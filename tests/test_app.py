import pytest
import os
from .context import polar_spider
from selenium import webdriver

from pathlib import Path

def test_app(capsys, example_fixture):
    # pylint: disable=W0612,W0613
    #polar_spider.Blueprint.run()
    captured = capsys.readouterr()
    #assert "Hello World..." in captured.out

def test_is_exist():

    path2= polar_spider.settings.SELENIUM_PATH
    driver = webdriver.Chrome(executable_path=path2)
    driver.get('https://www.google.pl')
    print(driver.title)
    print(driver.current_url)

if __name__ == '__main__':
    test_is_exist()