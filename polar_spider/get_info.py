import logging
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver

from pathlib import Path

from datetime import datetime


class Polar_Cursor:
    def get_currentTime(self):
        now = datetime.now()
        currentTime = now.strftime("%H:%M:%S")
        return currentTime



    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent
        DOWNLOAD_PATH = Path.joinpath(BASE_DIR, 'downloads')
        logging.basicConfig(filename='/home/zywko/PycharmProjects/ba_v2/polar_spider/download.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        opt = Options()
        opt.add_argument(f"download.default_directory={DOWNLOAD_PATH}")
        opt.add_experimental_option("debuggerAddress", "localhost:9222")
        self.driver = webdriver.Chrome(options=opt)
        self.driver.get("https://flow.polar.com/training/analysis/4999176666")

        #logging.basicConfig()

    def __findAndClick__(self, buttonVal, maxTime=5, error_message='', by=By.ID):
        try:
            button = WebDriverWait(self.driver, maxTime).until(
                EC.presence_of_element_located((by, buttonVal))
            )
            webdriver.ActionChains(self.driver).move_to_element(button).click(button).perform()
        except:
            error = self.get_currentTime() + " Error: "+str(error_message)
            logging.error(error)
            print(error)
        finally:
            #logging.info()
            pass
    def prev(self):
        left_arrow = self.driver.find_element_by_xpath('/html/body/div[4]/div[1]/div/div[1]/div/ul/li[1]')
        prev_button = left_arrow.find_element_by_tag_name('a')
        prev_training_href = prev_button.get_attribute('href')
        self.driver.get(prev_training_href)

    def next(self):
        next_button = self.driver.find_element_by_xpath \
            ('/html/body/div[4]/div[1]/div/div[1]/div/ul/li[2]/a')
        return next_button


    def clickExportButton(self):
        self.__findAndClick__('exportTrainingSessionPopup',
                            error_message='Not found export button.', maxTime=10)

    def downloadCSV(self):
        self.__findAndClick__('exportTrainingSessionRRAsCsvFile', error_message='CSV not downloaded')

    def downloadGPX(self):
        self.__findAndClick__('exportTrainingSessionRRAsCsvFile', 2, error_message='GPX not downloaded')

    def clickExportButton_old(self):
        try:
            button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, "exportTrainingSessionPopup"))
            )
            webdriver.ActionChains(self.driver).move_to_element(button).click(button).perform()
        except:
            print("Here should be log.")
        finally:
            print("log")

    def downloadCSV_old(self):
        try:
            csv_button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, "exportTrainingSessionRRAsCsvFile"))
            )
            webdriver.ActionChains(self.driver).move_to_element(csv_button).click(csv_button).perform()
        finally:
            pass
        # csv_button = self.driver.find_element_by_id('exportTrainingSessionAsCsvFile')
        # return csv_button

    def downloadGPX_old(self):
        succes = True
        try:
            gpx_button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, "exportTrainingSessionRRAsCsvFile"))
            )
            webdriver.ActionChains(self.driver).move_to_element(gpx_button).click(gpx_button).perform()
        except:
            succes = False
        finally:
            return succes

    def loop(self):
        while True:
            self.clickExportButton()
            self.__findAndClick__("//*[contains(text(), 'Session (TCX)')]",
                                2, 'not downloaded TCV', By.XPATH)
            self.__findAndClick__("//*[contains(text(), 'Session (CSV)')]",
                                2, 'not downloaded CSV', By.XPATH)

            self.__findAndClick__("//*[contains(text(), 'HRV data (CSV)')]",
                                2, 'not downloaded HRV in CSV', By.XPATH)

            self.__findAndClick__("//*[contains(text(), 'Route (GPX)')]",
                                2, 'not downloaded GPX ', By.XPATH)
            self.prev()


if __name__ == '__main__':
    cursor = Polar_Cursor()
    cursor.loop()
