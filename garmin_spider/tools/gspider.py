from pathlib import Path
from datetime import datetime
import logging
import os
import re
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver



class Garmin_Cursor:
    def get_currentTime(self):
        now = datetime.now()
        currentTime = now.strftime("%H:%M:%S")
        return currentTime

    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent
        DOWNLOAD_PATH = '/home/zywko/PycharmProjects/BA_Code/garmin_spider/downloads'
        if not os.path.isdir(DOWNLOAD_PATH):
            print("NOT EXIST DOWNLOAD PATH")

        LOG_PATH = Path.joinpath(BASE_DIR, '../downloads.log')
        if not os.path.isfile(LOG_PATH):
            with open(LOG_PATH, mode='a'): pass

        logging.basicConfig(filename=LOG_PATH,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        opt = Options()
        arg = 'download.default_directory={}'.format(DOWNLOAD_PATH)
        opt.add_argument(arg)
        opt.add_experimental_option("debuggerAddress", "localhost:9222")
        self.driver = webdriver.Chrome(options=opt)
        self.driver.get("https://connect.garmin.com/modern/activity/5398019870")

        # logging.basicConfig()

    def __findAndClick__(self, buttonVal, maxTime=5, error_message='', by=By.ID):
        try:
            button = WebDriverWait(self.driver, maxTime).until(
                EC.presence_of_element_located((by, buttonVal))
            )
            webdriver.ActionChains(self.driver).move_to_element(button).click(button).perform()
        except:
            error = " Error: " + str(error_message)
            logging.error(error)
            error = self.get_currentTime() + error
            print(error)
        finally:

            pass

    def __getButton__(self, buttonVal, by=By.ID, maxTime=5, error_message='Not Found button'):
        try:
            button = WebDriverWait(self.driver, maxTime).until(
                EC.presence_of_element_located((by, buttonVal))
            )
            return button
        except:
            error = " Error: " + str(error_message)
            logging.error(error)
            error = self.get_currentTime() + error
            print(error)
        finally:
            pass

    '''
    //*[@id="activityIntroViewPlaceholder"]/div[2]/button[1]/i
    '''

    def openPrevSite(self):
        self.__findAndClick__(buttonVal='//*[@id="activityIntroViewPlaceholder"]/div[2]/button[1]/i',
                              maxTime=5,
                              error_message='Could not find prev Page',
                              by=By.XPATH)

    """//*[@id="activityToolbarViewPlaceholder"]/div[3]/div[3]/button"""

    def clickSettingButton(self):
        self.__findAndClick__(buttonVal='//*[@id="activityToolbarViewPlaceholder"]/div[3]/div[3]/button',
                              maxTime=5,
                              error_message='Not found settings button',
                              by=By.XPATH)
        pass

    def exportOriginal(self):
        button = self.__getButton__(buttonVal='//*[@id="btn-export-original"]',
                                    by=By.XPATH,
                                    maxTime=5,
                                    error_message='Not Found Export Original button')
        webdriver.ActionChains(self.driver).move_to_element(button).click().perform()
        return button

    def exportTCX(self):
        button = self.__getButton__(buttonVal='//*[@id="btn-export-tcx"]',
                                    by=By.XPATH,
                                    maxTime=5,
                                    error_message='Not Found Export TCX button')
        pass

    def exportGPX(self):
        button = self.__getButton__(buttonVal='//*[@id="btn-export-gpx"]',
                                    by=By.XPATH,
                                    maxTime=5,
                                    error_message='Not Found Export TCX button')
        return button

    def exportSplitsCSV(self):
        button = self.__getButton__(buttonVal='//*[@id="btn-export-csv"]',
                                    by=By.XPATH,
                                    maxTime=5,
                                    error_message='Not Found Export TCX button')
        return button
        pass

    def download_all(self):
        self.clickSettingButton()
        b = self.__getButton__(buttonVal="#btn-export-original > a",
                               by=By.CSS_SELECTOR,
                               maxTime=2,
                               error_message='Not found Export original button')
        b.click()
        self.clickSettingButton()
        b = self.__getButton__(buttonVal="#btn-export-tcx > a",
                               by=By.CSS_SELECTOR,
                               maxTime=2,
                               error_message='Not found Export TCX button')
        b.click()
        self.clickSettingButton()
        b = self.__getButton__(buttonVal="#btn-export-gpx > a",
                               by=By.CSS_SELECTOR,
                               maxTime=2,
                               error_message='Not found Export GPX button')
        b.click()
        self.clickSettingButton()
        b = self.__getButton__(buttonVal="#btn-export-csv > a",
                               by=By.CSS_SELECTOR,
                               maxTime=2,
                               error_message='Not found Export CSV button')
        b.click()

    def isGPSactivity(self):
        map = self.__getButton__(
            buttonVal="#activity-map-canvas > div.leaflet-pane.leaflet-map-pane >"
                      " div.leaflet-pane.leaflet-overlay-pane > svg",
            by=By.CSS_SELECTOR,
            maxTime=2,
            error_message='No found gps track')
        if map:
            return True
        return False

    def isDistanceLonger(self, minDinstance=5):
        element = self.__getButton__(
            buttonVal="#react-activitySmallStats > div > div > div:nth-child(1) > div > div",
            by=By.CSS_SELECTOR,
            maxTime=2,
            error_message='No found distance of activity')
        km = int(re.findall(r'\d+', element.text)[0])

        return km >= minDinstance

    def isRunningActivity(self):
        element = self.__getButton__(
            buttonVal='//*[@id="activityIntroViewPlaceholder"]/div[1]/h3/div/div/span',
            by=By.XPATH,
            maxTime=2,
            error_message='No found Title of activity')
        return "Running" in element.text

    def isPaceInRange(self, min=180, max=330):
        element = self.__getButton__(
            buttonVal='//*[@id="react-activitySmallStats"]/div/div/div[3]/div/div',
            by=By.XPATH,
            maxTime=2,
            error_message='No found Avg Pace')
        minutes, sec = re.findall(r'\d+', element.text)
        time = (int(minutes) * 60) + int(sec)
        return time in range(min, max + 1)

    def waitUntilPageRefreshed(self):
        maxtime = 10
        element = self.__getButton__(
            buttonVal='#activityIntroViewPlaceholder > div.page-header-content > h1 > div',
            by=By.CSS_SELECTOR,
            maxTime=maxtime,
            error_message='Page not refreshed in {}.'.format(maxtime))

        pass

    def loop(self):
        conditions = []
        conditions.append(self.isGPSactivity)
        conditions.append(self.isDistanceLonger)
        conditions.append(self.isRunningActivity)
        conditions.append(self.isPaceInRange)
        try:
            while True:
                self.waitUntilPageRefreshed()
                if all(cond() for cond in conditions):
                    self.download_all()
                self.openPrevSite()

        except KeyboardInterrupt:
            logging.info("Program terminated by user.")
            exit(0)
        pass


if __name__ == '__main__':
    cursor = Garmin_Cursor()
    cursor.loop()
    pass