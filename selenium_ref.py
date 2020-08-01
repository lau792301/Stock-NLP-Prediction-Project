# %%
import os
import datetime
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

# %%
import logging
logging.getLogger().setLevel(logging.INFO)

DOWNLOAD_PATH = os.getcwd()

# %%
class Scraper(object):
    def __init__(self):


        ### Firefox Setting ###
        fp = webdriver.FirefoxProfile()
        fp.set_preference("browser.download.dir",DOWNLOAD_PATH);
        fp.set_preference("browser.download.folderList", 2)
        fp.set_preference("browser.download.manager.showWhenStarting",False);
        fp.set_preference("browser.helperApps.neverAsk.saveToDisk","application/vnd.ms-excel")
        fp.set_preference("intl.accept_languages","zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6")
        fp.set_preference("general.useragent.override", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0")

        self.driver = webdriver.Firefox(firefox_profile=fp, executable_path = './geckodriver')
        ### Init Login ###
        # self.driver.get(URL)
        # self.login()
    def _export_button_click(self):
        try:
            export_button = './/button[contains(@class, "artdeco-button--primary")]'
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH,export_button))
            )
            self.driver.find_elements_by_xpath(export_button)[0].click()
            print('Export button is clicked')
        except:
            print('Export Button does not exist')

    def _time_picker_click(self):
        try:
            time_picker = './/div[contains(@class, "timepicker")]'
            WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, time_picker)))
            self.driver.find_elements_by_xpath(time_picker)[0].click()
            print('Time_picker is clicked')
        except:
            print('Time Picker Button does not exist')
    
    def _custom_time_range_click(self):
        try:
            container = './/div[contains(@class, "org-analytics-time-range-dropdown__dropdown-content-container")]'
            WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, container)))
            container = self.driver.find_elements_by_xpath(container)[0]
            date_button_list = container.find_elements_by_xpath('.//ul/li')
            for li in date_button_list:
                button_name = li.text
                if button_name == 'Custom':
                    li.click()
                    print('Custom date button clicked')
        except:
            print('Dropdown-content-container does not exist')
    
    def _custom_date_select(self, date):
        def clean_date(input_element):
            for _ in range(12):
                input_element.send_keys(Keys.BACK_SPACE)
        # Start Date
        input_start_date = self.driver.find_element_by_class_name("artdeco-start-date")
        input_start_date.click()
        clean_date(input_start_date)
        input_start_date.send_keys(date) #month/day/year e.gstart_date = "5/2/2020"

        # End Date
        input_end_date = self.driver.find_element_by_class_name("artdeco-end-date")
        input_end_date.click()
        clean_date(input_end_date)
        input_end_date.send_keys(date)
        input_end_date.send_keys(Keys.TAB)

        # Click for update
        button_list = self.driver.find_elements_by_class_name("org-analytics-time-range-dropdown__dropdown-content-buttons")
        button_list[0].find_elements_by_xpath('.//button[contains(@class, "button")]')[0].click()
        print('Date is selected')