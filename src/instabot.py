import os
import time
import logging
import requests
from io import BytesIO

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs

from src.yolov3 import YoloV3

DESIRED_LABELS = ['dog', 'cat']


class InstaBot:
    def __init__(self, **kwargs):
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')
        self.path_out = kwargs.get('path_out')

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--incognito')
        self.driver = webdriver.Chrome(chrome_options=chrome_options)

        self.yolov3 = YoloV3()

    @staticmethod
    def get_image(url_image):
        """
        Download image from url image.
        """
        image_data = requests.get(url_image)
        image_data_content = image_data.content
        return Image.open(BytesIO(image_data_content))

    def save_image(self, image, label):
        """
        Save image in the folder indicated by the label.
        """
        out_folder_path = os.path.join(self.path_out, label)
        os.makedirs(out_folder_path, exist_ok=True)
        out_image_path = os.path.join(out_folder_path, '{}.jpg'.format(str(time.time())))
        image.save(out_image_path)

    def labels_in_image(self, image):
        """
        Get object labels in the image using Yolo V3.
        """
        return list(set(self.yolov3(image=image)))

    def get_data_images(self):
        """
        Get data images from html.
        """
        html_to_parse = str(self.driver.page_source)
        html = bs(html_to_parse, 'html.parser')
        return html.findAll('img', {'class': 'FFVAD'})

    def download_images(self):
        """
        Main process to find, filter and download the images.
        Obtaining the images by scrolling and using the YoloV3 to check if it includes what we are looking for.
        """
        downloaded_images = []
        r_scroll_h = 'return document.body.scrollHeight'
        scroll_h = 'window.scrollTo(0, document.body.scrollHeight);'

        lh = self.driver.execute_script(r_scroll_h)
        while True:
            self.driver.execute_script(scroll_h)
            self.driver.implicitly_wait(1)
            nh = self.driver.execute_script(r_scroll_h)

            if nh == lh:
                self.driver.execute_script(scroll_h)
                continue
            else:
                lh = nh
                self.driver.implicitly_wait(1)

            all_images_data = self.get_data_images()
            non_downloaded_images = list(set(all_images_data) - set(downloaded_images))
            for image_data in non_downloaded_images:
                try:
                    image = self.get_image(image_data.attrs['src'])
                    labels = self.labels_in_image(image)
                    for label in labels:
                        if label in DESIRED_LABELS:
                            self.save_image(image, label)
                            downloaded_images.append(image_data)
                except Exception:
                    logging.warning('Error downloading an image')

    def __call__(self, *args, **kwargs):
        """
        1. Open Instagram main page
        2. Log In: user, password, submit
        3. No Instagram notifications
        4. Download images
        5. Hashtag #pets
        """
        self.driver.get('https://instagram.com')
        self.driver.implicitly_wait(2)

        self.driver.find_element_by_xpath('//input[@name=\"username\"]').send_keys(self.user)
        self.driver.find_element_by_xpath('//input[@name=\"password\"]').send_keys(self.password)
        self.driver.find_element_by_xpath('//button[@type=\"submit\"]').click()
        self.driver.implicitly_wait(4)

        self.driver.find_element_by_xpath('//button[contains(text(), "Ahora no")]').click()
        self.driver.implicitly_wait(4)

        self.driver.find_element_by_xpath('//input[@type=\"text\"]').send_keys('#pets')
        time.sleep(3)
        for _ in range(2):
            self.driver.find_element_by_xpath('//input[@type=\"text\"]').send_keys(Keys.ENTER)

        self.download_images()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Process some integers.')
    ap.add_argument('--user', required=True, help='Instagran user name')
    ap.add_argument('--password', required=True, help='Instagram password')
    ap.add_argument('--path_out', required=True, help='Path to save images')
    args = ap.parse_args()

    bot = InstaBot(user=args.user, password=args.password, path_out=args.path_out)
    bot()
