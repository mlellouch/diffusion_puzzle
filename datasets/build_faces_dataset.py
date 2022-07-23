from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
import os
import time
import requests
import tqdm


def run(images=256):
    browser = Chrome(r'C:\Program Files\Python39\chromedriver.exe')
    browser.get('https://this-person-does-not-exist.com/en')
    input("Press Enter to continue when page has loaded...")

    def download_image():
        img = browser.find_element(By.ID, 'avatar')
        src = img.get_attribute('src')
        image_name = os.path.basename(src)
        with open(f"./faces/{image_name}", 'wb') as f:
            f.write(requests.get(src).content)

    button = browser.find_element(By.ID, 'reload-button')
    for i in tqdm.tqdm(range(images)):
        button.click()
        time.sleep(7.5)
        download_image()

    browser.close()

if __name__ == '__main__':
    run(8)