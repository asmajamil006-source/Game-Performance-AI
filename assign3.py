# Subhan Ahmed (2023F-BSE-031)
"""Selenium assignment: fill in the TutorialsPoint practice form and save a screenshot.

This script is independent of the AI application (app.py) and is not imported by it.
It needs Selenium plus Google Chrome and an internet connection:

    pip install -r requirements-selenium.txt
    python assign3.py [path/to/picture.jpg]

If no picture is given (or the file does not exist) the picture-upload step is skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://www.tutorialspoint.com/selenium/practice/selenium_automation_practice.php"
SCREENSHOT = Path(__file__).resolve().parent / "Filled_Form.png"
DEFAULT_PICTURE = Path(__file__).resolve().parent / "sample_picture.jpg"   # optional, any .jpg


def click_with_js(driver, element) -> None:
    driver.execute_script("arguments[0].click();", element)


def fill_form(driver, picture: Path | None) -> None:
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.NAME, "name"))).send_keys("Muhammad")
    driver.find_element(By.NAME, "email").send_keys("talha@example.com")

    click_with_js(driver, wait.until(EC.presence_of_element_located((By.ID, "gender"))))

    driver.find_element(By.NAME, "mobile").send_keys("03123456789")

    dob = driver.find_element(By.NAME, "dob")
    dob.clear()
    dob.send_keys("14/05/2005")

    driver.find_element(By.NAME, "subjects").send_keys("Software Testing")

    click_with_js(driver, driver.find_element(By.XPATH, "(//input[@type='checkbox'])[1]"))

    if picture is not None and picture.is_file():
        driver.find_element(By.NAME, "picture").send_keys(str(picture))
    else:
        print("No picture file found; skipping the picture upload step.")

    # 'Currend Address' is the site's own (misspelled) placeholder text.
    driver.find_element(By.XPATH, "//textarea[@placeholder='Currend Address']").send_keys("Karachi Pakistan")

    Select(driver.find_element(By.XPATH, "//select[@id='state']")).select_by_visible_text("Uttar Pradesh")
    Select(driver.find_element(By.XPATH, "//select[@id='city']")).select_by_visible_text("Agra")

    driver.find_element(By.XPATH, "//input[@type='submit']").click()


def main() -> None:
    picture = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PICTURE
    driver = webdriver.Chrome()
    try:
        driver.maximize_window()
        driver.get(URL)
        fill_form(driver, picture)
        WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")
        driver.save_screenshot(str(SCREENSHOT))
        print(f"Screenshot saved to {SCREENSHOT}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
