import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager

def get_first_google_result(url):
    match = re.search(r'https://(.*)\.', url)
    if match:
        context = match.group(1)
    else:
        context = ''

    # Form search query
    query = f"{context} terms and conditions"

    options = Options()


    # Configure Chrome options
    options.add_argument("--headless")  
    options.add_argument("--disable-gpu") 
    options.add_argument("--window-size=1920,1080") 

    # Initialize WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    # driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

    
    try:
        # Navigate to Google
        driver.get("https://www.google.com")

        # Find the search box, enter the query, and submit
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)

        # Wait for results to load
        driver.implicitly_wait(5)

        # Extract the first result link
        first_element_locator = (By.CSS_SELECTOR, 'div.yuRUbf a')
        wait = WebDriverWait(driver, 20)
        wait.until(EC.visibility_of_element_located(first_element_locator))
        result_element = driver.find_element(*first_element_locator).get_attribute('href')
        return result_element
    finally:
        driver.quit()
