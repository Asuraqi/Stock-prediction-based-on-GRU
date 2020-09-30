from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# 打开一个Chrome浏览器
browser = webdriver.Chrome()
# 请求百度首页
browser.get('https://www.baidu.com')
# 找到输入框位置
input = WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="kw"]'))
            )
# 在输入框中输入Python
input.send_keys('Python')
# 找到输入按钮
button = WebDriverWait(browser, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//*[@id="su"]'))
            )
# 点击一次输入按钮
button.click()
# browser.quit()