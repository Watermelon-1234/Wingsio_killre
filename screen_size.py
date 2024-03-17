from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains

# 創建 WebDriver
driver = webdriver.Edge()

# 打開網頁
driver.get("https://wings.io/")

# 等待網頁載入完成

# 獲取可見區域的大小
viewport_width = driver.execute_script("return document.documentElement.clientWidth")
viewport_height = driver.execute_script("return document.documentElement.clientHeight")

# 獲取滑鼠目前位置
mouse_x = driver.execute_script("return window.pageXOffset + window.innerWidth")
mouse_y = driver.execute_script("return window.pageYOffset + window.innerHeight")

# 移動滑鼠到最右邊
ActionChains(driver).move_to_element_with_offset(driver.find_element(By.TAG_NAME, 'cancas'), viewport_width - mouse_x, 0).perform()

# 獲取滑鼠可移動的最大範圍
max_mouse_x = driver.execute_script("return window.pageXOffset + window.innerWidth")
max_mouse_y = driver.execute_script("return window.pageYOffset + window.innerHeight")

# 顯示滑鼠可移動的最大範圍
print("Max Mouse X:", max_mouse_x)
print("Max Mouse Y:", max_mouse_y)

# 關閉瀏覽器視窗
driver.quit()