import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import gym.spaces
import gym.utils.seeding
from stable_baselines3 import DQN
import torch.nn.functional as F
import time
import math
#import pyautogui


class WingsEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4) # 定義動作空間
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(855, 1472, 3), dtype=np.uint8) # 定義觀察空間
        self.reward_range = (-float('inf'), 0.0) # 定義獎勵範圍
        self.driver = webdriver.Edge() # 啟動瀏覽器
        self.driver.set_window_size(800, 600)  
        self.driver.get('https://wings.io/') # 訪問遊戲網站
        self.url = 'https://wings.io/'
        

    def _get_screenshot(self):
        screenshot = self.driver.get_screenshot_as_png()
        return screenshot

    def _get_obs(self):
        screenshot = self._get_screenshot()
        obs = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        obs = cv2.resize(obs, (1472, 855))##修改大小  
             
        return obs
        
    
    
    
    def reset(self):
        ##self.driver.get(self.url) # 加載遊戲頁面
        start_button = self.driver.find_element(By.ID, 'playButton') # 獲取開始按鈕
        ActionChains(self.driver).click(start_button).perform() # 模擬按下開始按鈕
        obs = self._get_obs() # 獲取初始觀察值
        return obs



    def step(self, action):
        print("yee")
        direction = 0
        distance = 100

        # 獲取 canvas元素
        canvas_element = self.driver.find_element(By.ID, 'canvas')

        # 獲取 canvas標籤元素的大小
        canvas_width = canvas_element.size['width']
        canvas_height = canvas_element.size['height']

        
        
        
        ##print(direction,distance,target_x, target_y,current_x, current_y)
        # 創建 ActionChains 對象
        actions = ActionChains(self.driver)

        # 獲取當前滑鼠的座標
        #current_x, current_y = self.driver.execute_script("return [window.scrollX, window.scrollY];")
        
        # 移動滑鼠到目標座標
        # 計算滑鼠移動的方向和距離
        if 1 <= action :
                
            direction = (action - 1) * 90 * (math.pi / 2)
            # 計算滑鼠移動的目標座標
            x = distance * math.cos(direction)+canvas_width*0.5
            y = distance * math.sin(direction)+canvas_height*0.5
            print(action,x,y,canvas_width,canvas_height)
            
            actions.move_to_element_with_offset(self.driver.find_element(By.TAG_NAME, 'canvas'),x,y).perform()
            
            print("yee")
        else:
            actions.click().perform()

        # 添加圖標以顯示模擬滑鼠的當前位置
        #self.driver.execute_script(f"var cursorElement = document.getElementById('cursorElement'); if (!cursorElement) {{ var img = document.createElement('img'); img.id = 'cursorElement'; img.src = 'cursor.gif'; img.style.position = 'absolute'; img.style.zIndex = 9999; document.body.appendChild(img); }} cursorElement = document.getElementById('cursorElement'); cursorElement.style.left = '{x}px'; cursorElement.style.top = '{y}px';")

        # 暫停一段時間，限制滑鼠移動速度
        time.sleep(0.1)

        # 計獎勵值
        reward = 0

        # 計算終止標誌
        done = False

        # 其他資訊
        info = {}

        # 返回當前狀態、獎勵和終止標誌
        obs = self._get_obs()
        return obs, reward, done, info





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 116 * 156, 256)
        self.fc2 = nn.Linear(256, 3)

        # 初始化權重和偏差
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    

    def forward(self, x): 
        x = F.relu(self.conv1(x)) 
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv2(x)) 
        x = F.max_pool2d(x, 2) 
        x = x.view(-1, 32 * 48 * 48)        
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x




env = WingsEnv()


if torch.cuda.is_available():
    print('GPU is available!')
    # 將模型移動到GPU上
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001, buffer_size=1000,exploration_fraction=0.1, exploration_final_eps=0.02).to('cuda')
else:
    print('GPU is not available!')
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001, buffer_size=1000, exploration_fraction=0.1, exploration_final_eps=0.02)

model.learn(total_timesteps=10000, log_interval=1000)
