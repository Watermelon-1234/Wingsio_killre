import math
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from bs4 import BeautifulSoup
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
height,width = 480,640
# 定義遊戲環境類別
class WingsEnv(gym.Env):
    def __init__(self):
        super(WingsEnv, self).__init__()
        
        self.action_space = spaces.Discrete(4)  # 動作空間：0表示攻擊，1-4:0到360-90
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640 ,3), dtype=np.uint8)  # 觀察空間：遊戲畫面的像素值
        self.driver = webdriver.Edge()  # 初始化瀏覽器驅動程式
        self.driver.set_window_size(width, height)
        self.driver.get("https://wings.io/")  # 訪問遊戲網站
        time.sleep(1)  # 等待遊戲載入

    def _get_obs(self):
        # Capture the game screen and convert it to a NumPy array
        screenshot = self.driver.get_screenshot_as_png()
        obs = np.frombuffer(screenshot, dtype=np.uint8)
        '''
        # Decode the image using OpenCV
        img = cv2.imdecode(obs, cv2.IMREAD_COLOR)

        # Resize the image while maintaining the aspect ratio
        target_width, target_height = 640, 480
        aspect_ratio = img.shape[1] / img.shape[0]
        if aspect_ratio > 1:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_width = int(target_height * aspect_ratio)
            new_height = target_height

        resized_img = cv2.resize(img, (new_width, new_height))

        # Pad the resized image to match the target shape
        padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
        '''
        print(type(screenshot))
        return screenshot

    def reset(self):
        # 重新載入遊戲
        self.driver.refresh()
        time.sleep(3)  # 等待遊戲載入
        start_button = self.driver.find_element(By.ID, 'playButton') # 獲取開始按鈕
        ActionChains(self.driver).click(start_button).perform() # 模擬按下開始按鈕
        time.sleep(1)
        # 返回初始觀察值
        return self._get_obs()

    def step(self, action):
        #direction = 0
        #distance = 50

        # 獲取 canvas元素
        canvas_element = self.driver.find_element(by=By.TAG_NAME, value = 'canvas')

        # 獲取 canvas標籤座標位置
        location = canvas_element.location

        # 模擬按鍵動作 還沒寫好
        if action == 0:  # 攻擊
            ActionChains(self.driver).move_to_element_with_offset(canvas_element, location['x'] + 500, location['y'] + 400).click().perform()
        elif action == 1:  # 左轉
            direction = 180
            ActionChains(self.driver).move_to_element_with_offset(canvas_element, location['x'] + 500, location['y'] + 400).click().perform()
        elif action == 2:  # 右轉
            direction = 0
            ActionChains(self.driver).move_to_element_with_offset(canvas_element, location['x'] + 500, location['y'] + 400).click().perform()
        elif action == 3:  # 前進
            ActionChains(self.driver).move_to_element_with_offset(canvas_element, location['x'] + 500, location['y'] + 400).click().perform()

        time.sleep(0.2)  # 等待遊戲狀態更新

        # 檢查遊戲是否結束
        game_over = self._is_game_over()

        # 返回觀察值、獎勵、遊戲結束標誌和額外資訊
        obs = self._get_obs()
        reward = 0  # TODO: 根據遊戲狀態計算獎勵
        done = game_over
        info = {}  # 可以添加額外的訓練資訊

        return obs, reward, done, info

    def _is_game_over(self):
        # 檢查遊戲是否結束
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        game_over_element = soup.find('div', {'id': 'gameOver'})  # 檢查遊戲結束的元素
        return game_over_element is not None

    def close(self):
        # 關閉瀏覽器驅動程式
        self.driver.quit()


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定義策略網路模型
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    
if __name__ == "__main__":
    env = WingsEnv()

    input_shape = env.observation_space.shape
    
    num_actions = env.action_space.n
    #model = PolicyNetwork(input_shape, num_actions)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    

    # 進行訓練
    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward = 0
        '''
        while not done:
            # 將觀察值轉換為 PyTorch 張量
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).permute(0, 3, 1, 2).float()
            print(obs_tensor)
            # 在模型上運行觀察值並獲取動作概率分佈
            action_probs = model(obs_tensor)

            # 從動作概率分佈中抽樣獲取動作
            m = Categorical(action_probs)
            action = m.sample().item()

            # 在環境中執行動作，並獲取下一個觀察值、獎勵和遊戲結束標誌
            next_obs, reward, done, _ = env.step(action)

            # 計算總獎勵
            total_reward += reward

            # 計算損失
            loss = -m.log_prob(torch.tensor(action)).unsqueeze(0) * reward

            # 清除梯度
            optimizer.zero_grad()

            # 執行反向傳播
            loss.backward()

            # 更新模型參數
            optimizer.step()

            # 更新觀察值
            obs = next_obs

        print("Episode:", episode, "Total Reward:", total_reward)
    '''
    env.close()
