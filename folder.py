import os 

for i in range(1,100000000):
    os.makedirs("./wings_tensorboard/DQN_"+str(i), exist_ok=True)