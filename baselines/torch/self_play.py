import gym
import random

from stable_baselines3 import PPO

env = gym.make("PongNoFrameskip-v4") 

model = PPO('MlpPolicy', env)

# 重复以下训练循环
for i_episode in range(1000):
  
  # 创建两个agent实例对抗
  player1 = model.policy 
  player2 = model.policy

  while not done:
    
    # player1动作
    # give action
    action1 = player1(observation1) 
    # action to env, game server will calculate and feedback
    observation1, reward1, done, info = env.step(action1)
    
    if done:
      break
      
    # player2动作    
    # give action
    action2 = player2(observation2)
    # action to env, game server will calculate and feedback
    observation2, reward2, done, info = env.step(action2)

  # 记录数据    
  model.memory.add(observation1, action1, reward1, observation1, done)
  model.memory.add(observation2, action2, reward2, observation2, done)

  # 使用经验回放更新模型  
  model.train()