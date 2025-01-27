import os
from collections import deque
import random
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

class ExperienceReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    model.save(file_path)

def load_model(file_path, env):
    """
    Load a trained model from a file.
    """
    return PPO.load(file_path, env=env)

def create_env(data, window_size):
    """
    Creates a trading environment.
    """
    from gym_anytrading.envs import StocksEnv
    env = StocksEnv(df=data, window_size=window_size, frame_bound=(window_size, len(data)))
    return DummyVecEnv([lambda: env])
