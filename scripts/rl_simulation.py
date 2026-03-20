
 import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class UPIEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)  # -3 to +3 server change
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        self.current_servers = 8000
        self.td = 3.8
        self.bd = 0.8
        self.step_count = 0
    
    def reset(self, seed=None):
        self.current_servers = 8000
        self.td = 3.8
        self.bd = 0.8
        self.step_count = 0
        return np.zeros(12), {}
    
    def step(self, action):
        delta = action - 3
        self.current_servers = max(5000, min(15000, self.current_servers + delta * 200))
        
        # Simulate improvement
        self.td = max(1.5, self.td - 0.05)
        self.bd = max(0.3, self.bd - 0.01)
        
        reward = 10 * (self.td <= 2.5) + 8 * (self.bd <= 0.8) - 0.8*self.td - 25*self.bd - 0.015*self.current_servers
        
        self.step_count += 1
        done = self.step_count >= 150  # one day
        return np.zeros(12), reward, done, False, {}

# Train / evaluate
def run_rl():
    env = make_vec_env(UPIEnv, n_envs=4)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./rl_logs/")
    model.learn(total_timesteps=500_000)
    
    # Evaluate
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(200):  # 200 simulated days
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward.mean()
        if done.any(): obs, _ = env.reset()
    
    print(f"Average episodic reward (200 simulated days): {total_reward/200:.2f}")
    model.save("results/aoof_ppo_model.zip")

if __name__ == "__main__":
    run_rl()
