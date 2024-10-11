import gymnasium as gym
from stable_baselines3 import  PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Used to vectorize environment for A2C/PPO

from default_env import Gym2OpEnv
from grid2op.PlotGrid import PlotMatplot
import matplotlib.pyplot as plt

# Main method for PPO
def train_ppo():
    env = DummyVecEnv([lambda: Gym2OpEnv()])  # Vectorizing the environment for PPO

    # Create the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("ppo_grid2op")

    # Evaluate the agent's performance
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":
    train_ppo()
