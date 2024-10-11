import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt

import os

import grid2op
from grid2op import gym_compat
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self, max_steps):
        super().__init__()

        backend = LightSimBackend()
        env_name = "l2rpn_case14_sandbox"
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Combines L2RPN and N1 rewards

        # Create Grid2Op environment with specified parameters
        self.g2op_env = grid2op.make(
            env_name, backend=backend, action_class=action_class, 
            observation_class=observation_class, reward_class=reward_class
        )

        ##########
        # REWARD #
        ##########
        cr = self.g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self.g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self.g2op_env)

        self.max_steps = max_steps 
        self.curr_step = 0 

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self):
        low = []
        high = []
        for key, space in self._gym_env.action_space.spaces.items():
            if isinstance(space, gym.spaces.MultiBinary):
                low.extend([0] * space.n)
                high.extend([1] * space.n)
            elif isinstance(space, gym.spaces.Box):
                low.extend(space.low.tolist())
                high.extend(space.high.tolist())
            else:
                raise NotImplementedError(f"Unsupported action space type: {type(space)}")

        self.action_space =  gym.spaces.Box(np.array(low), np.array(high), dtype=np.int32)
        # self.action_space =  gym.spaces.flatten_space(self._gym_env.action_space)

    def step(self, action):
        original_action = self.unflatten_action(action)
        self.curr_step += 1 
        obs, reward, terminated, truncated, _ = self._gym_env.step(original_action)
        
        if self.curr_step >= self.max_steps:
            terminated = True

        return obs, reward, terminated, truncated, _

    def unflatten_action(self, action):
        original_action = {}
        idx = 0
        for key, space in self._gym_env.action_space.spaces.items():
            if isinstance(space, gym.spaces.MultiBinary):
                size = space.n
                original_action[key] = action[idx:idx + size]
                idx += size
            elif isinstance(space, gym.spaces.Box):
                size = space.shape[0]
                original_action[key] = action[idx:idx + size]
                idx += size
        return original_action

    def reset(self, seed=None):  # Add seed argument here
        self.curr_step = 0 
        return self._gym_env.reset(seed=seed)

    def render(self, mode="human"):
        return self._gym_env.render(mode=mode)

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        # Get the reward for the current step
        reward = self.locals['rewards'][0]
        self.current_rewards.append(reward)

        # Check if the episode is done, then log the reward
        done = self.locals['dones']
        if done:
            episode_reward = np.sum(self.current_rewards)
            self.episode_rewards.append(episode_reward)
            self.current_rewards = []
            if self.verbose > 0:
                print(f"Episode reward: {episode_reward}")
        return True

    def get_rewards(self):
        return self.episode_rewards

def create_env(max_steps):
    return Monitor(Gym2OpEnv(max_steps))

def train(model_class, model_name, env, total_timesteps=25000):
    print('Training ' + model_name)

    reward_logger = RewardLoggerCallback()

    model = model_class("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=reward_logger)
    model.save(f"baseline/{model_name}")

    print('Completed Training ' + model_name)
    
    # Plotting rewards after training
    rewards = reward_logger.get_rewards()

    return model, rewards

def evaluate(env, model, n_episodes=10, random_agent=False):
    
    print('Evaluating agent')

    rewards = []
    for _ in range(n_episodes):
        episode_reward = 0
        done = False
        obs = env.reset()[0]
        while not done:
            if (random_agent):
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print('Completed evaluating agent')

    return mean_reward, std_reward

def plot_returns(random_return, ppo_return, a2c_return):
    agents = ['Random', 'PPO', 'A2C']
    means = [random_return[0], ppo_return[0], a2c_return[0]]
    stds = [random_return[1], ppo_return[1], a2c_return[1]]

    plt.figure(figsize=(10, 6))
    plt.bar(agents, means, yerr=stds, capsize=10)
    plt.title('Agent Performance Comparison')
    plt.ylabel('Mean Return')
    plt.ylim(bottom=0)
    for i, v in enumerate(means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig('plots/agent_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_return[2], label='PPO', marker='o', linestyle='-')
    plt.plot(a2c_return[2], label='A2C', marker='s', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Comparison of PPO, A2C, and Random Agent')
    plt.legend()
    plt.savefig('plots/agent_reward.png')
    plt.close()

def main():
    max_steps = 100
    env = create_env(max_steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Train PPO
    if not os.path.exists('baseline/ppo_grid2op.zip'):
        ppo_model, ppo_reward = train(PPO, "ppo_grid2op", vec_env)
    else:
        ppo_model = PPO.load('baseline/ppo_grid2op.zip', env=env)

    # Train A2C
    if not os.path.exists('baseline/a2c_grid2op.zip'):
        a2c_model, a2c_reward = train(A2C, "a2c_grid2op", vec_env)
    else:
        a2c_model = A2C.load('baseline/a2c_grid2op.zip', env=env)
    
    # Evaluate PPO
    ppo_mean, ppo_std = evaluate(env, ppo_model)

    # Evaluate A2C
    a2c_mean, a2c_std = evaluate(env, a2c_model)

    # Evaluate Random
    random_mean, random_std = evaluate(env, None, random_agent=True)

    # Plot returns
    plot_returns((random_mean, random_std), (ppo_mean, ppo_std, ppo_reward), (a2c_mean, a2c_std, a2c_reward))

if __name__ == "__main__":
    main()