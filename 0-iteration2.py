import copy
import csv
from typing import Dict, Literal, Any
import json

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box

import numpy as np

import matplotlib.pyplot as plt

import os

import grid2op 
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

# AlertReward, BaseReward
from grid2op.Reward import Reward, BridgeReward, CloseToOverflowReward, EconomicReward, EpisodeDurationReward
from grid2op.Reward import FlatReward, GameplayReward, DistanceReward, CombinedReward, ConstantReward, 
from grid2op.Reward import AlarmReward, IncreasingFlatReward, LinesReconnectedReward, LinesCapacityReward, RedispReward

from grid2op.Parameters import Parameters

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

from typing import Dict, Literal, Any

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self,
                 env_config: Dict[Literal["obs_attr_to_keep",
                                          "act_type",
                                          "act_attr_to_keep"],
                                  Any] = None):
        super().__init__()
        if env_config is None:
            env_config = {}

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Combines L2RPN and N1 rewards

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Create Grid2Op environment with specified parameters
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = GymEnv(self._g2op_env)

        self.max_steps = max_steps 
        self.curr_step = 0 

        self.setup_observations(env_config)
        self.setup_actions(env_config)

    def setup_observations(self, env_config):

        obs_attr_to_keep = copy.deepcopy(env_config["obs_attr_to_keep"])
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space, attr_to_keep=obs_attr_to_keep)
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,  low=self._gym_env.observation_space.low, high=self._gym_env.observation_space.high)
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self, env_config):

        # customize the action space
        act_type = env_config["act_type"]
        self._gym_env.action_space.close()

        if act_type == "discrete":            
            act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space, attr_to_keep=act_attr_to_keep)
            self.action_space = Discrete(self._gym_env.action_space.n)
        
        elif act_type == "box":
            act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space, attr_to_keep=act_attr_to_keep)
            self.action_space = Box(shape=self._gym_env.action_space.shape, low=self._gym_env.action_space.low, high=self._gym_env.action_space.high)
        
        elif act_type == "multi_discrete":
            act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, attr_to_keep=act_attr_to_keep)
            self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)

    def step(self, action):
        return self._gym_env.step(action)

    def reset(self, seed=None, options=None): 
        self.curr_step = 0 
        return self._gym_env.reset(seed=seed, options=options)

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
    
class EpisodeLengthLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeLengthLoggerCallback, self).__init__(verbose)
        self.episode_lengths = []
        self.current_length = 0

    def _on_step(self) -> bool:
        # Increment the step count
        self.current_length += 1

        # Check if the episode is done, then log the episode length
        done = self.locals['dones']
        if done:
            self.episode_lengths.append(self.current_length)
            self.current_length = 0  # Reset for the next episode
            if self.verbose > 0:
                print(f"Episode length: {self.episode_lengths[-1]}")
        return True

    def get_lengths(self):
        return self.episode_lengths

def create_env(max_steps, iteration):

    if iteration == 'Space1':
        act_attr_to_keep = ['change_bus', 'change_line_status', 'curtail', 'redispatch', 'set_bus', 'set_line_status', 'set_line_status_simple', 'set_storage']

        obs_attr_to_keep = ['a_ex', 'a_or', 'actual_dispatch', 'attention_budget', 'current_step', 'curtailment', 'curtailment_limit_effective', 
                            'delta_time', 'gen_margin_down', 'gen_margin_up', 'gen_p', 'gen_q', 'gen_theta', 'gen_v', 'load_p', 'load_q', 'load_theta', 
                            'load_v', 'max_step', 'p_ex', 'p_or', 'q_ex', 'q_or', 'rho', 'target_dispatch', 'thermal_limit', 'theta_ex', 'theta_or', 
                            'topo_vect', 'v_ex', 'v_or'
                        ]

    if iteration == 'Space2':
        act_attr_to_keep = ['change_bus', 'change_line_status', 'curtail', 'redispatch']

        obs_attr_to_keep = ['a_ex' ,'a_or' ,'active_alert' ,'actual_dispatch' ,'alert_duration' ,'attack_under_alert' ,'attention_budget' ,'was_alert_used_after_attack'
	                   ,'current_step' ,'curtailment' ,'curtailment_limit' ,'curtailment_limit_effective' ,'curtailment_limit_mw' ,'curtailment_mw'
	                   ,'day' ,'day_of_week' ,'delta_time' ,'duration_next_maintenance' ,'gen_margin_down' ,'gen_margin_up' ,'gen_p' ,'gen_p_before_curtail'
	                   ,'gen_q' ,'gen_theta' ,'gen_v' ,'hour_of_day' ,'is_alarm_illegal' ,'last_alarm' ,'line_status' ,'load_p', 'load_q' ,'load_theta'
	                   ,'load_v' ,'max_step' ,'minute_of_hour' ,'month' ,'p_ex' ,'p_or' ,'prod_p' ,'prod_q' ,'prod_v','q_ex' ,'q_or' ,'rho' ,'storage_charge'
	                   ,'storage_power' ,'storage_power_target' ,'storage_theta' ,'target_dispatch' ,'thermal_limit' ,'theta_ex' ,'theta_or' ,'year'
	                   ,'time_before_cooldown_line' ,'time_before_cooldown_sub' ,'time_next_maintenance' ,'time_since_last_alarm' ,'time_since_last_alert'
	                   ,'time_since_last_attack' ,'timestep_overflow' ,'topo_vect' ,'total_number_of_alert' ,'v_ex' ,'v_or' ,'was_alarm_used_after_game_over'
                    ]
        
    if iteration == 'Space3':
        act_attr_to_keep = ['curtail', 'redispatch', 'set_bus', 'set_line_status']

        obs_attr_to_keep = ['a_ex' ,'a_or' ,'active_alert' ,'actual_dispatch' ,'alert_duration' ,'attack_under_alert' ,'attention_budget' ,'was_alert_used_after_attack'
	                   ,'current_step' ,'curtailment' ,'curtailment_limit' ,'curtailment_limit_effective' ,'curtailment_limit_mw' ,'curtailment_mw'
	                   ,'day' ,'day_of_week' ,'delta_time' ,'duration_next_maintenance' ,'gen_margin_down' ,'gen_margin_up' ,'gen_p' ,'gen_p_before_curtail'
	                   ,'gen_q' ,'gen_theta' ,'gen_v' ,'hour_of_day' ,'is_alarm_illegal' ,'last_alarm' ,'line_status' ,'load_p', 'load_q' ,'load_theta'
	                   ,'load_v' ,'max_step' ,'minute_of_hour' ,'month' ,'p_ex' ,'p_or' ,'prod_p' ,'prod_q' ,'prod_v','q_ex' ,'q_or' ,'rho' ,'storage_charge'
	                   ,'storage_power' ,'storage_power_target' ,'storage_theta' ,'target_dispatch' ,'thermal_limit' ,'theta_ex' ,'theta_or' ,'year'
	                   ,'time_before_cooldown_line' ,'time_before_cooldown_sub' ,'time_next_maintenance' ,'time_since_last_alarm' ,'time_since_last_alert'
	                   ,'time_since_last_attack' ,'timestep_overflow' ,'topo_vect' ,'total_number_of_alert' ,'v_ex' ,'v_or' ,'was_alarm_used_after_game_over'
                    ]

    if iteration == 'Space4':
        act_attr_to_keep = ['change_bus', 'change_line_status', 'curtail', 'redispatch']

        obs_attr_to_keep = ['a_ex', 'a_or', 'actual_dispatch', 'attention_budget', 'current_step', 'curtailment', 'curtailment_limit_effective', 
                            'delta_time', 'gen_margin_down', 'gen_margin_up', 'gen_p', 'gen_q', 'gen_theta', 'gen_v', 'load_p', 'load_q', 'load_theta', 
                            'load_v', 'max_step', 'p_ex', 'p_or', 'q_ex', 'q_or', 'rho', 'target_dispatch', 'thermal_limit', 'theta_ex', 'theta_or', 
                            'topo_vect', 'v_ex', 'v_or'
                        ]
    
    if iteration == 'Space5':
        act_attr_to_keep = ['curtail', 'redispatch', 'set_bus', 'set_line_status']

        obs_attr_to_keep = ['a_ex', 'a_or', 'actual_dispatch', 'attention_budget', 'current_step', 'curtailment', 'curtailment_limit_effective', 
                            'delta_time', 'gen_margin_down', 'gen_margin_up', 'gen_p', 'gen_q', 'gen_theta', 'gen_v', 'load_p', 'load_q', 'load_theta', 
                            'load_v', 'max_step', 'p_ex', 'p_or', 'q_ex', 'q_or', 'rho', 'target_dispatch', 'thermal_limit', 'theta_ex', 'theta_or', 
                            'topo_vect', 'v_ex', 'v_or'
                        ]

    env_config = {
        "obs_attr_to_keep": obs_attr_to_keep,
        "act_type": "discrete",
        "act_attr_to_keep": act_attr_to_keep
    }

    return Monitor(Gym2OpEnv(max_steps, env_config=env_config))

def train(model_class, model_name, env, iteration, total_timesteps=102400):
    print('Training ' + model_name)

    reward_logger = RewardLoggerCallback()
    length_logger = EpisodeLengthLoggerCallback()

    callback_list = CallbackList([reward_logger, length_logger])

    model = model_class("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    model.save(f'iteration1/{iteration}/{model_name}')

    print('Completed Training ' + model_name)
    
    # Plotting rewards and Episode lengths after training
    rewards = reward_logger.get_rewards()
    episode_lengths = length_logger.get_lengths()

    return model, rewards, episode_lengths

def evaluate(env, model, n_episodes=10, random_agent=False):
    
    print('Evaluating agent')

    rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        episode_reward = 0
        steps = 0
        done = False
        obs = env.reset()[0]
        while not done:
            steps += 1
            if (random_agent):
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)
        episode_lengths.append(steps)
    
    mean_r_reward = np.mean(rewards)
    std_r_reward = np.std(rewards)
    mean_l_reward = np.mean(episode_lengths)
    std_l_reward = np.std(episode_lengths)

    print('Completed evaluating agent')

    return mean_r_reward, std_r_reward, mean_l_reward, std_l_reward

def plot_returns(random_return, ppo_return, a2c_return, iteration):
    agents = ['Random', 'PPO', 'A2C']
    r_means = [random_return[0], ppo_return[0], a2c_return[0]]
    r_stds = [random_return[1], ppo_return[1], a2c_return[1]]
    l_means = [random_return[2], ppo_return[2], a2c_return[2]]
    l_stds = [random_return[3], ppo_return[3], a2c_return[3]]

    with open(f'iteration1/data/{iteration}/agent_comparison.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Agent', 'Mean Return', 'Return Std Dev', 'Mean Length', 'Length Std Dev'])
        for i in range(len(agents)):
            writer.writerow([agents[i], r_means[i], r_stds[i], l_means[i], l_stds[i]])

    # Write PPO rewards
    with open(f'iteration1/data/{iteration}/episode_rewards_ppo.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'PPO Reward'])
        for i in range(len(ppo_return[4])):
            writer.writerow([i, ppo_return[4][i]])

    # Write A2C rewards
    with open(f'iteration1/data/{iteration}/episode_rewards_a2c.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'A2C Reward'])
        for i in range(len(a2c_return[4])):
            writer.writerow([i, a2c_return[4][i]])

        # Write PPO rewards
    with open(f'iteration1/data/{iteration}/episode_lengths_ppo.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'PPO Length'])
        for i in range(len(ppo_return[5])):
            writer.writerow([i, ppo_return[5][i]])

    # Write A2C rewards
    with open(f'iteration1/data/{iteration}/episode_lengths_a2c.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'A2C Length'])
        for i in range(len(a2c_return[5])):
            writer.writerow([i, a2c_return[5][i]])

    plt.figure(figsize=(10, 6))
    plt.bar(agents, r_means, yerr=r_stds, capsize=10)
    plt.title('Final Agent Return Comparison')
    plt.ylabel('Mean Return')
    plt.ylim(bottom=0)
    for i, v in enumerate(r_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'iteration1/plots/{iteration}/agent_r_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(agents, l_means, yerr=l_stds, capsize=10)
    plt.title('Final Agent Length Comparison')
    plt.ylabel('Mean Length')
    plt.ylim(bottom=0)
    for i, v in enumerate(l_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'iteration1/plots/{iteration}/agent_l_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_return[4], label='PPO', marker='o', linestyle='-')
    plt.plot(a2c_return[4], label='A2C', marker='s', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Comparison of PPO, A2C, and Random Agent')
    plt.legend()
    plt.savefig(f'iteration1/plots/{iteration}/agent_reward_over_time.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_return[5], label='PPO', marker='o', linestyle='-')
    plt.plot(a2c_return[5], label='A2C', marker='s', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Over Time for PPO, A2C and Random Agents')
    plt.legend()
    plt.savefig(f'iteration1/plots/{iteration}/episode_length_over_time.png')
    plt.close()

def main():
    iterations = ['Space1','Space2','Space3','Space4','Space5']
    for iteration in iterations:

        max_steps = 1000
        env = create_env(max_steps,iteration)

        ppo_reward = 0
        a2c_reward = 0

        # Train PPO
        if not os.path.exists(f'iteration1/{iteration}/ppo_grid2op.zip'):
            ppo_model, ppo_reward, ppo_length = train(PPO, "ppo_grid2op", env, iteration)
        else:
            ppo_model = PPO.load(f'iteration1/{iteration}/ppo_grid2op.zip', env=env)

        # Train A2C
        if not os.path.exists(f'iteration1/{iteration}/a2c_grid2op.zip'):
            a2c_model, a2c_reward, a2c_length = train(A2C, "a2c_grid2op", env, iteration)
        else:
            a2c_model = A2C.load(f'iteration1/{iteration}/a2c_grid2op.zip', env=env)
        
        # Evaluate PPO
        ppo_r_mean, ppo_r_std, ppo_l_mean, ppo_l_std = evaluate(env, ppo_model)

        # Evaluate A2C
        a2c_r_mean, a2c_r_std, a2c_l_mean, a2c_l_std = evaluate(env, a2c_model)

        # Evaluate Random
        random_r_mean, random_r_std, random_l_mean, random_l_std = evaluate(env, None, random_agent=True)

        # Plot returns
        plot_returns((random_r_mean, random_r_std, random_l_mean, random_l_std), (ppo_r_mean, ppo_r_std, ppo_l_mean, ppo_l_std, ppo_reward, ppo_length), (a2c_r_mean, a2c_r_std, a2c_l_mean, a2c_l_std, a2c_reward, a2c_length), iteration)

if __name__ == "__main__":
    main()