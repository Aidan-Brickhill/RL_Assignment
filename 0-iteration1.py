import random
import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt

import os

import grid2op
from grid2op import gym_compat
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.Parameters import Parameters

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self, max_steps):
        super().__init__()

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

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.max_steps = max_steps 
        self.curr_step = 0 

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        '''
        Keep:
        gen_p, gen_q, gen_v, gen_theta: Essential for generator status.
        load_p, load_q, load_v, load_theta: Critical for understanding load demands.
        p_or, q_or, v_or, a_or, theta_or: Important for line operational parameters.
        p_ex, q_ex, v_ex, a_ex, theta_ex: Relevant for export parameters.
        rho: Important for system stability and grid performance.
        line_status: Essential for understanding the operational status of lines.
        target_dispatch, actual_dispatch: Crucial for understanding how much power is being generated and dispatched.
        storage_charge, storage_power_target, storage_power, storage_theta: Necessary for managing storage systems.
        curtailment_mw, curtailment, curtailment_limit: Important for managing generation limits.
        is_alarm_illegal, time_since_last_alarm, last_alarm: Critical for monitoring illegal states.
        total_number_of_alert: Useful for understanding alert dynamics.

        Remove:
        year, month, day, hour_of_day, minute_of_hour, day_of_week: May not be critical unless temporal analysis is necessary.
        n_gen, n_load, n_line, n_sub, n_storage, dim_alerts: Counts can be less useful compared to the actual data.
        topo_vect: Only necessary if topological analysis is relevant.
        timestep_overflow, time_before_cooldown_line, time_before_cooldown_sub, time_next_maintenance, duration_next_maintenance: These may be less critical unless maintenance management is a focus.
        attention_budget, max_step, current_step, delta_time: These could be omitted if not specifically relevant to your model’s objectives.
        '''
          
        obs_space = self._gym_env.observation_space
        # Example: filter observation to only include powerline status
        filtered_obs_space = gym.spaces.Dict({
            "delta_time": obs_space["delta_time"], 
            "current_step": obs_space["current_step"], 
            "gen_p": obs_space["gen_p"], 
            "gen_q": obs_space["gen_q"], 
            "gen_v": obs_space["gen_v"], 
            "gen_margin_up": obs_space["gen_margin_up"], 
            "gen_margin_down": obs_space["gen_margin_down"], 
            "gen_theta": obs_space["gen_theta"], 
            "load_p": obs_space["load_p"], 
            "load_q": obs_space["load_q"], 
            "load_v": obs_space["load_v"], 
            "load_theta": obs_space["load_theta"], 
            "p_or": obs_space["p_or"], 
            "q_or": obs_space["q_or"], 
            "a_or": obs_space["a_or"], 
            "v_or": obs_space["v_or"], 
            "p_ex": obs_space["p_ex"], 
            "q_ex": obs_space["q_ex"], 
            "a_ex": obs_space["a_ex"], 
            "v_ex": obs_space["v_ex"], 
            "rho": obs_space["rho"], 
            "theta_ex": obs_space["theta_ex"], 
            "theta_or": obs_space["theta_or"], 
            "_shunt_bus": obs_space["_shunt_bus"], 
            "_shunt_p": obs_space["_shunt_p"], 
            "_shunt_q": obs_space["_shunt_q"], 
            "_shunt_v": obs_space["_shunt_v"], 
            "curtailment": obs_space["curtailment"], 
            "curtailment_limit_effective": obs_space["curtailment_limit_effective"], 
            "topo_vect": obs_space["topo_vect"], 
            "max_step": obs_space["max_step"], 
            "target_dispatch": obs_space["target_dispatch"], 
            "actual_dispatch": obs_space["actual_dispatch"], 
            "thermal_limit": obs_space["thermal_limit"], 
            "attention_budget": obs_space["attention_budget"], 
        })

        self.observation_space = filtered_obs_space

    def filter_observation(self, obs):
        
        filtered_obs = {key: obs[key] for key in self.observation_space.keys()}

        return filtered_obs

    def setup_actions(self):
        
        action_space = self._gym_env.action_space

        if 'set_bus' in action_space.spaces:
            del action_space.spaces['set_bus']

        if 'set_line_status' in action_space.spaces:
            del action_space.spaces['set_line_status']

        if 'curtail' in action_space.spaces:
            del action_space.spaces['curtail']

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

    def filter_action(self, action, method):

        def first_4(array):
            count = 0  # Counter for non-zero values
            for i in range(len(array)):
                if array[i] != 0:
                    count += 1
                if count > 4:
                    array[i] = 0  # Set remaining values to zero
            return array

        def last_4(array):
            count = 0
            for i in range(len(array) - 1, -1, -1):
                if array[i]!= 0:
                    count += 1
                if count > 4:
                    array[i] = 0  # Set remaining values to zero
            return array

        def random_4(array):
            count = 0
            while count < 4:
                for i in range(len(array)):
                    if array[i] == 0:
                        continue
                    else:
                        array[i] = random.choice([0, 1])  # Set remaining values to random values
                        if array[i] == 1:
                            count += 1 # Only incremement the count if the random number chosen is a 1  
            return array

        action = [int(x) for x in action]

        dynamic_action = np.array(action)

        if method == 'first_4':
            dynamic_action[:57] = first_4(dynamic_action[:57])
            dynamic_action[57:77] = first_4(dynamic_action[57:77])

        if method == 'last_4':
            dynamic_action[:57] = last_4(dynamic_action[:57])
            dynamic_action[57:77] = last_4(dynamic_action[57:77])

        if method == 'random_4':
            dynamic_action[:57] = random_4(dynamic_action[:57])
            dynamic_action[57:77] = random_4(dynamic_action[57:77])
            
        return dynamic_action

    def step(self, action):
        original_action = self.unflatten_action(self.filter_action(action, 'last_4'))
        
        obs, reward, terminated, truncated, info = self._gym_env.step(original_action)

        is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        print(f"\t is action valid = {is_action_valid}")
        if not is_action_valid:
            print(f"\t\t reason = {info['exception']}")

        filtered_obs = self.filter_observation(obs)

        return  filtered_obs, reward, terminated, truncated, info

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

def create_env(max_steps):
    return Monitor(Gym2OpEnv(max_steps))

def train(model_class, model_name, env, total_timesteps=10000):
    print('Training ' + model_name)

    reward_logger = RewardLoggerCallback()
    length_logger = EpisodeLengthLoggerCallback()

    callback_list = CallbackList([reward_logger, length_logger])

    model = model_class("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    model.save(f"baseline/{model_name}")

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

def plot_returns(random_return, ppo_return, a2c_return):
    agents = ['Random', 'PPO', 'A2C']
    r_means = [random_return[0], ppo_return[0], a2c_return[0]]
    r_stds = [random_return[1], ppo_return[1], a2c_return[1]]
    l_means = [random_return[2], ppo_return[2], a2c_return[2]]
    l_stds = [random_return[3], ppo_return[3], a2c_return[3]]

    plt.figure(figsize=(10, 6))
    plt.bar(agents, r_means, yerr=r_stds, capsize=10)
    plt.title('Final Agent Return Comparison')
    plt.ylabel('Mean Return')
    plt.ylim(bottom=0)
    for i, v in enumerate(r_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig('iteration1/plots/agent_r_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(agents, l_means, yerr=l_stds, capsize=10)
    plt.title('Final Agent Length Comparison')
    plt.ylabel('Mean Length')
    plt.ylim(bottom=0)
    for i, v in enumerate(r_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig('iteration1/plots/agent_l_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_return[4], label='PPO', marker='o', linestyle='-')
    plt.plot(a2c_return[4], label='A2C', marker='s', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Comparison of PPO, A2C, and Random Agent')
    plt.legend()
    plt.savefig('iteration1/plots/agent_reward_over_time.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_return[5], label='PPO', marker='o', linestyle='-')
    plt.plot(a2c_return[5], label='A2C', marker='s', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Over Time for PPO, A2C and Random Agents')
    plt.legend()
    plt.savefig('iteration1/plots/episode_length_over_time.png')
    plt.close()

def main():
    max_steps = 200
    env = create_env(max_steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    ppo_reward = 0
    a2c_reward = 0
    # Train PPO
    if not os.path.exists('iteration1/ppo_grid2op.zip'):
        ppo_model, ppo_reward, ppo_length = train(PPO, "ppo_grid2op", vec_env)
    else:
        ppo_model = PPO.load('iteration1/ppo_grid2op.zip', env=env)

    # Train A2C
    if not os.path.exists('iteration1/a2c_grid2op.zip'):
        a2c_model, a2c_reward, a2c_length = train(A2C, "a2c_grid2op", vec_env)
    else:
        a2c_model = A2C.load('iteration1/a2c_grid2op.zip', env=env)
    
    # Evaluate PPO
    ppo_r_mean, ppo_r_std, ppo_l_mean, ppo_l_std = evaluate(env, ppo_model)

    # Evaluate A2C
    a2c_r_mean, a2c_r_std, a2c_l_mean, a2c_l_std = evaluate(env, a2c_model)

    # Evaluate Random
    random_r_mean, random_r_std, random_l_mean, random_l_std = evaluate(env, None, random_agent=True)

    # Plot returns
    plot_returns((random_r_mean, random_r_std, random_l_mean, random_l_std), (ppo_r_mean, ppo_r_std, ppo_l_mean, ppo_l_std, ppo_reward, ppo_length), (a2c_r_mean, a2c_r_std, a2c_l_mean, a2c_l_std, a2c_reward, a2c_length))

if __name__ == "__main__":
    main()