from typing import OrderedDict
import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self):
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

        self.action_space = gym.spaces.flatten_space(action_space)

    def filter_action(self, float_action):
        dynamic_action = np.array(float_action) 
        dynamic_action[:77] = np.round(dynamic_action[:77]).astype(int)

        # TODO: 
        # 4. In altering the action space, is it valid to somehow manipulate the actions where all actions are valid? YES - multiple methods, 1st 4, random 4, last 4, etc 

        return dynamic_action

    def step(self, action):
        original_action = self.unflatten_action(self.filter_action(action))
        
        obs, reward, terminated, truncated, _ = self._gym_env.step(original_action)

        filtered_obs = self.filter_observation(obs)

        return  filtered_obs, reward, terminated, truncated, _

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

def create_env():
    return Monitor(Gym2OpEnv())

def train_and_evaluate(model_class, model_name, env, total_timesteps=5000):
    model = model_class("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{model_name}")
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"{model_name} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, mean_reward, std_reward

def random_agent(env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        episode_reward = 0
        done = False
        obs = env.filter_observation(env.reset()[0])
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Random Agent - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
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
    plt.savefig('agent_comparison.png')
    plt.close()

def main():
    env = create_env()
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Random Agent
    random_return = random_agent(env)

    # PPO Agent
    ppo_model, ppo_mean, ppo_std = train_and_evaluate(PPO, "ppo_grid2op", vec_env)

    # A2C Agent
    a2c_model, a2c_mean, a2c_std = train_and_evaluate(A2C, "a2c_grid2op", vec_env)

    # Plot returns
    plot_returns(random_return, (ppo_mean, ppo_std), (a2c_mean, a2c_std))

if __name__ == "__main__":
    main()
