import gymnasium as gym

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.PlotGrid import PlotMatplot
from gymnasium.spaces import Dict

from lightsim2grid import LightSimBackend

import matplotlib.pyplot as plt
import tqdm


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
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

        self.setup_observations()
        self.setup_actions()

        # self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

        self.plotter = PlotMatplot(self._g2op_env.observation_space)

    def setup_observations(self):
        """ Dynamic - change per step
        obs.year, obs.month, obs.day, obs.hour_of_day, obs.minute_of_hour, obs.day_of_week
        (2019, 1, 6, 0, 0, 6)

        Static - doesnt change per step
        print("Number of generators of the powergrid: {}".format(obs.n_gen))
        print("Number of loads of the powergrid: {}".format(obs.n_load))
        print("Number of powerline of the powergrid: {}".format(obs.n_line))
        print("Number of elements connected to each substations in the powergrid: {}".format(obs.sub_info))
        print("Total number of elements: {}".format(obs.dim_topo))
        Number of generators of the powergrid: 6
        Number of loads of the powergrid: 11
        Number of powerline of the powergrid: 20
        Number of elements connected to each substations in the powergrid: [3 6 4 6 5 7 3 2 5 3 3 3 4 3]
        Total number of elements: 57

        GENERATOR INFO (each genrator is an index in the array)
        print("Generators active production: {}".format(obs.gen_p))
        print("Generators reactive production: {}".format(obs.gen_q))
        print("Generators voltage setpoint : {}".format(obs.gen_v))
        Generators active production: [81.4     79.3      5.3      0.       0.      82.24667]
        Generators reactive production: [ 19.496038  71.34023   24.368923  24.368923  24.01807  -17.27466 ]
        Generators voltage setpoint : [142.1      142.1       22.        22.        13.200001 142.1     ]

        LOAD INFO (each load is an index in the array)
        print("Loads active consumption: {}".format(obs.load_p))
        print("Loads reactive consumption: {}".format(obs.load_q))
        print("Loads voltage (voltage magnitude of the bus to which it is connected) : {}".format(obs.load_v))
        Loads active consumption: [21.9 85.8 44.3  6.9 11.9 28.5  8.8  3.5  5.4 12.6 14.4]
        Loads reactive consumption: [15.4 59.7 30.8  4.8  8.3 19.4  6.1  2.4  3.9  8.8 10.5]
        Loads voltage (voltage magnitude of the bus to which it is connected) : [142.1      142.1      138.66075  139.29695   22.        21.13022 21.12955   21.478817  21.571596  21.432823  20.750198]
        
        CABLE INFO (each cable is an index in the array (orogin a->b, extremity b->a))
        print("Origin active flow: {}".format(obs.p_or))
        print("Origin reactive flow: {}".format(obs.q_or))
        print("Origin current flow: {}".format(obs.a_or))
        print("Origin voltage (voltage magnitude to the bus to which the origin end is connected): {}".format(obs.v_or))
        print("Extremity active flow: {}".format(obs.p_ex))
        print("Extremity reactive flow: {}".format(obs.q_ex))
        print("Extremity current flow: {}".format(obs.a_ex))
        print("Extremity voltage (voltage magnitude to the bus to which the origin end is connected): {}".format(obs.v_ex))
        Origin active flow: [ 4.2346096e+01  3.9900578e+01  2.3991766e+01  4.1828262e+01  3.5666172e+01  1.7225140e+01 -2.7542929e+01  8.1183472e+00  7.4602180e+00  1.7347816e+01  4.3849845e+00  8.2175179e+00  -4.4212246e+00  1.9712504e+00  6.4163899e+00  2.6171078e+01 1.4931423e+01  3.9526379e+01 -2.9753977e-14 -2.6171078e+01]
        Origin reactive flow: [-16.060501    -1.2141596   -7.423434     0.40774456  -0.44919857 7.7376227   -2.1186779   10.543067     5.6506634   15.18845 -1.5994288    3.6139119   -7.7159214    1.5654972    7.370694 -16.11945     -3.0833588   -5.630818   -23.178274    -4.492154  ]
        Origin current flow: [184.01025  162.1905   102.03776  169.9557   144.92264   76.722275 115.02098  349.20593  245.6016   605.0953   127.5342   245.28467 242.99077   67.37295  263.24207  127.98145   63.482616 165.48074 900.4441   725.5414  ]
        Origin voltage (voltage magnitude to the bus to which the origin end is connected): [142.1      142.1      142.1      142.1      142.1      142.1 138.66075   22.        22.        22.        21.13022   21.13022 21.12955   21.571596  21.432823 138.66075  138.66075  139.29695 14.861537  21.13022 ]
        Extremity active flow: [-4.1986198e+01 -3.9088322e+01 -2.3725140e+01 -4.0866714e+01 -3.4981895e+01 -1.6992859e+01  2.7643835e+01 -7.9793596e+00 -7.3712506e+00 -1.7057173e+01 -4.3787756e+00 -8.1257477e+00 4.4793596e+00 -1.9592170e+00 -6.2742519e+00 -2.6171078e+01 -1.4931423e+01 -3.9526379e+01  2.9753977e-14  2.6171078e+01]
        Extremity reactive flow: [ 11.560926    -0.54758495   3.9026122   -1.0090021   -1.0585623 -8.46951      2.4369648  -10.252009    -5.4654975  -14.616084 1.6159215   -3.4187043    7.8520093   -1.5546099   -7.0812955 17.991196     4.285767     9.055665    24.01807      5.187079  ]
        Extremity current flow: [ 176.93805   162.027      97.690315  170.2111    145.05737    79.05555  115.02098   349.20593   245.6016    605.0953    127.5342    245.28467  242.99077    67.37295   263.24207  1233.7778    424.4511   1064.1736  1050.5181   1036.4877  ]
        Extremity voltage (voltage magnitude to the bus to which the origin end is connected): [142.1      139.29695  142.1      138.66075  139.29695  138.66075 139.29695   21.478817  21.571596  21.432823  21.12955   20.750198  21.478817  21.432823  20.750198  14.861537  21.13022   22. 13.200001  14.861537]
        
        the p ratio, the ratio between the current flow in the powerline and its thermal limit
        obs.rho
        array([0.34012985, 0.36042336, 0.2721007 , 0.26722595, 0.82812935, 0.26920095, 0.3433462 , 0.5315159 , 0.4951645 , 0.7316751 , 0.28853893, 0.38265938, 0.28927472, 0.43187788, 0.39644888, 0.5446019 , 0.53346735, 0.9244734 , 0.4533958 , 0.4615403 ], dtype=float32)
        
        the number of timestep each of the powerline is in overflow (1 powerline per component)
        obs.timestep_overflow

        the status of each powerline: True connected, False disconnected
        obs.line_status 

        the topology vector the each element (generator, load, each end of a powerline) to which the object
        obs.topo_vect  

         # TODO: keep
        # TIME: year, month, day, hour_of_day, minute_of_hour, day_of_week, delta_time, current_step
        # GENERATOR INFO: gen_p, gen_q, gen_v
        # gen_margin_up
        # gen_margin_down
        # gen_theta

        # LOAD INFO: load_p, load_q, load_v)
        # load_theta

        # CABLE INFO - Origin: p_or, q_or, a_or, v_or
        # CABLE INFO - Extremity: p_ex, q_ex, a_ex, v_ex
        # CABLE INFO - rho
        # theta_ex
        # theta_or

        # _shunt_bus
        # _shunt_p
        # _shunt_q
        # _shunt_v

        # curtailment
        # curtailment_limit_effective

        # duration_next_maintenance 
        # time_next_maintenance

        # topo_vect

        # timestep_overflow

        # time_before_cooldown_line
        # time_before_cooldown_sub

        # max_step

        # target_dispatch
        # actual_dispatch

        # thermal_limit

        # attention_budget


        # TODO: remove
        #  bus connectivity matrix - dimension changes
        # curtailment_limit
        # gen_p_before_curtail
        # was_alarm_used_after_game_over
        # time_since_last_alarm
        # is_alarm_illegal
        # line_status


        """

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
        
        # This method will extract and return only the relevant parts of the observation
        filtered_obs = gym.spaces.Dict({
            "delta_time": obs["delta_time"], 
            "current_step": obs["current_step"], 
            "gen_p": obs["gen_p"], 
            "gen_q": obs["gen_q"], 
            "gen_v": obs["gen_v"], 
            "gen_margin_up": obs["gen_margin_up"], 
            "gen_margin_down": obs["gen_margin_down"], 
            "gen_theta": obs["gen_theta"], 
            "load_p": obs["load_p"], 
            "load_q": obs["load_q"], 
            "load_v": obs["load_v"], 
            "load_theta": obs["load_theta"], 
            "p_or": obs["p_or"], 
            "q_or": obs["q_or"], 
            "a_or": obs["a_or"], 
            "v_or": obs["v_or"], 
            "p_ex": obs["p_ex"], 
            "q_ex": obs["q_ex"], 
            "a_ex": obs["a_ex"], 
            "v_ex": obs["v_ex"], 
            "rho": obs["rho"], 
            "theta_ex": obs["theta_ex"], 
            "theta_or": obs["theta_or"], 
            "_shunt_bus": obs["_shunt_bus"], 
            "_shunt_p": obs["_shunt_p"], 
            "_shunt_q": obs["_shunt_q"], 
            "_shunt_v": obs["_shunt_v"], 
            "curtailment": obs["curtailment"], 
            "curtailment_limit_effective": obs["curtailment_limit_effective"], 
            "topo_vect": obs["topo_vect"], 
            "max_step": obs["max_step"], 
            "target_dispatch": obs["target_dispatch"], 
            "actual_dispatch": obs["actual_dispatch"], 
            "thermal_limit": obs["thermal_limit"], 
            "attention_budget": obs["attention_budget"], 
        })
        return filtered_obs

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        print("WARNING: setup_actions is not doing anything. Implement your own code in this method.")

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def random():
    # Random agent interacting in environment #
    
    max_steps = 100

    env = Gym2OpEnv()

    plot_helper = PlotMatplot(env._g2op_env.observation_space)

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    # print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    # print(env.action_space)
    print("#####################\n\n")

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()

    while not is_done and curr_step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        print(f"step = {curr_step}: ")
        is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        print(f"\t is action valid = {is_action_valid}")
        print(f"topo_vect = {obs['topo_vect']}: ")

    
    _ = plot_helper.plot_obs(observation=env._g2op_env.get_obs())
    plt.show()

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")

def plot():

    env = Gym2OpEnv()
    obs, info = env.reset()
    print(env.observation_space)
    plot_helper = PlotMatplot(env._g2op_env.observation_space)

    # _ = plot_helper.plot_obs(obs)
    # plt.show()
    print(obs)

    _ = plot_helper.plot_info(line_values=env._g2op_env._thermal_limit_a, gen_values=env._g2op_env.gen_pmax, load_values=[el for el in range(env._g2op_env.n_load)])
    plt.show()


if __name__ == "__main__":
    # plot()
    random()
