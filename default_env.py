import gymnasium as gym

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend
import numpy as np


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

        self.observation_space = self._gym_env.observation_space
        # self.action_space = self._gym_env.action_space

    def setup_observations(self):
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        print("WARNING: setup_observations is not doing anything. Implement your own code in this method.")
    
    def setup_actions(self):
        # Get the original action space
        original_action_space = self._gym_env.action_space

        # Modify the action space by removing 'set_bus' and 'set_line_status'
        # and limiting 'change_line_status' and 'change_bus' to affect only up to 4 elements

        def filter_action_space(action_space):
            """
            Customizes the action space:
            - Removes 'set_bus' and 'set_line_status'.
            - Ensures that 'change_line_status' and 'change_bus' only affect up to 4 elements.
            """
            # Example action keys (modify based on actual structure if needed)
            if 'set_bus' in action_space.spaces:
                del action_space.spaces['set_bus']

            if 'set_line_status' in action_space.spaces:
                del action_space.spaces['set_line_status']

            # Modify 'change_line_status' and 'change_bus' spaces to allow only up to 4 changes
            if 'change_line_status' in action_space.spaces:
                action_space.spaces['change_line_status'] = gym.spaces.MultiBinary(action_space.spaces['change_line_status'].n)
            
            if 'change_bus' in action_space.spaces:
                action_space.spaces['change_bus'] = gym.spaces.MultiBinary(action_space.spaces['change_bus'].n)

            return action_space

        # Filter the action space to meet the criteria
        self.action_space = filter_action_space(original_action_space)



    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):

        if 'change_line_status' in action:
            
            # Get indices where change_line_status is 1 (i.e., lines that are affected)
            affected_lines = np.where(action['change_line_status'] == 1)[0]

            # If more than 4 lines are affected, randomly turn off some of them
            if len(affected_lines) > 4:
                # Randomly select lines to turn off
                lines_to_turn_off = np.random.choice(affected_lines, size=len(affected_lines) - 4, replace=False)
                # Set the selected lines' status to 0 (turn them off)
                action['change_line_status'][lines_to_turn_off] = 0

        # Check if 'change_bus' or 'set_bus' is in the action
        if 'change_bus' in action:
            # Get indices where change_bus is 1 (substations affected)
            affected_substations = np.where(action['change_bus'] == 1)[0]
            
            # If more than 4 substations are affected, randomly select 4 to keep as affected
            if len(affected_substations) > 4:
                substations_to_turn_off = np.random.choice(affected_substations, size=len(affected_substations) - 4, replace=False)
                # Set the selected substations' status to 0 (turn them off)
                action['change_bus'][substations_to_turn_off] = 0

        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def main():
    # Random agent interacting in environment #

    max_steps = 100

    env = Gym2OpEnv()

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

    while not is_done and curr_step < max_steps:
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        print(f"step = {curr_step}: ")
        print(f"\t obs = {obs}")
        print(f"\t reward = {reward}")
        print(f"\t terminated = {terminated}")
        print(f"\t truncated = {truncated}")
        print(f"\t info = {info}")

        # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
        # Invalid actions are replaced with 'do nothing' action
        is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        print(f"\t is action valid = {is_action_valid}")
        if not is_action_valid:
            print(f"\t\t reason = {info['exception']}")
        print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")


if __name__ == "__main__":
    main()
