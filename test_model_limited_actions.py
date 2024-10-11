import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import grid2op
from grid2op import gym_compat
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
import numpy as np

from lightsim2grid import LightSimBackend

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

        self.observation_space = self._gym_env.observation_space
        self.action_space = self.flatten_action_space(self._gym_env.action_space)

    def flatten_action_space(self, action_space):
        low = []
        high = []
        for key, space in action_space.spaces.items():
            if isinstance(space, gym.spaces.MultiBinary):
                low.extend([0] * space.n)
                high.extend([1] * space.n)
            elif isinstance(space, gym.spaces.Box):
                low.extend(space.low.tolist())
                high.extend(space.high.tolist())
            else:
                raise NotImplementedError(f"Unsupported action space type: {type(space)}")

        return gym.spaces.Box(np.array(low), np.array(high), dtype=np.float32)

    def setup_observations(self):
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
        
    def step(self, action):
        original_action = self.unflatten_action(action)

        if 'change_line_status' in original_action:

            # Round the values in 'change_line_status' based on the threshold of 0.5
            original_action['change_line_status'] = np.where(original_action['change_line_status'] > 0.5, 1, 0)
            
            # Get indices where change_line_status is 1 (i.e., lines that are affected)
            affected_lines = np.where(original_action['change_line_status'] == 1)[0]

            # If more than 4 lines are affected, randomly turn off some of them
            if len(affected_lines) > 1:
                # Randomly select lines to turn off
                lines_to_turn_off = np.random.choice(affected_lines, size=len(affected_lines) - 1, replace=False)
                # Set the selected lines' status to 0 (turn them off)
                original_action['change_line_status'][lines_to_turn_off] = 0

        # Check if 'change_bus' or 'set_bus' is in the action
        if 'change_bus' in original_action:

            # Round the values in 'change_line_status' based on the threshold of 0.5
            original_action['change_bus'] = np.where(original_action['change_bus'] > 0.5, 1, 0)

            # Get indices where change_bus is 1 (substations affected)
            affected_substations = np.where(original_action['change_bus'] == 1)[0]
            
            # If more than 4 substations are affected, randomly select 4 to keep as affected
            if len(affected_substations) > 1:
                substations_to_turn_off = np.random.choice(affected_substations, size=len(affected_substations) - 1, replace=False)
                # Set the selected substations' status to 0 (turn them off)
                original_action['change_bus'][substations_to_turn_off] = 0

        return self._gym_env.step(original_action)

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


def model():
    max_steps = 100
    env = Gym2OpEnv()

    curr_step = 0
    curr_return = 0
    is_done = False

    model_path = "models/js74rcjt/model.zip"
    model = PPO.load(model_path, env=env)

    # Reset the environment before starting
    obs, info = env.reset()
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

    while not is_done and curr_step < max_steps:
        
        action, _states = model.predict(obs)
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
        print(f"\t is action valid = {not (info['is_illegal'] or info['is_ambiguous'])}")

        if info["is_illegal"] or info["is_ambiguous"]:
            print(f"\t\t reason = {info['exception']}")
        else:
            print(f"\t\t VALID")
            
        print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")

model()