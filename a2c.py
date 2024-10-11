import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import A2C 
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.spaces import MultiDiscrete

from default_env import Gym2OpEnv
from grid2op.PlotGrid import PlotMatplot

class ActionWrapper(gym.ActionWrapper):
    """
    Custom action wrapper to flatten Dict action space into a MultiDiscrete space.
    This wrapper converts the complex Grid2Op action space (Dict) into a simpler
    format that Stable Baselines3 algorithms can work with.
    """
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        # Flatten the Dict action space into a single MultiDiscrete space
        self.action_space = self.flatten_action_space(self.env.action_space)

    def flatten_action_space(self, action_space):
        """
        Flatten the Dict action space into a single MultiDiscrete action space.
        Each sub-action will be treated as a part of a large discrete action.
        """
        # You can flatten by adding up the size of each discrete or binary action.
        # If you have continuous actions (Box), you need to discretize them.
        
        change_bus = action_space.spaces['change_bus']  # MultiBinary(57)
        change_line_status = action_space.spaces['change_line_status']  # MultiBinary(20)
        curtail = action_space.spaces['curtail']  # Box(-1, 1, (6,))
        redispatch = action_space.spaces['redispatch']  # Box(-5, 5, (6,))
        set_bus = action_space.spaces['set_bus']  # Box(-1, 2, (57,))
        set_line_status = action_space.spaces['set_line_status']  # Box(-1, 1, (20,))

        # Flatten all the components into a single MultiDiscrete space
        flattened_action_space = MultiDiscrete([
            change_bus.n,  # 57 (MultiBinary)
            change_line_status.n,  # 20 (MultiBinary)
            5,  # Discretize curtail (e.g., 5 discrete levels)
            5,  # Discretize redispatch (e.g., 5 discrete levels)
            3,  # Discretize set_bus (-1, 0, 1)
            2,  # Discretize set_line_status (-1, 1)
        ])
        
        return flattened_action_space

    def action(self, action):
        """
        Convert a flat MultiDiscrete action back into the original Dict structure.
        """
        unpacked_action = {
            'change_bus': np.array([action[0]]),  # Example: Flattened action for bus change
            'change_line_status': np.array([action[1]]),  # Example: Flattened action for line status
            'curtail': np.array([action[2]]),  # Map back the discretized curtail action
            'redispatch': np.array([action[3]]),  # Map back the redispatch action
            'set_bus': np.array([action[4]]),  # Convert back to the set_bus array
            'set_line_status': np.array([action[5]])  # Convert back to the set_line_status array
        }
        return unpacked_action

    def reverse_action(self, action):
        """
        Convert the Dict action from the environment back into a flat MultiDiscrete array.
        """
        flattened_action = [
            action['change_bus'][0],  # Flattened bus change
            action['change_line_status'][0],  # Flattened line status change
            int(action['curtail'][0]),  # Map curtail back to a single discrete action
            int(action['redispatch'][0]),  # Redispatch back to discrete
            int(action['set_bus'][0]),  # Set bus discrete representation
            int(action['set_line_status'][0]),  # Set line status discrete representation
        ]
        return flattened_action


# Main method for training
def train_a2c():
    # Create the Grid2Op environment
    env = Gym2OpEnv()

    # Wrap the environment with the custom ActionWrapper
    wrapped_env = ActionWrapper(env)

    # Vectorize the environment for Stable Baselines3
    vec_env = DummyVecEnv([lambda: wrapped_env])

    # Train A2C model
    model = A2C('MultiInputPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("a2c_grid2op_custom")

    # Evaluate the agent's performance
    obs = vec_env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()

if __name__ == "__main__":
    train_a2c() 