import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plotting A2C Rewards
spaces = ['Space1', 'Space2', 'Space3', 'Space4', 'Space5']
file_name = 'episode_rewards_a2c.csv'
window_size = 10  # You can adjust this to change the smoothness

plt.figure(figsize=(10, 6))

for space in spaces:
    file_path = os.path.join(space, file_name)
    if os.path.exists(file_path):
        rewards = pd.read_csv(file_path)
        
        # Calculate rolling average
        rewards['Rolling Average'] = rewards['A2C Reward'].rolling(window=window_size).mean()
        
        # Plot the rolling average
        plt.plot(rewards['Episode'], rewards['Rolling Average'], label=space)

    else:
        print(f"File not found: {file_path}")

plt.xlabel('Episode')
plt.ylabel('A2C Reward (Rolling Average)')
plt.title('Episode Rewards Comparison Across Spaces (A2C)')
plt.legend()
plt.grid(True)
plt.savefig('../plots/1_rewards_across_spaces_A2C.png')  # Added file extension
plt.close()


# Plotting PPO Rewards
spaces = ['Space1', 'Space2', 'Space3', 'Space4', 'Space5']
file_name = 'episode_rewards_ppo.csv'
window_size = 10  # Adjust this to change the smoothness

plt.figure(figsize=(10, 6))

for space in spaces:
    file_path = os.path.join(space, file_name)
    if os.path.exists(file_path):
        rewards = pd.read_csv(file_path)
        
        # Calculate rolling average
        rewards['Rolling Average'] = rewards['PPO Reward'].rolling(window=window_size).mean()
        
        # Plot the rolling average
        plt.plot(rewards['Episode'], rewards['Rolling Average'], label=space)

    else:
        print(f"File not found: {file_path}")

plt.xlabel('Episode')
plt.ylabel('PPO Reward (Rolling Average)')
plt.title('Episode Rewards Comparison Across Spaces (PPO)')
plt.legend()
plt.grid(True)
plt.savefig('../plots/1_rewards_across_spaces_PPO.png')  # Added file extension
plt.close()
spaces = ['Space1', 'Space2', 'Space3', 'Space4', 'Space5']
window_size = 10  # Adjust this to change the smoothness

# Plotting A2C Episode Lengths
file_name = 'episode_lengths_a2c.csv'
plt.figure(figsize=(10, 6))

for space in spaces:
    file_path = os.path.join(space, file_name)
    
    if os.path.exists(file_path):
        lengths = pd.read_csv(file_path)
        
        # Calculate rolling average
        lengths['Rolling Average'] = lengths['A2C Length'].rolling(window=window_size).mean()
        
        # Plot the rolling average
        plt.plot(lengths['Episode'], lengths['Rolling Average'], label=space)
    else:
        print(f"File not found: {file_path}")

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('A2C Length (Rolling Average)')
plt.title('A2C Length Across Episodes for Each Space')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('../plots/2_episode_lengths_comparison_A2C.png')
plt.close()

# Plotting PPO Episode Lengths
file_name = 'episode_lengths_ppo.csv'
plt.figure(figsize=(10, 6))

for space in spaces:
    file_path = os.path.join(space, file_name)
    
    if os.path.exists(file_path):
        lengths = pd.read_csv(file_path)
        
        # Calculate rolling average
        lengths['Rolling Average'] = lengths['PPO Length'].rolling(window=window_size).mean()
        
        # Plot the rolling average
        plt.plot(lengths['Episode'], lengths['Rolling Average'], label=space)
    else:
        print(f"File not found: {file_path}")

plt.xlabel('Episode')
plt.ylabel('PPO Length (Rolling Average)')
plt.title('PPO Length Across Episodes for Each Space')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('../plots/2_episode_lengths_comparison_PPO.png')
plt.close()


# Plotting overall comparison
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_agent_mean_returns(spaces, file_name='agent_comparison.csv', output_dir='iteration1/plots'):
    agents = ['PPO', 'A2C']
    
    # Initialize lists to store mean returns and standard deviations for each agent
    mean_returns = {agent: [] for agent in agents}
    return_std_dev = {agent: [] for agent in agents}

    # Loop through each space and read the agent_comparison.csv file
    for space in spaces:
        file_path = os.path.join(space, file_name)
        
        if os.path.exists(file_path):
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Aggregate data for each agent
            for agent in agents:
                agent_data = data[data['Agent'] == agent]
                if not agent_data.empty:
                    mean_returns[agent].append(agent_data['Mean Return'].values[0])
                    return_std_dev[agent].append(agent_data['Return Std Dev'].values[0])
        else:
            print(f"File not found: {file_path}")

    # Convert lists to numpy arrays for easier calculations
    for agent in agents:
        mean_returns[agent] = np.array(mean_returns[agent])
        return_std_dev[agent] = np.array(return_std_dev[agent])

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot mean returns for PPO and A2C across spaces
    bar_width = 0.35  # Width of bars
    x = np.arange(len(spaces))  # X locations for the groups

    plt.figure(figsize=(12, 6))

    # Plot for PPO
    plt.bar(x - bar_width/2, mean_returns['PPO'], yerr=return_std_dev['PPO'], 
            capsize=10, width=bar_width, label='PPO', alpha=0.7)

    # Plot for A2C
    plt.bar(x + bar_width/2, mean_returns['A2C'], yerr=return_std_dev['A2C'], 
            capsize=10, width=bar_width, label='A2C', alpha=0.7)

    # Customize the plot
    plt.title('Mean Return Comparison of PPO and A2C Across Spaces')
    plt.xlabel('Spaces')
    plt.ylabel('Mean Return')
    plt.ylim(bottom=0)
    plt.xticks(x, spaces)  # Set x-ticks to space names
    plt.legend()
    
    # Add mean values on top of bars
    for i in range(len(spaces)):
        plt.text(i - bar_width/2, mean_returns['PPO'][i] + 0.5, f'{mean_returns["PPO"][i]:.2f}', ha='center')
        plt.text(i + bar_width/2, mean_returns['A2C'][i] + 0.5, f'{mean_returns["A2C"][i]:.2f}', ha='center')

    plt.tight_layout()
    
    # Save the plot in the specified directory
    plt.savefig('../plots/3_mean_return_comparison.png')
    plt.close()

# Example usage
spaces = ['space1', 'space2', 'space3', 'space4', 'space5']
plot_agent_mean_returns(spaces)



def plot_agent_mean_lengths(spaces, file_name='agent_comparison.csv'):
    agents = ['PPO', 'A2C']
    
    # Initialize lists to store mean returns and standard deviations for each agent
    mean_returns = {agent: [] for agent in agents}
    return_std_dev = {agent: [] for agent in agents}

    # Loop through each space and read the agent_comparison.csv file
    for space in spaces:
        file_path = os.path.join(space, file_name)
        
        if os.path.exists(file_path):
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Aggregate data for each agent
            for agent in agents:
                agent_data = data[data['Agent'] == agent]
                if not agent_data.empty:
                    mean_returns[agent].append(agent_data['Mean Length'].values[0])
                    return_std_dev[agent].append(agent_data['Length Std Dev'].values[0])
        else:
            print(f"File not found: {file_path}")

    # Convert lists to numpy arrays for easier calculations
    for agent in agents:
        mean_returns[agent] = np.array(mean_returns[agent])
        return_std_dev[agent] = np.array(return_std_dev[agent])


    # Plot mean returns for PPO and A2C across spaces
    bar_width = 0.35  # Width of bars
    x = np.arange(len(spaces))  # X locations for the groups

    plt.figure(figsize=(12, 6))

    # Plot for PPO
    plt.bar(x - bar_width/2, mean_returns['PPO'], yerr=return_std_dev['PPO'], 
            capsize=10, width=bar_width, label='PPO', alpha=0.7)

    # Plot for A2C
    plt.bar(x + bar_width/2, mean_returns['A2C'], yerr=return_std_dev['A2C'], 
            capsize=10, width=bar_width, label='A2C', alpha=0.7)

    # Customize the plot
    plt.title('Mean Return Comparison of PPO and A2C Across Spaces')
    plt.xlabel('Spaces')
    plt.ylabel('Mean Return')
    plt.ylim(bottom=0)
    plt.xticks(x, spaces)  # Set x-ticks to space names
    plt.legend()
    
    # Add mean values on top of bars
    for i in range(len(spaces)):
        plt.text(i - bar_width/2, mean_returns['PPO'][i] + 0.5, f'{mean_returns["PPO"][i]:.2f}', ha='center')
        plt.text(i + bar_width/2, mean_returns['A2C'][i] + 0.5, f'{mean_returns["A2C"][i]:.2f}', ha='center')

    plt.tight_layout()
    
    # Save the plot in the specified directory
    plt.savefig('../plots/3_mean_episode_length_comparison.png')
    plt.close()

# Example usage
spaces = ['space1', 'space2', 'space3', 'space4', 'space5']
plot_agent_mean_lengths(spaces)

