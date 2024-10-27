import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

agents_string = ['PPO', 'A2C']
rewards_string = [
    'L2RPNReward', 'N1Reward', 'CloseToOverflowReward', 
    'EconomicReward', 'LinesCapacityReward', 'IncreasingFlatReward'
]

folders = ["iteration20", "iteration21" "iteration22", "iteration23", "iteration24"]

# combines seperate csv files into 1
for agent in agents_string:
    for reward in rewards_string:
        data_frames = []

        for folder in folders:
            file_name = f"{reward}_{agent}_results.csv"
            file_path = os.path.join(folder, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df.drop(columns=['Reward Mean List', 'Length Mean List'], errors='ignore')
                data_frames.append(df)
            else:
                print(f"{file_path} not found.")

        if data_frames:
            avg_df = pd.concat(data_frames).groupby(level=0).mean()
            avg_file_path = f"{reward}_{agent}_results.csv"
            avg_df.to_csv(avg_file_path, index=False)
            print(f"Averaged results saved in the root folder as {avg_file_path}")
        else:
            print(f"No data found for {reward}_{agent}.")

plot_dir = "iteration2/plots"
os.makedirs(plot_dir, exist_ok=True)

# plot the results
data = {agent: {reward: {'r_mean': [], 'r_std': [], 'l_mean': [], 'l_std': []} for reward in rewards_string} for agent in agents_string}

for agent in agents_string:
    for reward in rewards_string:
        data_frames = []
        
        for folder in folders:
            file_name = f"{reward}_{agent}_results.csv"
            file_path = os.path.join(folder, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df.drop(columns=['Reward Mean List', 'Length Mean List'], errors='ignore')
                data_frames.append(df)
        
        if data_frames:
            avg_df = pd.concat(data_frames).groupby(level=0).mean()
            data[agent][reward]['r_mean'] = avg_df['R Mean Avg'].mean()
            data[agent][reward]['l_mean'] = avg_df['L Mean Avg'].mean()
            data[agent][reward]['r_std'] = avg_df['R Std Avg'].mean() 
            data[agent][reward]['l_std'] = avg_df['L Std Avg'].mean()

for agent in agents_string:
    r_means, r_stds = [], []
    for reward in rewards_string:
        r_means.append(data[agent][reward]['r_mean'])
        r_stds.append(data[agent][reward]['r_std'])
    
    plt.figure(figsize=(10, 6))
    plt.bar(rewards_string, r_means, yerr=r_stds, capsize=10)
    plt.title(f'Final Return Comparison for {agent}')
    plt.ylabel('Mean Return')
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    for i, v in enumerate(r_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center') 
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{agent}_return_comparison.png")
    plt.close()

for agent in agents_string:
    l_means, l_stds = [], []
    for reward in rewards_string:
        l_means.append(data[agent][reward]['l_mean'])
        l_stds.append(data[agent][reward]['l_std'])
    
    plt.figure(figsize=(10, 6))
    plt.bar(rewards_string, l_means, yerr=l_stds, capsize=10)
    plt.title(f'Final Length Comparison for {agent}')
    plt.ylabel('Mean Length')
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    for i, v in enumerate(l_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{agent}_length_comparison.png")
    plt.close()

for agent in agents_string:
    r_means_agent, l_means_agent = [], []
    r_stds_agent, l_stds_agent = [], []

    for reward in rewards_string:
        r_means_agent.append(data[agent][reward]['r_mean'])
        r_stds_agent.append(data[agent][reward]['r_std'])
        l_means_agent.append(data[agent][reward]['l_mean'])
        l_stds_agent.append(data[agent][reward]['l_std'])

    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(len(rewards_string))

    plt.bar(index, r_means_agent, bar_width, yerr=r_stds_agent, label='Return', capsize=5)
    for i, v in enumerate(r_means_agent):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    
    plt.bar(index + bar_width, l_means_agent, bar_width, yerr=l_stds_agent, label='Length', capsize=5)
    for i, v in enumerate(l_means_agent):
        plt.text(i + bar_width, v + 0.5, f'{v:.2f}', ha='center')
    
    plt.xlabel('Rewards')
    plt.ylabel('Mean Values')
    plt.title(f'{agent} Reward Comparison')
    plt.xticks(index + bar_width / 2, rewards_string, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{agent}_rewards_comparison.png")
    plt.close()
