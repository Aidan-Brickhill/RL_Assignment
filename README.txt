Overview
This project includes multiple scripts and configurations to train and evaluate reinforcement learning agents across different baselines and iterations on a cluster. Each iteration introduces variations in training spaces and rewards.

Files
run.sh: Submits the job to the cluster for execution.
job.batch: Defines the batch job that will run the code on the cluster.
Iterations and Baselines
Baseline:

Contains the code for baseline agents.
Trains 5 agents per algorithm and generates plots of their performance.
Baseline results do not require merged CSV files for analysis.
Iteration 1:

Contains code for agents in the first iteration, training 5 agents per algorithm across multiple predefined spaces.
Each space is defined within the code, and results are plotted directly.
Like the baseline, this iteration does not require the merged CSV file for result analysis.
Iteration 2:

Contains code for agents in the second iteration, training 5 agents per algorithm across different predefined rewards.
Results from multiple CSV files are merged using the merge_csv script, which consolidates results and generates plots for analysis.
Result Merging
merge_csv:
Used in Iteration 2 to merge individual CSV files from each agent, consolidating the data for comprehensive result plots.