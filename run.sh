#!/bin/bash

# Source the .bashrc file
source ~/.bashrc

# Activate the conda environment
conda activate RL_Lab_DQN_env

# Submit the job multiple times using sbatch
sbatch job.batch

# Monitor the job queue
watch -n 1 squeue -u abrickhill