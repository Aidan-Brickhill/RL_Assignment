the run.sh file submits the job to the cluster
the job.batch file runs the code on the cluster

baseline and iteration 1 are stand alone and dont need the merge csv file to get results
merge csv results, merges the csvs from iteration 2

baseline contains the code for the baseline agents. it trains 5 agents per algorithm and plots the results
iteration1 contains the code for the iteration1 agents. it trains 5 agents per algorithm per space and plots the results, each sapce is predefined within the code
iteration2 contains the code for the iterationw agents. it trains 5 agents per algorithm per reward, merge_csv plots the results, each reward is predefined within the code