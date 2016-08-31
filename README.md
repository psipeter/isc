# influence_susceptibility_conformity
A Psychologically-Motivated Model of Opinion Change with Applications to American Politics
Peter Duggins

Install
============

## Clone the GitHub repository
1. Install Git - https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
2. Open a terminal (Max/Linux) or a command prompt (Windows)
3. Enter 'git clone https://github.com/psipeter/influence_susceptibility_conformity.git'

# Install necessary packages
1. Python 2.7.X (https://www.python.org/downloads/)
2. The SciPy Stack: Matplotlib, Numpy, Scipy, and Pandas (https://www.scipy.org/install.html)
3. Seaborn (https://stanford.edu/~mwaskom/software/seaborn/installing.html)

## Peruse the files
1. 'agent.py' is the python class for agents, and contains all the important equations
2. 'ISC.py' is the main file, which imports the experimental parameters, initializes the simulation, runs it, exports the data, and plots pretty pictures
3. 'parameters.txt' is the parameter file (duh). Written in dictionary format for easy import/export.
	- Be sure to reset the seed to get unique behavior.
4. 'fit_empirical.py' is the optimization algorithm used to find parameters which reproduced the empirical data. You probably won't use this. It takes a long time to run.
	- It utilizes the package 'hyperopt' which can be installed through Git (https://github.com/hyperopt/hyperopt), or a parallelization package for hyperopt called MongoDB (https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB), or a homemade evolutionary algorithm. 
5. 'fitting_parameters.txt' is the parameter file for 'fit_empirical'.
	- 'sim_threshold' is the similarity beyond which the 'fit_empirical' will start outputting plots and data.
	- 'issue' refers to the subdataset from Broockman.
	- 'optimization' refers to the optimization method: 'evolve', 'hyperopt', or 'mongodb'.
	- 'loss_metric referes to how similarity is measured: 'RMSE' for root mean square error, or 'JSD' for Jensen-Shannon Divergence
	- 'averages' is the number of uniquely-seeded realizations that each are run for each parameter combo, then averaged to calculate loss
	- 'max_evals' is the number of parameter combos to run for 'hyperopt'
	- 'evo_popsize', 'generations', 'threads', 'size_tournaments', 'mutation_rate', and 'mutation_amount' are parameters for the evolutionary algorithm
6. 'broockman_data.txt' and 'pew_data.txt' are the empirical data files, also in dictionary format.
7. 'data' folder is where model outputs are saved. Each is saved into a subfolder with a randomly-generated name
8. 'figures' includes the parameters, data, and figures for the results reported in the paper

Run
=======

1. Navigate to the 'influence_susceptibility_conformity' folder in the terminal/cmd
2. Edit the 'parameters.txt' file
3. run 'python ISC.py'
4. You should see the message 'Running Simulation...' followed by a progress indicator
5. When the simulation finishes, check out the 'data' folder to see your results. 