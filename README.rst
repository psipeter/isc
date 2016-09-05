A Psychologically-Motivated Model of Opinion Change with Applications to American Politics

Install
============

Clone the GitHub repository
---------------------------
1. Install Python 2.7.X (https://www.python.org/downloads/) and Git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
2. Open a terminal (Max/Linux) or a command prompt (Windows)
3. Clone the repository:
	- git clone https://github.com/psipeter/influence_susceptibility_conformity.git
4. Install with Pip (OR python):
	- cd isc
	- pip install .
	- (python setup.py develop)
5. Install missing packages if necessary
	- The SciPy Stack: Matplotlib, Numpy, Scipy, and Pandas (https://www.scipy.org/install.html)
	- Seaborn (https://stanford.edu/~mwaskom/software/seaborn/installing.html)
	- Optional packages required for fit_empirical_ISC: 'hyperopt' which can be installed through Git (https://github.com/hyperopt/hyperopt); 'MongoDB', a parallelization package for hyperopt (https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB)

What are these files?
=====================
1. 'agent.py' is the python class for agents, and contains all the important equations
2. 'model.py' is the main file, which imports the experimental parameters, initializes the simulation, runs it, exports the data, and plots pretty pictures
3. 'parameters.txt' is the masin parameter file for model.py. Written in dictionary format for easy import/export.
4. 'fit_empirical.py' is the optimization algorithm used to find parameters which reproduced the empirical data.
5. 'fit_parameters.txt' is the parameter file for 'fit_empirical'.
6. 'broockman_data.txt' and 'pew_data.txt' are the empirical data files, also in dictionary format.
7. 'data' folder is where model outputs are saved. Each is saved into a subfolder with a randomly-generated name
8. 'figures' includes the parameters, data, and figures for the results reported in the paper

Run
=======

1. Navigate to the 'isc/isc' folder in the terminal/cmd
2. Edit the 'parameters.txt' file
	- Be sure to reset the seed to get unique behavior.
3. run 'python model.py'
4. You should see the message 'Running Simulation...' followed by a progress indicator
5. When the simulation finishes, check out the 'data' folder to see your results. 

Questions?
==========
1. Email 'psipeter@gmail.com'
