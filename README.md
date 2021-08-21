# DQN_Breakdown

## Abstract -
Deep Q-Network(DQN) being the first deep reinforcement learning method proposed by DeepMind, approximates a state-value function in a Q-learning framework with a neural network {cite}. The DQN algorithm proposed by {cite} 'Playing Atari with Deep Reinforcement Learning' on NIPS 2013 uses a Convoluted Neural Network architecture as a function approximator and uses three techniques (-namely Q-Targets, Target Network, Error Truncation) to overcome instability and divergence. The research below examines the importance of each of these factors on solving control theory problems from the classic RL literature(Acrobot, CartPole and MountainCar). The experiments are conducted with different combinations of these techniques present. Furthermore, each of these techniques were analysed exclusively on Acrobot environment and the common trends are presented.  

## Setting up the Anaconda Environment - 
The conda environment used for performing the experiments is condensed into a .yml file and is added to the repository with the name 'dqnvariations.yml'\
If you wish to replicate the experiments, I'd suggest creating an exclusive anaconda environment using this file by following the command below.

```conda env create -f dqnvariations.yml```

This line of code run on your terminal from the directory with the .yml file should create an environment with the name 'rl'. To activate this environment,
use the command below.

```conda activate rl```

## Project Structure -
There are 3 different experiments conducted. Each of these are organised into the respective directories.
Please read the report to understand the experiments in detail.

**DQN_Main** - \
8 Variations(±Q,±E,±T) on 3 environments.\
Each environment has its code in seperate directories. (DQN_<gym_env>)\

**DQN_QTargets** - \
This experiment has 4 variations with different frequencies of target network update. (Q1 = 1, Q2 = 5, Q3 = 10, Q4 = 20)\
Although the results are only present for Acrobot Environment, the code for all the environments is available in the directory for conducting simlar experiments.

**DQN_ExpReplay** - \
This experiment has 10 variations. [(10k,64),(10k,32),(5k,64),(5k,32),(1,1) where (n1,n2) is an ExpReplay configuration with memory size n1 and mini batch size n2]
Although the results are only present for Acrobot Environment, the code for all the environments is available in the directory for conducting similar experiments.

**DQN_Truncation** - \
This experiment has 4 variations. Different levels of truncation on the gradient generated for updating the main QNetwork was experimented with(T1 = ±1, T2 = ±5, T3 = ±10, T4 = ±20)\
Although the results are only present for Acrobot Environment, the code for all the environments is available in the directory for conducting similar experiments.

## Directory Structure - 
Each project directory has four classes of files.\

1) **Base Files** - (Agents.py, QNetworks.py and ReplayBuffer.py)\
These files have all the variations in them. QNetworks.py contains the neural networks used as QNetworks and ReplayBufer.py contains all possible variations in the Replay Buffer. These files are different for different experiments
as the variations needed for each experiment is different. You wouldn't have to run any of these files for performing the experiment. You can add more variations in here. 

2) **Variation Files** - (DQN_<gym_env>_<experiment_type><variation_number>_Python.py)\
These files use the functions defined in the base files. Each of these files represent a variation. For getting the result of an experiment, one needs to run these files. 
For running the file, use the following command after activating the conda environment,
```python3 DQN_<gym_env>_<experiment_type>....(File name).py```
Running these files save the variables in Data files (Next type of file).\

3) **Data Files** - (Data_<gym_env>_<experiment_type>)\
Running the Variation Files above, store the variables generated during the program (Variables = Lives, Scores in each episode until the agent learns the episode, time taken for the agent to learn). These data files can also be found in the 'DQN_Data' directory all put together to visualise and analyse the data. 

4) **PBS Files** - (corresponding_variation_file_name.PBS)\
PACE cluster-(Georgia Tech) was extensively used for the computing of all these experiments. For running a certain code on their servers, one has to create a PBS file corresponding to the python files one wish to run. The corresponding PBS file for every variation file is also added to the directories. Please refer to the official documentation - https://pace.gatech.edu/ to assist you in setting up and getting started. To run the code once all the setting up is done, use the following command on the terminal to submit your job to the cluster.
``` qsub <pace_file_name>.PBS``` 

## Visualizing Results
The data generated from the experiments mentioned above is stored in the folder called **Data_DQN**.\
Each of the above experiments have their own dedicated folder. Each folder has a seperate Jupyter Notebook for visualising the files.\
Hickle was used to store the python variables and reading them. Please refer to the official documentation to follow through the code used in this project to store and load files using hickle from here - https://pypi.org/project/hickle/.

I Personally thank [Zaiwen Chen](https://www.zaiweichen.com/) for mentoring me through the project and [Prof.Siva Theja Maguluri](https://sites.google.com/site/sivatheja/group?authuser=0) for guiding me through my research experience. PACE Cluster was extensively used for my experiments and I'm thankful for all the resources I could use. I also thank the other mentors (Prakirt Raj Jhunjhunwala,Sajad Khodadadian,Sushil Varma) for regular reviews and suggestion for the project. I also thank my co-participants (Adam Profilli,Hyen Jay Lee,Kyung Min (Brian) Ko,Sam Hodges) in Prof.Siva's REU program for a wonderful research experience. 
