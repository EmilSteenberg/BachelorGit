# BachelorGit
This repo is for our final submission of our Bachelor Thesis report.

There is two subfolders; UAV task and XOR task, which holds the code to run the UAV active learning and XOR active learning respectively.

The current results folder contains all of the results used in the actual Bachelor Thesis report. 

To run any of the methods do:
- Open the main.py file
- Choose your settings for the experiment in Part 1; Neural Network (NN) architecture, saving setup, sampling parameters, which methods to run, training parameters and what outputs to save.
- Run the main.py file and await completion

Once the main.py file has finished executing a results folder has been created, which contain all outputs and a pickle file per seed. The pickle file has saved:
- The seed
- All trained models
- All training and validation histories
- All sampling and training timings
- All kept indiced for all $L_p$ for all methods
- The Fisher Information scores calculated for that seed
- The configuration of the entire experiment

For the UAV task a jupyter file, test_AllMethods.ipynb has been created, to recreate the outputs of the main.py file using the saved pickle file.

For the XOR task, no such jupyter file has been made since the task is quick to execute.
