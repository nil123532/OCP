# OCP
Enhancing Graph Neural Networks for Materials Discovery  through Probability-Based Loss Function Design

This repository provides materials discovery enhancements to Graph Neural Networks through the design of a probability-based loss function. This work extends the Open Catalyst Project (OCP).

# Prerequisites
Before using this repository, you'll need to clone the official OCP repository into your local directory:
`git clone https://github.com/Open-Catalyst-Project/ocp.git`

# Dataset
Download the 200k s2ef dataset as instructed in the official OCP repository.

# Training
To train the SchNet models, utilize the scripts found in the train_scripts directory. Be sure to modify the paths within these scripts to match your local setup.

# Example 
If you want to train a schnet model that utilizes Evidential Loss function with seed 0, the following command should be used:
`python3 train_scripts/evidential/train_standard_evidential_0.py`

The result and subsequent .pth file generated from this command will be stored in results/evidential/evidential_0 
You can change this path by changing train_dir variable in the script.
