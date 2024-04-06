#!/bin/bash
########################################################################################################################
# 1_install.sh - installs any libraries needed for the pipeline
########################################################################################################################

# Install python package requirements from requirements.txt
# Add packages you need for your pipeline to requirements.txt file
pip install -r requirements.txt

# Add additional steps that you need for your model training pipeline to work

# Make sure all pretrained models, additional datasets, or configuration files are stored in resources directory
