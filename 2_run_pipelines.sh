#!/bin/bash
########################################################################################################################
# 2_run_pipelines.sh - Runs the train and predict for each dataset to train and then generate predictions
########################################################################################################################

########################################################################################################################
# Data - Is Epic Intro
########################################################################################################################
python train.py -d "path/to/data/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models/Is Epic/"
python predict.py -d "path/to/data/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/models/Is Epic/" -o "path/to/predictions/Is Epic Intro Full.csv"

########################################################################################################################
# Data - Needs Respray
########################################################################################################################
python train.py -d "path/to/data/Data - Needs Respray - 2024-03-26" -l "Labels-NeedsRespray-2024-03-26" -t "Needs Respray" -o "path/to/models/Needs Respray/"
python predict.py -d "path/to/data/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Needs Respray" -m "path/to/models/Needs Respray/" -o "path/to/predictions/Needs Respray Full.csv"

########################################################################################################################
# Data - Is GenAI
########################################################################################################################
python train.py -d "path/to/data/Data - Is GenAI 2024-03-25" -l "Labels-IsGenAI-2024-03-25.csv" -t "Is GenAI" -o "path/to/models/Is GenAI/"
python predict.py -d "path/to/data/Data - Is GenAI Full" -l "Is GenAI Files.txt" -t "Is GenAI" -m "path/to/models/Is GenAI/" -o "path/to/predictions/Is GenAI Full.csv"
