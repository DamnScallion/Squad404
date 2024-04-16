#!/bin/bash
########################################################################################################################
# 2_run_pipelines.sh - Runs the train and predict for each dataset to train and then generate predictions
########################################################################################################################

########################################################################################################################
# Data - Is Epic Intro
########################################################################################################################
python train.py -d "data/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "models/Is Epic/"
python predict.py -d "data/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "models/Is Epic/" -o "predictions/Is Epic Intro Full.csv"

########################################################################################################################
# Data - Needs Respray
########################################################################################################################
python train.py -d "data/Data - Needs Respray - 2024-03-26" -l "Labels-NeedsRespray-2024-03-26.csv" -t "Needs Respray" -o "models/Needs Respray/"
python predict.py -d "data/Data - Needs Respray Full" -l "Needs Respray Files.txt" -t "Needs Respray" -m "models/Needs Respray/" -o "predictions/Needs Respray Full.csv"

########################################################################################################################
# Data - Is GenAI
########################################################################################################################
python train.py -d "data/Data - Is GenAI - 2024-03-25" -l "Labels-IsGenAI-2024-03-25.csv" -t "Is GenAI" -o "models/Is GenAI/"
python predict.py -d "data/Data - Is GenAI Full" -l "Is GenAI Files.txt" -t "Is GenAI" -m "models/Is GenAI/" -o "predictions/Is GenAI Full.csv"
