Berrijam Jam 2024 Submission Template
-------------------------------------
The example code and structure is provided as a guide to get you started. Replace the functions and code with your own
solution and algorithm. As long as you don't change the interface (i.e. command line arguments and the output format of
predictions) the automated evaluation process will be able to use your code (recipe) to train and generate predictions
for each problem.

Python Code
-----------
We are providing three files:
 * common.py - common utility functions that are shared between train.py and predict.py such as loading, saving data.
 * train.py - the script to train a problem specific model.
 * predict.py - the script to generate prediction using the model produced by train.py

Note that we have provided stubs and samples. It is your job to add and modify to make the code work. You'll see
RuntimeError Exceptions raised from functions that have not been implemented, along with comments to get you started.

You can add additional functions or modify the signature of the functions as needed.

The only thing you should NOT modify are:
 * command line arguments and their behavior
 * final prediction output structure of the predictions CSV files.

If you modify those it might break the automated evaluation process, so we will not be able to use your code, resulting
in your team getting a score of 0.

Command Line Arguments for train.py
-----------------------------------

usage: train.py [-h] -d TRAIN_DATA_IMAGE_DIR -l TRAIN_DATA_LABELS_CSV -t
                TARGET_COLUMN_NAME -o TRAINED_MODEL_OUTPUT_DIR

options:
  -h, --help            show this help message and exit
  -d TRAIN_DATA_IMAGE_DIR, --train_data_image_dir TRAIN_DATA_IMAGE_DIR
                        Path to image data directory
  -l TRAIN_DATA_LABELS_CSV, --train_data_labels_csv TRAIN_DATA_LABELS_CSV
                        Path to labels CSV
  -t TARGET_COLUMN_NAME, --target_column_name TARGET_COLUMN_NAME
                        Name of the column with target label in CSV
  -o TRAINED_MODEL_OUTPUT_DIR, --trained_model_output_dir TRAINED_MODEL_OUTPUT_DIR
                        Output directory for trained model


Command Line Arguments for predict.py
-------------------------------------

usage: predict.py [-h] -d PREDICT_DATA_IMAGE_DIR -l PREDICT_IMAGE_LIST -t
                  TARGET_COLUMN_NAME -m TRAINED_MODEL_DIR -o
                  PREDICTS_OUTPUT_CSV

options:
  -h, --help            show this help message and exit
  -d PREDICT_DATA_IMAGE_DIR, --predict_data_image_dir PREDICT_DATA_IMAGE_DIR
                        Path to image data directory
  -l PREDICT_IMAGE_LIST, --predict_image_list PREDICT_IMAGE_LIST
                        Path to text file listing file names within
                        predict_data_image_dir
  -t TARGET_COLUMN_NAME, --target_column_name TARGET_COLUMN_NAME
                        Name of column to write prediction when generating
                        output CSV
  -m TRAINED_MODEL_DIR, --trained_model_dir TRAINED_MODEL_DIR
                        Path to directory containing the model to use to
                        generate predictions
  -o PREDICTS_OUTPUT_CSV, --predicts_output_csv PREDICTS_OUTPUT_CSV
                        Path to CSV where to write the predictions

Final Prediction Output Format
------------------------------
The final predictions generated for each dataset should be a CSV with following format

    Filename, <TARGET>
    predict_file_001.png, Yes
    predict_file_002.png, No
    predict_file_003.png, No
    ...

Where <TARGET> is supplied via the -t argument (i.e. "Is Epic", "Needs Respray" or "Is GenAI")


Bash Scripts
------------
We are also providing two bash scripts:

 * 1_install.sh - to install any python packages, code or libraries that are specific to your algorithms. You should
   look at it carefully adding to it or requirements.txt file.

 * 2_run_pipelines.sh - this script will call train.py and predict.py of each of the datasets. The stub shows how it
   calls train for the three datasets that we have provided. The predict stage will call the full evaluation dataset.

Evaluation Process
------------------
The automated evaluation process will start by:
 1. unzipping the team's submission zip file
 2. create a venv (Virtual Python Environment) within the extracted folder
 3. call 1_install.sh to set up the venv
 4. call our evaluation process, which looks similar to 2_run_pipeline.sh to train models and generate predictions
 5. compare the predictions for each image (across 5 problems) and comparing them with ground truth.
 6. Calculate an overall F1 score

Since the process is automated, it is important to make sure your code works automatically without manual intervention.

WARNING: If the automated process fails due to your code not meeting the interface requirements, wrong python version,
wrong package version, or missing dependencies your team will get a score of 0. Please test, test and check your code
in new fresh venv to make sure it works.

Getting started - step by step
-------------------------------
1. Create a folder for your team's submission (e.g. BerrijamTeam)
2. Copy requirements.txt, 1_install.sh, 2_run_pipelines.sh, common.py, train.py and predict.py into folder.
3. Make these executable - i.e. chmod u+x *.py ; chmod u+x *.sh
4. Add .python-version file with the version of python you are using
5. Update requirements.txt to include any python packages that your code needs.
6. Update 1_install.sh to add additional configuration steps
7. Create a directory called "resources" and place any pre-trained models, config files, etc that you code relies upon
   inside resources. This will help ensure the code base is easier to manage and neatly organised.
8. Modify common.py, train.py and predict.py to make the pipeline steps work.
9. Modify 2_run_pipelines.sh to run and test your code
10. Test, debug, and develop until you are satisfied
11. Add documentation or other instructions not already part of 1_install.sh to a Readme.txt
12. Add your 2-min video as either an .mp4 or .webm
13. Add slides or presentation that you used for video in resources folder
14. When you are ready to submit, zip up the folder (e.g. BerrijamTeam.zip)
15. Upload the zip to the unique location provided for your team.

What does the final submission folder look like
-----------------------------------------------

At the minimum your final submission folder when unzipped will look something like the following:
    BerrijamTeam
    ├── .python-version
    ├── 1_install.sh
    ├── 2-min-video.mp4
    ├── 2_run_pipelines.sh
    ├── Readme.txt
    ├── common.py
    ├── predict.py
    ├── requirements.txt
    ├── resources
    │   ├── externaldata
    │   └── pretrained
    └── train.py

If you are using a pre-trained model, include it under resources/pretrained as well as licence and source of model.

Only include additional data if your pipeline uses it under the resources/externaldata folder. No need to include the
sample data Berrijam provided, we already have a copy of that.


FAQ:
----
 Q1: Can we make changes to the code?
 Ans: Yes, you will modify the code, filling in stubs, adding functions or libraries. However, do NOT change the command
 line arguments interface or the format of the predictions that are generated. Otherwise, the automated evaluation
 system may fail.

 Q2: Help code doesn't work or throws an exception. What should we do?
 Ans: Debug. Remember the code is just a reference guide, it is your job to figure out and modify it to work. We will
 not provide debugging help to individual teams.

 Q3: Where do I find the data for train and predict?
 Ans: We previously released the data samples here: https://drive.google.com/drive/folders/1Gqdcd-It2OeAGCFR4U80SJJ7nMBD5kE9
 We will NOT be releasing the full dataset. You can use the sample dataset to create your own train and predict sets.

 Q4: What version of python does the evaluation process support?
 Ans: Versions after 3.9 are supported. If you need a specific python version for your code, please add a file called
 ".python-version" with the exact version you need. If we find this file in your code, we'll use that version of python.

 Q5: What should we put in .python-version file?
 Ans: The exact python version. For example if your code uses 3.9.5 your .python-version file will have a single line:
      3.9.5

 Q6: How do we include open source python packages that we need in my code?
 Ans: Add the python package (and ideally with specific version) in requirements.txt. See included requirements.txt
 for an example.

 Q7: We wrote my own python module. How do we include it?
 Ans: Great. You can include the python code in the base directory or include the python wheel in the resources. If you
 are including the wheel, ensure you update requirements.txt to make sure it is installed. If you might want to modify
 1_install.sh to extend the PYTHONPATH variable including directories with python code.

 Q8: We are using pre-trained deep learning model X. Where do we put it?
 Ans: Put the model file(s) and licenses associated with it inside resources directory.

 Q9: We are using additional training data. How do we include it?
 Ans: Put it inside the resources directory. you might want to organize it in sub-directory

 Q10: We don't want to use Pillow library. Can we use another library instead?
 Ans: Yes. You can modify the code to include other open source libraries and custom solutions. These are just to get
 you started.

 Q11: What happens if the install, train or predict encounters errors during the evaluation process?
 Ans: If the automated process fails, we will not be able to generate a score for you and your team will be given a
 score of 0. Given the large number of teams, we will not be able to manually debug each team's scripts. So we strongly
 encourage you to test your scripts and code in fresh environments?

 Q12: What is venv?
 Ans: See primer on Python Virtual Environments here: https://realpython.com/python-virtual-environments-a-primer/

 Q13: Can I use Jupyter Notebooks?
 Ans: You can use them to develop your solution or document, but the automated method will NOT use jupyter notebooks.

 Q14: Can I use a GPU?
 Ans: Teams can use CPU or GPU.  Solutions are expected to run on an Ubuntu machine with a GPU with 8GB RAM,
 Nvidia Drivers 535.154.05 and CUDA version 12.2. The full re-training and prediction process for each problem should
 not exceed 10 minutes, so that evaluation across five problems should finish within 60 minutes.

 Q15: What if we need to install additional executables or libraries that are not python or pip installable?
 Ans: You can, as long as you can ensure the libraries are part of the 1_install.sh and work on Ubuntu 20.04 or 22.04
 versions of linux. Make sure you include the libraries to install within resource folder and include the steps to
 install it within 1_install.sh

