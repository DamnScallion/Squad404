# 🔥Squad404

##### A Machine Learning competition project held by Berrijam Jam 2024 (https://www.berrijam.com/jam)

##### 📊Dataset Link: https://drive.google.com/drive/folders/1Gqdcd-It2OeAGCFR4U80SJJ7nMBD5kE9

&nbsp;

# 🔗Project GitHub Link

##### https://github.com/DamnScallion/Squad404

&nbsp;

# 🚀Quick Start

### Step 1: Install Python 3.11 (Version 3.11.8 Recommended)

Ensure your laptop has Python 3.11.8 installed.
You can download it from the official Python website (https://www.python.org/downloads/release/python-3118).
OR
Install it with Anaconda-Navigator.

### Step 2: Clone or download the repository

Open your laptop terminal and locate the path to somewhere you like, then run the following command

```
git clone https://github.com/DamnScallion/Squad404.git
```

Or download the whole repository as a zip file then unzip it.

### Step 3: Create a Python Virtual Environment

1. Open a terminal in VScode and navigate to your project folder.
2. Make sure you set Python 3.11.8 as the environment for your current VSCode workspace.
3. Run the following command to create a virtual environment named `venv`.
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   python3 -m venv ~/Documents/Dev/Squad404/venv   # macOS
   python -m venv C:\Documents\Dev\Squad404/venv   # Windows
   ```
4. Activate the virtual environment.
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   source ~/Documents/Dev/Squad404/venv/bin/activate     # macOS
   source C:\Documents\Dev\Squad404\venv\Scripts\activate.bat   # Windows
   ```

### Step 4: Project runing pinelines

1. Grant permission to these 2 shell script files, run the following command:
   ```
   chmod u+x 1_install.sh 2_run_pipelines.sh
   ```
2. Run the scirpt to install the listed packages within the virtual environment:
   ```
   ./1_install.sh
   ```
3. (❗️Optional❗️) If u wish to run the project in Windows system with CUDA as device, u need run few more steps after above command:
   ```
   pip3 uninstall torch torchvision torchaudio
   ```
   ```
   pip3 cache purge
   ```
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
4. Run the script to train and predict for each dataset:
   ```
   ./2_run_pipelines.sh
   ```

&nbsp;

# 👻Git Sample Usage:

Step 1. Syncing all remote branch

```
git fetch --all
```

Step 2. (❗️Note: Always run this command before coding on your branch) Pull updated source code from remote branch MAIN and merge it with your branch code

```
git pull origin main
```

Step 3. Create your new branch

```
git checkout -b YourBranch
```

Step 4. Add all locally changed files ready for commit

```
git add .
```

Step 5. Save all changed files as a checkpoint ready for push

```
git commit -m 'feature: add a new feature to prototypical model'
```

Step 6. Push these changed files to a remote repository at YourBranch

```
git push
```

If you encounter a message 'fatal: The current branch YourBranch has no upstream branch.', you can run the following command

```
git push -set-upstream YourBranch
```

Step 7. Go to our GitHub Repo and create a Pull Request. If you reviewed your code and it had no conflict with the 'main' branch, then you can merge it into 'main'.

### ⚠️Important note: Repeat Step 1 and Step 2 if you have been inactive in the project for a while, or your branch is way out to date.

&nbsp;

# 👽Other Useful Commands:

Git clone:

```
git clone <https://name-of-the-repository-link>
```

Viewing branches:

```
git branch or git branch --list
```

Switching branch:

```
git checkout ANOTHER-BRANCH
```

Deleting a branch:

```
git branch -d BRANCH-NAME
```

&nbsp;

# 🛠️Packages Management:

1. In case you need to install any new package to the project.
   For example, install opencv-python in your local virtual environment. Run this command in your terminal:
   ```
   pip install opencv-python
   ```
2. Then run this command to freeze all virtual environment installed packages:
   ```
   pip freeze > requirements.txt
   ```
   Then use the Git tool to commit and push your requirements.txt to our GitHub Repo. So everyone can share the up-to-date packages.
3. This command can keep all your Python packages up to date in your local virtual environment.
   ```
   pip install -r requirements.txt
   ```

&nbsp;

# 🔧Setting up Git LFS (Git Large File Storage)

1. Install Git LFS: If you haven't already installed Git LFS, you can download it from https://git-lfs.github.com/. After downloading, run the installation.
2. Initialize Git LFS in Your Repository: Navigate to your local repository in your terminal and run the following command:
   ```
   git lfs install
   ```
3. Track Large Files with Git LFS: Before adding and committing your large files, you need to track all .pth files, you can run:
   ```
   git lfs track "*.pth"
   ```
4. Git add, commit and push your code to the repo.
