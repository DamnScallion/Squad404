# 🔥Squad404🚀
A competition project held by Berrijam Jam 2024 (https://www.berrijam.com/jam)

&nbsp;

# 📊Dataset
Link: https://drive.google.com/drive/folders/1Gqdcd-It2OeAGCFR4U80SJJ7nMBD5kE9

&nbsp;

# ⚙️Setting Up the Python Environment
### Step 1: Install Python 3.11 (Version 3.11.8 Recommended)
Ensure your laptop has Python 3.11 installed. 
You can download it from the official Python website (https://www.python.org/downloads/release/python-3118).
OR
Install it with Anaconda-Navigator.
### Step 2: Clone the repository
Open your laptop terminal and locate the path to somewhere you like, then run the following command
```
git clone https://github.com/DamnScallion/Squad404.git
```
### Step 3: Create a Python Virtual Environment
1. Open a terminal in VScode and navigate to your project folder.
2. Make sure you set Python 3.11 as the environment for your current VSCode workspace.
3. Run the following command to create a virtual environment named `venv`.
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   python3 -m venv ~/Documents/Dev/Squad404/venv   # macOS
   python -m venv C:\Documents\Dev\Squad404/venv   # Windows
   ```
4. Activate the virtual environment using.
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   source ~/Documents/Dev/Squad404/bin/activate     # macOS
   source C:\Documents\Dev\Squad404\Scripts\activate.bat   # Windows
   ```
5. Install the listed packages within the virtual environment.
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   pip3 install -r ~/Documents/Dev/Squad404/requirements.txt   # macOS
   pip install -r C:\Documents\Dev\Squad404\requirements.txt      # Windows
   ```

&nbsp;

# 👻Git Sample Usage:
1. Create your new branch
   ```
   git checkout -b YourBranch
   ```
2. (❗️Note: Always run this command before coding on your branch) Pull updated source code from remote branch MAIN and merge it with your branch code
   ```
   git pull origin main
   ```
3. Add all locally changed files ready for commit
   ```
   git add .
   ```
4. Save all changed files as a checkpoint ready for push
   ```
   git commit -m 'feature: add a new feature to prototypical model'
   ```
5. Push these changed files to a remote repository at YourBranch
   ```
   git push
   ```
   If you encounter a message 'fatal: The current branch YourBranch has no upstream branch.', you can run the following command
   ```
   git push -set-upstream YourBranch
   ```

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
   pip freeze -r requirements.txt
   ```
   Then use the Git tool to commit and push your requirements.txt to our GitHub Repo. So everyone can share the up-to-date packages.
3. This command can keep all your Python packages up to date in your local virtual environment.
   ```
   pip install -r requirements.txt
   ```
