# 🔥Squad404
A competition project held by Berrijam Jam 2024 (https://www.berrijam.com/jam)

&nbsp;

# 🚀Setting Up the Python Environment
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
   ```
   python3 -m venv /path/to/project/venv    # macOS
   python -m venv C:\path\to\project\venv   # Windows
   ```
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   python3 -m venv ~/Documents/Dev/Squad404/venv   # macOS
   python -m venv C:\Documents\Dev\Squad404/venv   # Windows
   ```
5. Activate the virtual environment using.
   ```
   source /path/to/project/bin/activate      # macOS
   C:\path\to\project\Scripts\activate.bat   # Windows
   ```
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   source ~/Documents/Dev/Squad404/bin/activate     # macOS
   C:\Documents\Dev\Squad404\Scripts\activate.bat   # Windows
   ```
6. Install the listed packages within the virtual environment.
   ```
   pip3 install -r /path/to/project/requirements.txt    # macOS
   pip install -r C:\path\to\project\requirements.txt   # Windows
   ```
   For example, your current project under path /Documents/Dev/Squad404 would be:
   ```
   pip3 install -r ~/Documents/Dev/Squad404/requirements.txt   # macOS
   pip install C:\Documents\Dev\Squad404\requirements.txt      # Windows
   ```
7. Once you've finished working on the project, you can deactivate the virtual environment by simply executing:
   ```
   deactivate
   ```
### Important Note: You will need to repeat 4 and 6, to activate and deactivate the virtual environment, each time you work on the project.

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
