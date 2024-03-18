# 🔥Squad404-Berrijam-Jam2024
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
git clone https://github.com/DamnScallion/Squad404-Berrijam-Jam2024.git
```
### Step 3: Create a Python Virtual Environment
1. Open a terminal in VScode and navigate to your project folder.
2. Make sure you set Python 3.11 as the environment for your current VSCode workspace.
3. Run the following command to create a virtual environment named `venv`.
   ```
   python3 -m venv /path/to/project    # macOS
   python -m venv C:\path\to\project   # Windows
   ```
   For example, your current project under path /Desktop/Squad404-Berrijam-Jam2024 would be:
   ```
   python3 -m venv ~/Desktop/Squad404-Berrijam-Jam2024   # macOS
   python -m venv C:\Desktop\Squad404-Berrijam-Jam2024   # Windows
   ```
4. Activate the virtual environment using.
   ```
   source /path/to/project/bin/activate      # macOS
   C:\path\to\project\Scripts\activate.bat   # Windows
   ```
   For example, your current project under path /Desktop/Squad404-Berrijam-Jam2024 would be:
   ```
   source ~/Desktop/Squad404-Berrijam-Jam2024/bin/activate     # macOS
   C:\Desktop\Squad404-Berrijam-Jam2024\Scripts\activate.bat   # Windows
   ```
5. Install the listed packages within the virtual environment.
   ```
   pip3 install -r /path/to/project/requirements.txt    # macOS
   pip install -r C:\path\to\project\requirements.txt   # Windows
   ```
   For example, your current project under path /Desktop/Squad404-Berrijam-Jam2024 would be:
   ```
   pip3 install -r ~/Desktop/Squad404-Berrijam-Jam2024/requirements.txt   # macOS
   pip install C:\Desktop\Squad404-Berrijam-Jam2024\requirements.txt      # Windows
   ```
6. Once you've finished working on the project, you can deactivate the virtual environment by simply executing:
   ```
   deactivate
   ```
### Important Note: You will need to repeat 4 and 6, to activate and deactivate the virtual environment, each time you work on the project.

&nbsp;

# 👻Git Sample Usage:
Pull updated source code from remote branch MAIN and merge it with your local branch code
```
git pull origin main
```
Add all locally changed files ready for commit
```
git add .
```
Save all changed files as a checkpoint ready for push
```
git commit -m 'feature: add a new feature to prototypical model'
```
Push these changed files to a remote repository at branch MAIN
```
git push origin main
```

&nbsp;

# 👽Other Useful Commands:
Git clone:
```
git clone <https://name-of-the-repository-link>
```
Creating a new branch:
```
git branch BRANCH-NAME
```
Viewing branches:
```
git branch or git branch --list
```
Deleting a branch:
```
git branch -d BRANCH-NAME
```
