import subprocess


# Command to activate the virtual environment
activate_command = r'.\venv\Scripts\activate.bat'

args = "./data/training ./vectors/vectors_training.csv 1"

# Command to run the first subprocess
subprocess.Popen(f'cmd /c "{activate_command} && python ./pose_training.py {args}"', shell=True).wait()

# Command to run the second subprocess
subprocess.run(f'cmd /c "{activate_command} && python ./moments_training.py {args}"', shell=True)
