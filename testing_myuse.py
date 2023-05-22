import subprocess
import csv
import knn
import csv_utils


# Command to activate the virtual environment
activate_command = r'.\venv\Scripts\activate.bat'

args = "./data/test ./vectors/vectors_testing.csv 0"

# Command to run the first subprocess
subprocess.Popen(f'cmd /c "{activate_command} && python ./pose_training.py {args}"', shell=True).wait()

# Command to run the second subprocess
subprocess.run(f'cmd /c "{activate_command} && python ./moments_training.py {args}"', shell=True)


trained_data = csv_utils.data_from_csv_with_label('./vectors/vectors_training.csv')
queries = csv_utils.data_from_csv('./vectors/vectors_testing.csv')
answers = ["021z001","021z002","021z003","021z004","021z005","021z006","021z007","021z008","021z009","021z010","024z011"]


correct = 0
for count, query in enumerate(queries):
    _, prediction = knn.knn(trained_data, query, k=1, distance_fn=knn.euclidean_distance, choice_fn=knn.mode)

    print("Real ID - " + answers[count])
    print("Prediction: " + prediction)
    if ((answers[count] + "ps.jpg") == prediction): correct = correct + 1
    print("")
    print("")


print("Correct: " + str(correct) + "/11")