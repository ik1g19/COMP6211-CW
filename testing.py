import subprocess
import csv
import knn


def data_from_csv_with_label(filename):
    d = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            float_row = [row[0]] + [float(value) for value in row[1:]]
            d.append(float_row)
    return d

def data_from_csv(filename):
    d = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            float_row = [float(value) for value in row[1:]]
            d.append(float_row)
    return d


# Command to activate the virtual environment
activate_command = r'.\venv\Scripts\activate.bat'

args = "./data/test ./vectors/vectors_testing.csv 0"

# Command to run the first subprocess
subprocess.Popen(f'cmd /c "{activate_command} && python ./pose_training.py {args}"', shell=True).wait()

# Command to run the second subprocess
subprocess.run(f'cmd /c "{activate_command} && python ./moments_training.py {args}"', shell=True)


trained_data = data_from_csv_with_label('./vectors/vectors_training.csv')
queries = data_from_csv('./vectors/vectors_testing.csv')
queries_and_label = data_from_csv_with_label('./vectors/vectors_testing.csv')


for count, query in enumerate(queries):
    k_labels, prediction = knn.knn(trained_data, query, k=1, distance_fn=knn.euclidean_distance, choice_fn=knn.mode)

    print("File: " + queries_and_label[count][0])
    print("Prediction: " + prediction)
    print("")
    print("")