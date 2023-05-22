import csv

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

def append_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)