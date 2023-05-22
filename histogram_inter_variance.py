import matplotlib.pyplot as plt
import numpy as np
import knn
import csv_utils

def calculate_and_plot_inter_class_variation(data):
    inter_class_variations = []

    # Generate a list of unique classes in the data
    unique_classes = list(set([row[0] for row in data]))

    # For each unique class
    for i in range(len(unique_classes)):
        for j in range(i + 1, len(unique_classes)):
            class_i_data = [row for row in data if row[0] == unique_classes[i]]
            class_j_data = [row for row in data if row[0] == unique_classes[j]]

            if class_i_data and class_j_data:
                inter_class_variations.append(knn.euclidean_distance(class_i_data[0][1:], class_j_data[0][1:]))

    # Plot the histogram
    plt.hist(inter_class_variations, bins=50, alpha=0.7, color='b', density=True)
    plt.xlabel('Inter-Class Variation')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Inter-Class Variation')
    plt.grid(True)
    plt.show()

    return np.mean(inter_class_variations)

trained_data = csv_utils.data_from_csv_with_label('./vectors/vectors_training.csv')
inter_class_variation = calculate_and_plot_inter_class_variation(trained_data)

print("Average inter-class variation:", inter_class_variation)
