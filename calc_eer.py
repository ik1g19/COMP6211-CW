import numpy as np
import matplotlib.pyplot as plt
import knn
import csv_utils


def calculate_eer(fars, frrs, thresholds):
    # Compute the difference between the FAR and the FRR for each threshold
    diffs = fars - frrs

    # Find the index where the difference changes from negative to positive, which indicates an EER
    for i in range(1, len(diffs)):
        if diffs[i - 1] < 0 and diffs[i] >= 0:
            return thresholds[i]
    return thresholds[-1]  # return the last threshold if no EER was found


trained_data = csv_utils.data_from_csv_with_label('./vectors/vectors_training.csv')
queries = csv_utils.data_from_csv('./vectors/vectors_testing.csv')
answers = ["021z001","021z002","021z003","021z004","021z005","021z006","021z007","021z008","021z009","021z010","024z011"]

max_distance = 30.0
num_thresholds = 100
thresholds = np.linspace(0, max_distance, num_thresholds)  # max_distance and num_thresholds are to be set appropriately
fars = []
frrs = []

for threshold in thresholds:
    fa = 0  # false accepts
    fr = 0  # false rejects
    total_positives = 0
    total_negatives = 0

    for count, query in enumerate(queries):
        k_nearest_distances_and_indices, prediction = knn.knn(trained_data, query, k=1,
                                                              distance_fn=knn.euclidean_distance, choice_fn=knn.mode)

        # The smallest distance corresponds to the most likely prediction
        smallest_distance = k_nearest_distances_and_indices[0][0]

        is_match = ((answers[count] + "ps.jpg") == prediction)

        if is_match:
            total_positives += 1
            if smallest_distance > threshold:
                fr += 1
        else:
            total_negatives += 1
            if smallest_distance <= threshold:
                fa += 1

    fars.append(fa / total_negatives if total_negatives > 0 else 0)
    frrs.append(fr / total_positives if total_positives > 0 else 0)

fars = np.array(fars)
frrs = np.array(frrs)
eer = calculate_eer(fars, frrs, thresholds)

print("Equal Error Rate (EER) is at threshold: " + str(eer))

# Plotting the FAR and FRR against thresholds
plt.figure(figsize=(10, 6))
plt.plot(thresholds, fars, label='FAR')
plt.plot(thresholds, frrs, label='FRR')
plt.scatter(eer, fars[np.argmin(abs(thresholds - eer))], color='red')  # mark EER on FAR curve
plt.scatter(eer, frrs[np.argmin(abs(thresholds - eer))], color='red')  # mark EER on FRR curve
plt.xlabel('Thresholds')
plt.ylabel('Error Rates')
plt.title('FAR and FRR at different thresholds')
plt.legend()
plt.grid(True)
plt.show()
