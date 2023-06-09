from collections import Counter
import math


def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []

    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[1:], query)

        # 3.2 Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))

    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][0] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality.")

    squared_diff = [(x - y) ** 2 for x, y in zip(vector1, vector2)]
    sum_squared_diff = sum(squared_diff)
    distance = math.sqrt(sum_squared_diff)

    return distance


def manhattan_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality.")

    distance = sum(abs(x - y) for x, y in zip(vector1, vector2))

    return distance