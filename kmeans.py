import argparse
import math
import os
import numpy as np


def read_file(file_path):
    if not os.path.exists(file_path):
        raise Exception("File not found: " + file_path)

    dataset = []
    is_header_skipped = False
    with open(file_path, "r") as f:
        for line in f:
            if not is_header_skipped:
                is_header_skipped = True
                continue

            cols = line.split(",")
            cols = [col.strip() for col in cols]
            dataset.append({"x": np.array([float(col) for col in cols[:-1]]), "y": float(cols[-1])})

    return dataset


def euclidean_distance(x_1, x_2):
    diff = x_1 - x_2
    diff_square = np.square(diff)
    return math.sqrt(np.sum(diff_square))


def manhattan_distance(x_1, x_2):
    absolute_diff = np.absolute(x_1 - x_2)
    return np.sum(absolute_diff)


"""
Citing paper: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
https://en.wikipedia.org/wiki/K-means%2B%2B
Below is the algorithm to choose randomly initial centroids for k-means is taken from above paper
"""
def get_random_centroids(dataset, k):
    centroids = []
    first_centroid_choice = np.random.choice(len(dataset))  # Select first centroid from uniform distribution
    centroids.append(dataset[first_centroid_choice])
    selected_centroids = {first_centroid_choice}

    for j in range(k - 1):
        probability = []
        for i in range(len(dataset)):
            # Do not consider the previously seleected centroid for probability calculation
            if i not in selected_centroids:
                instance = dataset[i]
                # Find out which one of the centroid is closest to instance 'x'
                closest_centroid = min(centroids, key=lambda item: euclidean_distance(instance["x"], item["x"]))
                probability.append(euclidean_distance(instance["x"], closest_centroid["x"]) ** 2)
            else:
                probability.append(0.0)

        probability = np.array(probability)
        probability = probability / np.sum(probability)  # normalize to get probability
        next_centroid = np.random.choice(a=dataset, p=probability)

        centroids.append(next_centroid)

    return centroids


def is_converge(old_centroids, new_centroids):
    if new_centroids is None:  # Return if the new-centroids are not been computed, i.e. this is first loop-pass
        return False

    """
        The k-Means algorithm converges when there are no changes in cluster assignments
        That is all the members remain their previous cluster
        In such a case, the centroids of the cluster remain same and do not change
        Hence, the newly computed centroids will be same as previous one
        Here we check whether all the corresponding centroid vectors of the new and old centroids are same or not
        If all the old and corresponding newly computed centroids matched then we return True
    """
    return all([np.isclose(old_centroids[i]['x'], new_centroids[i]['x']).all() for i, centroid in enumerate(old_centroids)])


def cluster(dataset, centroids, k, distance_metric, max_iterations=1000):
    clusters = {}

    t = 0
    new_centroids = None
    while not is_converge(centroids, new_centroids):
        centroids = new_centroids if new_centroids is not None else centroids  # Use centroids computed in previous loop-pass

        for index, centroid in enumerate(centroids):
            clusters[index] = []
            centroid['label'] = index

        for instance in dataset:
            # Using the distance_metric received to compute closest centroid
            closest_centroid = min(centroids, key=lambda item: distance_metric(item['x'], instance['x']))
            cluster_label = closest_centroid['label']
            clusters[cluster_label].append(instance)

        new_centroids = []
        for cluster_label, centroid in enumerate(centroids):  # the index of centroid in list is indirectly label itself
            members = clusters[cluster_label]
            size = float(len(members))

            new_centroid = np.zeros(members[0]['x'].shape[0])  # zero-initialize array of size of feature-dimension
            for member in members:
                new_centroid = new_centroid + member['x']

            new_centroid = new_centroid / size
            new_centroids.append({'x': new_centroid, 'label': cluster_label})

        t += 1
        if t > max_iterations:
            break

    print("Total iterations took to converge: {0}".format(str(t)))

    return clusters, centroids


def get_positive_diagnosis_percentages(clusters):
    percentages = []
    for cluster_label, members in clusters.items():
        count = sum([1.0 if member['y'] == 1.0 else 0.0 for member in members])
        percentages.append(count * 100.0 / len(members))

    return percentages


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--distance', type=str, default='Euclidean')
    parser.add_argument('--k', type=int, help='Specify k parameter')

    args = parser.parse_args()

    if not args.dataset:
        parser.error('please specify --dataset with corresponding path to dataset')

    distance_metric = None
    if args.distance is None or args.distance not in ['Manhattan', 'Euclidean']:
        print("Invalid distance measure")
        parser.error('please specify --distance with either Manhattan or Euclidean')
    else:
        distance_metric = euclidean_distance if args.distance == "Euclidean" else manhattan_distance

    dataset = read_file(args.dataset)
    print("Dataset read, size: {0}".format(str(len(dataset))))

    random_centroids = get_random_centroids(dataset, args.k)
    clusters, centroids = cluster(dataset, random_centroids, args.k, distance_metric, max_iterations=1000)
    positive_percentages = get_positive_diagnosis_percentages(clusters)

    print("Cluster sizes: " + ", ".join(str(len(members)) for members in clusters.values()))
    print(str("Positive diagnosis %: " + ", ".join([str(p) for p in positive_percentages])))
