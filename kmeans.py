import argparse
import os
import random
import numpy as np
import math


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


def cluster(dataset, centroids, k, iterations=1000):
    for t in range(1000):
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--distance', type=str, default='Euclidean')
    parser.add_argument('--k', type=int, help='Specify k parameter')

    args = parser.parse_args()

    if not args.dataset:
        parser.error('please specify --dataset with corresponding path to dataset')

    if args.distance not in ['Manhattan', 'Euclidean']:
        print("Invalid distance measure")
        parser.error('please specify --distance with either Manhattan or Euclidean')

    dataset = read_file(args.dataset)
    print("Dataset read, size: {0}".format(str(len(dataset))))

    random_centroids = get_random_centroids(dataset, args.k)
