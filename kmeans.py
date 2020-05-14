import argparse
import os
import math
import numpy as np


def read_file(file_path):
    if not os.path.exists(file_path):
        raise Exception("File not found: " + file_path)

    X = []
    y = []
    is_header_skipped = False
    with open(file_path, "r") as f:
        for line in f:
            if not is_header_skipped:
                is_header_skipped = True
                continue

            cols = line.split(",")
            cols = [col.strip() for col in cols]
            X.append(np.array(cols[:-1], np.float))
            y.append(np.array(cols[-1:], np.float))

    return np.array(X), np.array(y)


def cluster():
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--distance', type=str, default='Euclidean')
    parser.add_argument('--k', type=int, default=2)

    args = parser.parse_args()

    if not args.dataset:
        parser.error('please specify --dataset with corresponding path to dataset')

    if args.distance not in ['Manhattan', 'Euclidean']:
        print("Invalid distance measure")
        parser.error('please specify --distance with either Manhattan or Euclidean')

    X, y = read_file(args.dataset)
    print("Read Training sequence and label set: {0} {1}".format(str(X.shape), str(y.shape)))
