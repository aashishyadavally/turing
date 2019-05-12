"""
This script implements clustering algorithms like K-Means Clustering Algorithm

Author:
-------
Aashish Yadavally
"""

import sys
import random
import numpy as np


def distance(x, y, distance_type):
    """
    Calculates distance between two points, either Euclidean, or Manhattan

    Args:
        x (list):
            Point in dataset
        y (list):
            Point in dataset
        distance_type (str):
            Type of distance to be estimated

    Returns:
        Distance between two points
    """
    if distance_type == 'euclidean':
        return np.linalg.norm(np.asarray(x) - np.asarray(y), ord=2)
    elif distance_type == 'manhattan':
        return np.linalg.norm(np.asarray(x) - np.asarray(y), ord=1)
    else:
        print('Invalid distance type.')
        sys.exit()


def find_centroid(items):
    """
    Calculates centroid of clusters

    Args:
        items (list):
            List of points belonging to a cluster

    Returns:
        centroid (list):
            Centroid of given list of points
    """
    length = len(items)
    num_axes = len(items[0])
    centroid = [(np.sum(np.asarray(items)[:, i]))/length
                for i in range(num_axes)]
    return centroid


class KMeans:
    """
    Performs K-Means Clustering method on the given dataset
    """
    def __init__(self, k, points, max_iter=1000, distance_type='euclidean'):
        """
        Initializes KMeans class

        Args:
            k (int):
                Number of clusters
            points (list):
                List of points in space
            max_iter (int):
                Maximum number of iterations to run the model for
        """
        self.points = points
        length = {len(point) for point in self.points}
        if len(length) != 1:
            print("Input Error: All dataset points don't have same number\
                  of attributes.")
            sys.exit()
        self.k = k
        self.distance_type = distance_type
        self.cluster_labels = [0] * len(self.points)
        self.cluster_centers = self.initialize_cluster_centers(len(self.points[0]))
        for iteration in range(max_iter):
            previous_centers = self.cluster_centers
            self.print_epoch(iteration)
            self.cluster_assignment()
            self.update()
            if self.check_stopping_criterion(previous_centers):
                print("Stopping criterion met, model training is being\
                      stopped.")
                break

    def initialize_cluster_centers(self, n):
        """
        Initializes random clusters to all points in dataset in the
        first iteration

        Args:
            n (int):
                Number of features in dataset

        Returns:
            (list):
                List of 'k' randomly generated cluster centers
        """
        return [random.sample(range(100 * n), n) for cluster_center
                in range(self.k)]

    def cluster_assignment(self):
        """
        Assigns clusters to data
        """
        for index, point in enumerate(self.points):
            distances = [distance(point, cluster_point, self.distance_type)
                         for cluster_point in self.cluster_centers]
            self.cluster_labels[index] = np.argmin(np.asarray(distances))

    def update(self):
        """
        Updates centroid
        """
        for i in range(self.k):
            items = [self.points[j] for j in range(len(self.cluster_labels))
                     if self.cluster_labels[j] == i]
            if items != []:
                self.cluster_centers[i] = find_centroid(items)

    def check_stopping_criterion(self, previous_centers):
        """
        Check if the cluster labels are changing or not. Can stop the model
        training if the cluster labels do not change.

        Args:
            previous_centers (list):
                Cluster center assignment before cluster update

        Returns:
            (bool):
                True, if cluster center assignment doesn't change, else False
        """
        return bool(previous_centers == self.cluster_labels)

    def print_epoch(self, iteration):
        """
        Prints model information

        Args:
            iteration (int):
                Number of iteration
        """
        print('=' * 20)
        print(f'ITERATION #{iteration}:')
        print('Printing cluster centers: ')
        print(self.cluster_centers)
        print('\n')
        print('Printing cluster labels: ')
        print(self.cluster_labels)
        print('=' * 20)
