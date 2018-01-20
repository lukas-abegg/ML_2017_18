from typing import List
import random

import math
import numpy as np
import itertools


class Centroid:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def change_centroid(self, x, y):
        self.x = x
        self.y = y


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid: Centroid = None

    def assign_centroid(self, centroid: Centroid):
        self.centroid = centroid


def read_dataset() -> List[Point]:
    points: List[Point] = []
    path = "./data/dataset"
    with open(path, "r", encoding='latin-1') as f:
        for line in f.readlines():
            entry = line.split()
            points.append(Point(float(entry[1]) / 10, float(entry[2]) / 100))
    return points


# place cluster centers ck randomly for all k = 1, ... ,K
def init_cluster_centroids(num_clusters) -> List[Centroid]:
    centroids = [Centroid(random.uniform(0, 1), random.uniform(0, 1)) for _ in itertools.repeat(None, num_clusters)]
    return centroids


def get_distance(p: Point, c: Centroid) -> float:
    # euclidean distance
    square_diff_x = pow((p.x - c.x), 2)
    square_diff_y = pow((p.y - c.y), 2)
    accumulated_diff = square_diff_x + square_diff_y
    distance = math.sqrt(accumulated_diff)
    return distance


def find_smallest_distance_to_centroid(p: Point, centroids: List[Centroid]) -> Centroid:
    distances = []
    for centroid in centroids:
        distances.append(get_distance(p, centroid))
    index = distances.index(min(distances))
    return centroids[index]


def initial_assign_points_to_clusters(points: List[Point], centroids: List[Centroid]):
    for p in points:
        p.assign_centroid(find_smallest_distance_to_centroid(p, centroids))


def find_new_centroid(points: List[Point]) -> (int, int):
    list_x = []
    list_y = []
    for p in points:
        list_x.append(p.x)
        list_y.append(p.y)

    centroid_x = np.mean(list_x)
    centroid_y = np.mean(list_y)
    return centroid_x, centroid_y


# assign data point xi to nearest cluster center
def assign_points_to_clusters(points: List[Point], centroids: List[Centroid]):
    changes = False
    for p in points:
        nearest_centroid = find_smallest_distance_to_centroid(p, centroids)
        if p.centroid != nearest_centroid:
            changes = True
            p.assign_centroid(nearest_centroid)
    return changes


# recompute cluster centers
def recompute_centroids(points: List[Point], centroids: List[Centroid]):
    for centroid in centroids:
        cluster_points = []
        for p in points:
            if p.centroid == centroid:
                cluster_points.append(p)
        if len(cluster_points) > 0:
            x, y = find_new_centroid(cluster_points)
            centroid.change_centroid(x, y)


def init_clusters() -> (List[Point], List[Centroid]):
    points = read_dataset()
    centroids = init_cluster_centroids(6)
    initial_assign_points_to_clusters(points, centroids)
    return points, centroids


# reiterate until there are no changes in assignments
def fit_clusters(points: List[Point], centroids: List[Centroid]) -> (List[Point], List[Centroid]):
    changes = True
    i = 0
    while changes:
        recompute_centroids(points, centroids)
        changes = assign_points_to_clusters(points, centroids)
        i += 1
        print(i)
    return points, centroids


points, centroids = init_clusters()
print("--- Clusters initialized ---")
fit_clusters(points, centroids)
print("--- Clusters fitted ---")
# plot_clusters()
