import random
from typing import List, Dict, Tuple


# TODO: Refactor this file to use Point instead of List everywhere
class Point:
    def __init__(self, values: List) -> None:
        self.values = values
        self.normalized = [0] * len(values)
        self.cluster_id = -1


def _find(data: List, op) -> List:
    result = []

    for datum in data:
        for i, value in enumerate(datum):
            if len(result) < i + 1:
                result.append(value)

            result[i] = op(value, result[i])

    return result


def _perform(data: List, modifier, op) -> List:
    modified = modifier(data)

    for datum in data:
        for i, value in enumerate(datum):
            if len(modified) < i + 1:
                raise Exception("Invalid data.")

            datum[i] = op(value, modified[i])

    return data


def _find_minimums(data: List) -> List:
    return _find(data, min)


def _find_maximums(data: List) -> List:
    return _find(data, max)


def _subtract_mins(data: List) -> List:
    return _perform(data, _find_minimums, lambda a, b: a - b)


def _divide_maxes(data: List) -> List:
    return _perform(data, _find_maximums, lambda a, b: a / b)


def _normalize(data: List) -> List:
    """
    Normalizes the data by applying min-max scaling
    Subtract all data in one axis by the smallest in that axis then divide by the largest in that axis (post-subtraction)
    """
    return _divide_maxes(_subtract_mins(data))


def _distance(point: List, centroid: List) -> float:
    return sum([(p - c) ** 2 for p, c in zip(point, centroid)]) ** 0.5


def _inertia(single_cluster: List, centroid: List) -> float | int:
    """
    Calculates the precision of a particular k-means iteration
    inertia = sum of squared distance between center and each point in cluster
    lower inertia = more precision
    """
    return sum([_distance(point, centroid) for point in single_cluster])


def _total_inertia(clustered_data: Dict[int, List], centroids: List[List]) -> float | int:
    """
    Calculates the total inertia of a particular k-means iteration
    lower total inertia = better overall clustering
    """
    return sum([_inertia(clustered_data[c_key], centroids[c_key]) for c_key in clustered_data])


def _cluster_for_centroids(normalized_data: List, centroids: List) -> Dict[int, List]:
    """
    New centroids sent in - need to re-cluster the normalized data
    :param normalized_data:
    :param centroids:
    :return: new clusters based on the new centroids
    """
    clusters = {i: list() for i in range(len(centroids))}

    for point in normalized_data:
        min_index = -1
        min_ = float('inf')

        for i, centroid in enumerate(centroids):
            d = _distance(point, centroid)

            if d < min_:
                min_index = i
                min_ = d

        clusters[min_index].append(point)

    return clusters


def _reassign_centroids(clusters: Dict[int, List]) -> List:
    centroids = [list()] * len(clusters)

    for c_key, c in clusters.items():
        centroids[c_key] = [0] * len(c[0])

        for point in c:
            for i, value in enumerate(point):
                centroids[c_key][i] += value

        centroids[c_key] = [cent / len(c) for cent in centroids[c_key]]

    return centroids


def _optimize(normalized_data: List, k: int) -> Tuple[Dict[int, List], List]:
    """
    Finds the minimum of a function of the data
    k-means: function is the total inertia
    This optimizes the centroids each for each k-means iteration

    :return: Tuple where first element is the clusters based on the final centroids found and the second element is the
    final centroids
    """
    centroids = _random_centroids(normalized_data, k)
    clusters = _cluster_for_centroids(normalized_data, centroids)

    while True:
        old_centroids = centroids
        centroids = _reassign_centroids(clusters)

        if old_centroids == centroids:
            # Centroids converged done with optimization
            break

        clusters = _cluster_for_centroids(normalized_data, centroids)

    return clusters, centroids


def _random_centroids(normalized_data: List, k: int) -> List[List]:
    """Selects k numbers of centroids from which to start the k-means algorithm."""
    return [
        [random.random() for _ in range(len(normalized_data[0]))]
        for _ in range(k)
    ]


def _cluster(normalized_data: List) -> Dict[int, List]:
    """
    Decides which k is best for the data

    Decision - elbow of the graph made by the inertia of 0 <= k <= n
    Graph looks like an exponential decaying function where the y-axis is the inertia and the x-axis is k
    Generally, more clusters = lower inertia.
    The decay slows as k increases

    :return: k
    """
    k = 2
    clustered_data, centroids = _optimize(normalized_data, 1)
    total_inertia = _total_inertia(clustered_data, centroids)

    while True:
        old_inertia = total_inertia
        old_clusters, old_centroids = clustered_data, centroids

        clustered_data, centroids = _optimize(normalized_data, k)
        total_inertia = _total_inertia(clustered_data, centroids)

        if (old_inertia - total_inertia) / old_inertia < 0.07:
            break

    return old_clusters


def cluster(data: List) -> Dict[int, List]:
    return _cluster(_normalize(data))
