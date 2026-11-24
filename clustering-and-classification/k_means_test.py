import unittest
from typing import List

from mockito import when, any
from parameterized import parameterized

import k_means
from k_means import _find_minimums, _find_maximums, _subtract_mins, _divide_maxes, _distance, _inertia, _total_inertia, _cluster_for_centroids, _reassign_centroids


class KMeansTest(unittest.TestCase):
    def test__find_minimums(self):
        test_data = [
            [1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1],
            [4, 5, 6, 7, 1, 2, 3],
            [3, 4, 5, 6, 7, 1, 2],
        ]

        minimums = _find_minimums(test_data)

        self.assertEqual(minimums, [1, 2, 3, 4, 1, 1, 1])

    def test__subtract_mins(self):
        when(k_means)._find_minimums(any(List)).thenReturn([1, 2, 3, 4, 1, 1, 1])

        result = _subtract_mins([
            [1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1],
            [4, 5, 6, 7, 1, 2, 3],
            [3, 4, 5, 6, 7, 1, 2],
        ])

        self.assertEqual(result, [
            [0, 0, 0, 0, 4, 5, 6],
            [6, 4, 2, 0, 2, 1, 0],
            [3, 3, 3, 3, 0, 1, 2],
            [2, 2, 2, 2, 6, 0, 1],
        ])

    def test__find_maximums(self):
        test_data = [
            [0, 0, 0, 0, 4, 5, 6],
            [6, 4, 2, 0, 2, 1, 0],
            [3, 3, 3, 3, 0, 1, 2],
            [2, 2, 2, 2, 6, 0, 1],
        ]

        maximums = _find_maximums(test_data)

        self.assertEqual(maximums, [6, 4, 3, 3, 6, 5, 6])

    def test__divide_maxes(self):
        when(k_means)._find_maximums(any(List)).thenReturn([6, 4, 3, 3, 6, 5, 6])

        result = _divide_maxes([
            [0, 0, 0, 0, 4, 5, 6],
            [6, 4, 2, 0, 2, 1, 0],
            [3, 3, 3, 3, 0, 1, 2],
            [2, 2, 2, 2, 6, 0, 1],
        ])

        self.assertEqual(result, [
            [0,   0,    0,    0,   2/3, 1,   1  ],
            [1,   1,    2/3,  0,   1/3, 0.2, 0  ],
            [0.5, 0.75, 1,    1,   0,   0.2, 1/3],
            [1/3, 0.5,  2/3,  2/3, 1,   0,   1/6],
        ])

    @parameterized.expand([
        ([0, 1], [0, 0], 1),
        ([1, 1], [0, 0], 2 ** 0.5),
        ([5], [2], 3),  # 1D
        ([-3], [4], 7),  # 1D, negative
        ([3, 4], [0, 0], 5),  # 2D, 3-4-5
        ([1, 2, 2], [0, 0, 0], 3),  # 3D
        ([-1, -2, -2], [1, 2, 2], 6),  # 3D, negative
        ([1, 1, 1, 1], [0, 0, 0, 0], 2),  # 4D
        ([2, 3, 6, 0, 0], [0, 0, 0, 0, 0], 7),  # 5D
        ([1, 1, 1, 1, 1, 1, 1, 1, 1], [0] * 9, 3),  # 9D
    ])
    def test__distance(self, point, centroid, expected):
        d = _distance(point, centroid)

        self.assertEqual(d, expected)

    def test__inertia(self):
        when(k_means)._distance(any(List), any(List)).thenReturn(3**0.5, 12**0.5, 27**0.5, 48**0.5)

        inertia = _inertia([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [0, 0, 0])

        self.assertEqual(inertia, sum([3**0.5, 12**0.5, 27**0.5, 48**0.5]))

    def test__total_inertia(self):
        when(k_means)._inertia(any(List), any(List)).thenReturn(sum([3**0.5, 12**0.5, 27**0.5, 48**0.5]))
        total_inertia = _total_inertia(
            {
                0: [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                1: [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                2: [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
            }
            ,
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )
        self.assertAlmostEqual(total_inertia, sum([3**0.5, 12**0.5, 27**0.5, 48**0.5] * 3), delta=0.0000001)

    @parameterized.expand([
        ("single_centroid_gets_everything",
         [[0, 0], [5, 5], [-3, 2]],
         [[1, 1]],
         {0: [[0, 0], [5, 5], [-3, 2]]}),
        ("two_separated_blobs_1d",
         [[1], [2], [3], [10], [11], [12]],
         [[2], [11]],
         {0: [[1], [2], [3]], 1: [[10], [11], [12]]}),
        ("two_separated_blobs_2d",
         [[0, 0], [0, 1], [10, 10], [10, 11]],
         [[0, 0], [10, 10]],
         {0: [[0, 0], [0, 1]], 1: [[10, 10], [10, 11]]}),
        ("three_clusters_3d",
         [[0, 0, 0], [1, 0, 0], [10, 10, 10], [10, 11, 10], [-10, -10, -10], [-9, -10, -10]],
         [[0, 0, 0], [10, 10, 10], [-10, -10, -10]],
         {0: [[0, 0, 0], [1, 0, 0]], 1: [[10, 10, 10], [10, 11, 10]], 2: [[-10, -10, -10], [-9, -10, -10]]}),
        ("interleaved_input_order_preserved",
         [[10], [0], [11], [1], [12], [2]],
         [[1], [11]],
         {0: [[0], [1], [2]], 1: [[10], [11], [12]]}),
        ("point_on_centroid",
         [[4, 4], [9, 9]],
         [[4, 4], [9, 9]],
         {0: [[4, 4]], 1: [[9, 9]]}),
        ("negative_coordinates",
         [[-5, -5], [-4, -6], [5, 5], [6, 4]],
         [[-5, -5], [5, 5]],
         {0: [[-5, -5], [-4, -6]], 1: [[5, 5], [6, 4]]}),
        ("tie_goes_to_lowest_index",
         [[5]],
         [[0], [10]],
         {0: [[5]], 1: []}),
        ("tie_in_2d_goes_to_lowest_index",
         [[1, 0]],
         [[0, 0], [2, 0], [1, 5]],
         {0: [[1, 0]], 1: [], 2: []}),
        ("far_centroid_gets_nothing_but_key_kept",
         [[0, 0], [1, 1], [2, 2]],
         [[1, 1], [100, 100]],
         {0: [[0, 0], [1, 1], [2, 2]], 1: []}),
        ("duplicate_points_all_kept",
         [[1, 1], [1, 1], [9, 9], [9, 9], [9, 9]],
         [[0, 0], [10, 10]],
         {0: [[1, 1], [1, 1]], 1: [[9, 9], [9, 9], [9, 9]]}),
        ("nearest_is_euclidean_not_axis_aligned",
         # (3,3): Euclidean c0 = sqrt(18) ~ 4.24 < c1 = 5 -> c0
         # Manhattan would pick c1 (5 < 6), so this guards against the wrong metric
         [[3, 3]],
         [[0, 0], [3, -2]],
         {0: [[3, 3]]}),
    ])
    def test__cluster_for_centroids(self, _name, data, centroids, expected):
        clusters = _cluster_for_centroids(data, centroids)

        self.assertEqual(set(clusters.keys()), set(range(len(centroids))))

        for index in range(len(centroids)):
            self.assertEqual(clusters[index], expected.get(index, []))

    def test__cluster_for_centroids_partitions_all_points(self):
        data = [[0, 0], [1, 2], [8, 8], [9, 7], [4, 4], [-6, 3], [7, -7]]
        centroids = [[0, 0], [8, 8], [-6, 3], [7, -7]]

        clusters = _cluster_for_centroids(data, centroids)

        assigned = [point for points in clusters.values() for point in points]
        self.assertEqual(sorted(assigned), sorted(data))

    # New centroid k = coordinate-wise mean of the points assigned to cluster k.
    @parameterized.expand([
        ("single_cluster_single_point", {0: [[3, 4]]}, [[3, 4]]),
        ("single_cluster_1d", {0: [[1], [2], [6]]}, [[3]]),
        ("single_cluster_2d", {0: [[0, 0], [2, 4]]}, [[1, 2]]),
        ("identical_points", {0: [[5, 5], [5, 5], [5, 5]]}, [[5, 5]]),
        ("negative_values_cancel", {0: [[-2, -4], [2, 4]]}, [[0, 0]]),
        ("non_integer_mean", {0: [[0, 0], [0, 1], [1, 1]]}, [[1 / 3, 2 / 3]]),
        ("3d_line", {0: [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]}, [[2.5, 2.5, 2.5]]),
        ("two_clusters_separated", {
            0: [[0, 0], [0, 2], [2, 0], [2, 2]],
            1: [[10, 10], [10, 12], [12, 10], [12, 12]],
        }, [[1, 1], [11, 11]]),
        ("three_clusters_uneven_sizes", {
            0: [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            1: [[10, 0, -10]],
            2: [[0, 6, 0], [0, 0, 6]],
        }, [[2.5, 2.5, 2.5], [10, 0, -10], [0, 3, 3]]),
        ("outlier_pulls_mean", {0: [[0], [0], [0], [12]]}, [[3]]),
    ])
    def test__reassign_centroids(self, _name, clusters, expected):
        new_centroids = _reassign_centroids(clusters)

        self.assertEqual(len(new_centroids), len(expected))

        for actual_centroid, expected_centroid in zip(new_centroids, expected):
            self.assertEqual(len(actual_centroid), len(expected_centroid))

            for actual, exp in zip(actual_centroid, expected_centroid):
                self.assertAlmostEqual(actual, exp, places=9)

    def test__reassign_centroids_does_not_mutate_clusters(self):
        clusters = {0: [[0, 0], [2, 4]], 1: [[10, 10]]}

        _reassign_centroids(clusters)

        self.assertEqual(clusters, {0: [[0, 0], [2, 4]], 1: [[10, 10]]})
