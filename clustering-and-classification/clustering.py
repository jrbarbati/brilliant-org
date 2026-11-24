import k_means

import sklearn

from pprint import pprint

def main():
    wine_data_set = sklearn.datasets.load_wine()

    clusters = k_means.cluster(wine_data_set.data)

    pprint(clusters)
    print(len(clusters))


if __name__ == "__main__":
    main()
