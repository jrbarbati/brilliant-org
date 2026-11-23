def _inertia():
    """
    Calculates the precision of a particular k-means iteration
    inertia = sum of squared distance between center and each point in cluster
    lower inertia = more precision
    """
    pass


def _total_inertia():
    """
    Calculates the total inertia of a particular k-means iteration
    lower total inertia = better overall clustering
    """
    pass


def _optimize():
    """
    Finds the minimum of a function of the data
    k-means: function is the total inertia
    """
    pass

def _start(k: int):
    """Selects k numbers of centers from which to start the k-means algorithm."""
    pass


def _choose_k() -> int:
    """
    Decides which k is best for the data

    Decision - elbow of the graph made by the inertia of 0 <= k <= n
    Graph looks like an exponential decaying function where the y-axis is the inertia and the x-axis is k
    Generally, more clusters = lower inertia.
    The decay slows as k increases

    :return: k
    """
    return 1
