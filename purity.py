import numpy as np


def purity_calculation(clusters):
    # purity
    purity = []
    for cluster in clusters:
        cluster = np.array(cluster)
        cluster = cluster / float(cluster.sum())
        p = cluster.max()
        purity += [p]

    counts = np.array([sum(c) for c in clusters])
    coeffs = counts / float(counts.sum())

    pur = (coeffs * purity).sum()
    return pur
