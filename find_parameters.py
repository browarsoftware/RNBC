"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/

Source codes for paper:
Rough neighborhood graph: a method for proximity modeling and data clustering

Examine range of clustering algorithm parameters to find the best configuration for certain dataset
Results are output to the file. You must select dataset you want to process.

Execution might take some time as well as exception messages may appear when evaluation scores are calculated.
These are caused by a lack of solution, usually because the neighborhood is too small.
Just stay calm and wait for the results :-)
"""

import warnings
from itertools import cycle, islice
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from numpy import genfromtxt
from RNBC import RNBC

########################################################################
# Scores

# Purity indicates the classification accuracy in the positive regions
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# Quality indicates the proportion of positive regions to all n objects
def quality_score(y_true, y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

def calculate_eval(Y, pred):
    np.set_printoptions(threshold=np.inf)
    ars = metrics.adjusted_rand_score(Y, pred)
    ami = metrics.adjusted_mutual_info_score(Y, pred)
    [ho, com, vme] = metrics.homogeneity_completeness_v_measure(Y, pred)
    fms = metrics.fowlkes_mallows_score(Y, pred)
    return [ars, ami, ho, com, vme, fms]


########################################################################
# Generate datasets

n_samples = 1500
seed = 0

blobs_very_nice_sepparated = datasets.make_blobs(n_samples=n_samples, random_state=8)

noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), np.zeros(n_samples).astype(int)

three_structure = rng.rand(int(n_samples / 3), 2)
three_structure2 = rng.rand(int(n_samples / 3), 2) * 2 + 1 + 0.1
three_structure3 = rng.rand(int(n_samples / 3), 2) * 4 + 2 + 1 + 0.1 + 0.2

l = np.zeros(int(n_samples / 3)).astype(int)
l2 = np.ones(int(n_samples / 3)).astype(int)
l3 = np.repeat(2, int(n_samples / 3)).astype(int)

three_structure = np.concatenate((three_structure, three_structure2, three_structure3)), np.concatenate((l, l2, l3))


# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.8, -0.8], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

WingNutDifficult = genfromtxt("WingNutDifficult.csv", delimiter=',',
                              skip_header=True)
WingNutDifficult = (WingNutDifficult[:, 0:2], WingNutDifficult[:, 2].astype(int))
target = genfromtxt("Target.csv", delimiter=',', skip_header=True)
target = (target[:, 0:2], target[:, 2].astype(int))

datasets = [
    blobs_very_nice_sepparated,
    noisy_circles,
    noisy_moons,
    aniso,
    blobs,
    three_structure,
    WingNutDifficult,
    target
]

datasets_names = ['Blobs\nSeparated',
    'Noisy\nCircles',
    'Noisy\nMoons',
    'Anisotropic',
    'Blobs',
    'Three\nDensities',
    'WingNut\nHorizontal',
    'Target\n(Outliers)'
]

# slect dataset, some dataset should not be scaled - see run.py
# example on dataset "5"
dataset = datasets[5]
X, Y = dataset
X = StandardScaler().fit_transform(X)

# iterate all parameters and output scores to file
for n in range(2, 30):
    for t in [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]:
        for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            try:
                print("k=" + str(n))
                classes = RNBC(X, n, t=t, alpha=alpha, use_lower=True)
                classes = classes - 1
                file1 = open("RNBC_lower.txt", "a")
                [ars, ami, ho, com, vme, fms] = calculate_eval(Y, classes)
                file1.write(str(n) + "," + str(t) + "," + str(alpha) + "," +
                            str(ars) + "," + str(ami) + "," + str(ho) + "," + str(com) + "," + str(vme) + "," + str(
                    fms) + "\n")
                file1.close()
            except:
                print("An exception occurred")


# iterate all parameters and output scores to file
for n in range(2, 30):
    for t in [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]:
        for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            try:
                print("k=" + str(n))
                classes = RNBC(X, n, t=t, alpha=alpha, use_lower=False)
                classes = classes - 1
                file1 = open("RNBC_upper.txt", "a")
                [ars, ami, ho, com, vme, fms] = calculate_eval(Y, classes)
                file1.write(str(n) + "," + str(t) + "," + str(alpha) + "," +
                            str(ars) + "," + str(ami) + "," + str(ho) + "," + str(com) + "," + str(vme) + "," + str(
                    fms) + "\n")
                file1.close()
            except:
                print("An exception occurred")

for eps in list(np.arange(0.000001, 10, .01)):
    for min_samples in range(2,8):
        print("eps=" + str(eps))
        algorithm = cluster.DBSCAN(eps=eps, min_samples=min_samples)
        #algorithm = cluster.DBSCAN(eps=0.11, min_samples=3)
        algorithm.fit(X)
        y_pred = algorithm.labels_.astype(int)
        file1 = open("DBSCAN.txt", "a")

        [ars, ami, ho, com, vme, fms] = calculate_eval(Y, y_pred)
        file1.write(str(eps) + "," + str(min_samples) + "," +
                    str(ars) + "," + str(ami) + "," + str(ho) + "," + str(com) + "," + str(vme) + "," + str(
            fms) + "\n")
        file1.close()
