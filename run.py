"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/

Source codes for paper:
Rough neighborhood graph: a method for proximity modeling and data clustering

Rough neighborhood-based clustering evaluation
"""

import warnings
from itertools import cycle, islice
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from numpy import genfromtxt
from RNBC import RNBC
import matplotlib.pyplot as plt

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

########################################################################
# Evaluate methods
def evaluate_Spectral():
    res = []

    dataset = datasets[0]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.SpectralClustering(
        n_clusters=3,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[1]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.SpectralClustering(
        n_clusters=2,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[2]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.SpectralClustering(
        n_clusters=2,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[3]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.SpectralClustering(
        n_clusters=3,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[4]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.SpectralClustering(
        n_clusters=3,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[5]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.SpectralClustering(
        n_clusters=3,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[6]
    X, y = dataset
    algorithm = cluster.SpectralClustering(
        n_clusters=2,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[7]
    X, y = dataset
    algorithm = cluster.SpectralClustering(
        n_clusters=2,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    return res

def evaluate_Gaussian():
    res = []

    dataset = datasets[0]
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    algorithm = mixture.GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))

    dataset = datasets[1]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = mixture.GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))

    dataset = datasets[2]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = mixture.GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))

    dataset = datasets[3]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = mixture.GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))

    dataset = datasets[4]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = mixture.GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))

    dataset = datasets[5]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = mixture.GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))

    dataset = datasets[6]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = mixture.GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))


    dataset = datasets[7]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = mixture.GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    res.append((X, y_pred, y))

    return res

def evaluate_MiniBatchKMeans():
    res = []

    dataset = datasets[0]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=3,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[1]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=2,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[2]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=2,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[3]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=3,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[4]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=3,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[5]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=3,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[6]
    X, y = dataset
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=2,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[7]
    X, y = dataset
    algorithm = cluster.MiniBatchKMeans(
        n_clusters=2,
        random_state=0,
    )
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    return res

def evaluate_AgglomerativeClustering():
    res = []

    dataset = datasets[0]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=3, linkage="ward")
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))
    dataset = datasets[1]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=2, linkage="ward")

    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))
    dataset = datasets[2]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=2, linkage="ward")
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[3]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=3, linkage="ward")
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))
    dataset = datasets[4]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=3, linkage="ward")

    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))
    dataset = datasets[5]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=3, linkage="ward")
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[6]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=2, linkage="ward")
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[7]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.AgglomerativeClustering(
        n_clusters=2, linkage="ward")
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    return res

def evaluate_DBSCAN():
    res = []

    dataset = datasets[0]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.DBSCAN(eps=0.3, min_samples=5)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[1]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.DBSCAN(eps=.3, min_samples=5)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))
    
    dataset = datasets[2]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.DBSCAN(eps=.2, min_samples=5)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[3]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.DBSCAN(eps=.13, min_samples=5)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[4]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.DBSCAN(eps=0.145, min_samples=6)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[5]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    algorithm = cluster.DBSCAN(eps=0.11, min_samples=3)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[6]
    X, y = dataset
    algorithm = cluster.DBSCAN(eps=0.165, min_samples=3)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    dataset = datasets[7]
    X, y = dataset
    algorithm = cluster.DBSCAN(eps=0.9, min_samples=4)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    res.append((X, y_pred, y))

    return res


def evaluate_RNBC_UPPER():
    res = []

    dataset = datasets[0]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 50, t=0.3, alpha=0.9, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[1]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 27, t=0.3, alpha=0.5, use_lower=False)
    classes = classes - 1
    res.append((X, classes, y))
    
    dataset = datasets[2]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 50, t=0.3, alpha=0.5, use_lower=False)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[3]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 8, t=0.3, alpha=0.5, use_lower=False)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[4]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 4, t=0.025, alpha=0.5, use_lower=False)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[5]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 15, t=0.3, alpha=0.5, use_lower=False)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[6]
    X, y = dataset
    classes = RNBC(X, 6, t=0.3, alpha=0.1, use_lower=False)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[7]
    X, y = dataset
    classes = RNBC(X, 30, t=0.3, alpha=0.1, use_lower=False)
    classes = classes - 1
    res.append((X, classes, y))

    return res

def evaluate_RNBC_LOWER():
    res = []

    dataset = datasets[0]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 50, t=0.3, alpha=0.9, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))
    
    dataset = datasets[1]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 27, t=0.3, alpha=0.5, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))
    
    dataset = datasets[2]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 50, t=0.3, alpha=0.5, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))
    
    dataset = datasets[3]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 8, t=0.3, alpha=0.5, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[4]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 4, t=0.025, alpha=0.3, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[5]
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    classes = RNBC(X, 15, t=0.3, alpha=0.5, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))

    dataset = datasets[6]
    X, y = dataset
    classes = RNBC(X, 6, t=0.3, alpha=0.5, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))


    dataset = datasets[7]
    X, y = dataset
    classes = RNBC(X, 30, t=0.3, alpha=0.5, use_lower=True)
    classes = classes - 1
    res.append((X, classes, y))

    return res

res_res = []

########################################################################
# Run evaluation, some methods might generate warnings, we will catch them •ᴗ•

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning",
        category=FutureWarning,
    )


    res = evaluate_MiniBatchKMeans()
    res_res.append(res)

    res = evaluate_AgglomerativeClustering()
    res_res.append(res)

    res = evaluate_Spectral()
    res_res.append(res)

    res = evaluate_Gaussian()
    res_res.append(res)

    res = evaluate_DBSCAN()
    res_res.append(res)

    res = evaluate_RNBC_UPPER()
    res_res.append(res)

    res = evaluate_RNBC_LOWER()
    res_res.append(res)

########################################################################
# Plot results

name = ["MiniBatch\nKMeans","Agglomerative\nClustering","Spectral\nClustering","Gaussian\nMixture","DBSCAN","RNBC", "RNBC\n(Only Lower)"]


# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))

plot_num = 1

eval_res_all = []
for i_dataset in range(len(res_res[0])):
    for a in range(len(res_res)):
        (X, y_pred, y) = res_res[a][i_dataset]
        X = StandardScaler().fit_transform(X)
        eval_res = calculate_eval(y, y_pred)
        eval_res_all.append(eval_res)
        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.subplot(len(res_res[0]), len(res_res), plot_num)
        if i_dataset == 0:
            plt.title(name[a], size=10)
        if a == 0:
            plt.ylabel(datasets_names[i_dataset], rotation=90, size=10)
        plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[y_pred])

        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)

        plt.xticks(())
        plt.yticks(())
        plot_num += 1

########################################################################
# Print results to console in the latex-style table
datasets_names = ['Blobs Separated',
    'Noisy Circles',
    'Noisy Moons',
    'Anisotropic',
    'Blobs',
    'Three Densities',
    'WingNut Horizontal',
    'Target (Outliers)']
name = ["MiniBatch\nKMeans","Agglomerative\nClustering","Spectral\nClustering","Gaussian\nMixture","DBSCAN","RNBC", "RNBC\n(Only Lower)"]

method_name_id = 0
dataset_name_id = 0

for a in range(len(eval_res_all)):
    [ars, ami, ho, com, vme, fms] = eval_res_all[a]
    my_name = ""
    if method_name_id == 0:
        my_name = datasets_names[dataset_name_id]
    print(my_name + " & " + name[method_name_id] + " & " + "{:.3f}".format(ars) + ' & ' + "{:.3f}".format(ami) + ' & ' + "{:.3f}".format(ho) + ' & '
          + "{:.3f}".format(com) + ' & ' + "{:.3f}".format(vme) + " & " + "{:.3f}".format(fms) + " \\\\")
    method_name_id = method_name_id + 1
    if method_name_id >= len(name):
        method_name_id = 0
        dataset_name_id = dataset_name_id + 1

########################################################################
# Show plot
plt.show()
