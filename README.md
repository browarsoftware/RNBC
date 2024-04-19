### Source codes for paper:

# Rough neighborhood graph: a method for proximity modeling and data clustering 

Authors: [Tomasz Hachaj](https://home.agh.edu.pl/~thachaj/),
[Jarosław Wąs](https://home.agh.edu.pl/~jarek/)

In this paper we will introduce a novel concept of Rough distance graph as a tool for neighborhood modeling and data clustering. In contrary to previous researches we decided to use not directed by rather undirected graph which will be a convenient structure to perform community discovery algorithms. Also the approach proposed in this paper is is closer to spectral clustering from the perspective of dataset representation in as the graph structure. Our approach is also not a variation of rough fuzzy KMenas, the detected clusters are not concentric as well as we do not need to define a specific number of clusters we want to detect. We also use rough set inspired framework to represent proximity relation between objects pairs in the dataset. Our definition of neighborhood is distance-based, not approximation-based. Due to this fact we are able to process any dataset which objects reside in metrical space (there is a defined metric which allows objects pairwise comparison), not only categorical. We have validated our approach on various benchmark dataset achieving nearly perfect clustering results which overcome limitations of other popular algorithms. All required data and source codes of proposed method can be downloaded from online repository and the results presented in this paper can be reproduced.


Keywords: Rough sets, Clustering, Neighborhood approximation, Neighborhood graph

## Requirements

- Python >= 3.8
- scikit-learn >= 1.22
- numpy >= 1.24

## How to run

Run script [run.py](run.py), it will compare RNBC 
with other popular clustering algorithms on benchmark datasets.

![res.jpeg](res.jpeg)